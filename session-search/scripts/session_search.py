#!/usr/bin/env python3
"""session-search — hybrid (BM25 + vector) поиск по сессиям AI-агентов.

Единый CLI для базы знаний: индексация и поиск по сессиям kimi-code,
kimi-cli, claude code и omp. SQLite: FTS5 (BM25) + sqlite-vec (BGE-M3
embeddings через заданный endpoint), как в tools/skills-search.py.

Команды:
    session-search index [--max-new-chunks N] [--db PATH]
    session-search search QUERY [--agent X] [--workdir P] [--mode M]
                                [--top-k N] [--since YYYY-MM-DD] [--db PATH]
    session-search status [--db PATH]
    session-search reindex [--max-new-chunks N] [--db PATH]
    session-search list [--agent X] [--empty] [--db PATH]   # сессии; --empty — источники без чанков
    session-search tail SESSION_ID [--agent X] [-n N] [--db PATH]

Инкрементальность: у каждого источника хранится fingerprint (mtime_ns|size),
переиндексируются только новые/изменённые источники. `index` ограничен
бюджетом новых эмбеддингов за один вызов (--max-new-chunks, по умолчанию
200), чтобы вызов гарантированно укладывался в таймауты MCP-клиентов:
если осталась работа, команда печатает remaining_chunks > 0 — можно
вызвать ещё раз (принцип «агент решает»; никаких демонов и фоновых задач).

Выход команд — единственный JSON-объект в stdout (прогресс — в stderr):
венгерский паттерн для MCP-обёртки (см. mcp_server.py), которая читает
stdout, пока не получит валидный JSON, и убивает процесс (известный
shutdown-hang Python-процессов на Windows).

Конкурентный доступ двух MCP-серверов: search/status/list/tail открывают
SQLite read-only (URI mode=ro, fallback rw + query_only; БД при этом не
создаётся), index/reindex пишут и сериализуются файловым маркером
.rebuild.lock (pid|time, O_EXCL) — параллельная пересборка корректно
отказывает второму процессу вместо гонки open+unlink.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx

# Поисковые примитивы — общие с tools/skills-search.py (tools/search_core.py).
# При запуске как скрипт своей директории нет в sys.path — добавляем явно.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from search_core import (  # noqa: E402
    EMBEDDING_DIM,
    RERANKER_TOP_N,
    _is_cyrillic,
    embed_texts,
    fts_column_query,
    init_vec_conn,
    rerank_from_db as _rerank,
    rrf_fuse as _rrf_fuse,
    serialize_f32,
    stem_query,
    stem_text,
)
from skills_lib import _MODES, drop_db_files as _drop_db_files  # noqa: E402  (общие enum; WAL/SHM-уборка)

HOME = Path.home()
SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = SKILL_DIR / ".index" / "index.db"

# На Windows при перенаправлении stdout в pipe кодировка может быть cp1251 —
# фиксируем UTF-8, чтобы JSON с кириллицей не ломался на стороне MCP-обёртки.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Конфигурация ------------------------------------------------------------

CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200
DEFAULT_BUDGET = 200

# Блокировка перестроения (index/reindex): файловый маркер .rebuild.lock рядом
# с БД. Сериализует два MCP-сервера (kimi-code и kimi-cli) без гонок
# open+unlink на Windows. Маркер считается живым, пока жив его владелец ИЛИ
# он моложе LOCK_STALE_SEC (сирота от упавшего процесса перехватывается).
REBUILD_LOCK_NAME = ".rebuild.lock"
LOCK_STALE_SEC = 1800

# tail больших wire.jsonl (kimi-cli до 282 МБ): чтение блоками с конца
# вместо полного прохода. До TAIL_BIG_FILE байт — обычный разбор.
TAIL_BIG_FILE = 5 * 1024 * 1024
TAIL_BLOCK = 64 * 1024          # шаг окна чтения с конца
TAIL_AVG_MSG_BYTES = 2048       # оценка среднего размера сообщения
TAIL_WINDOW_MAX = 64 * 1024 * 1024  # потолок окна (не читать файл целиком)

AGENT_ROOTS = {
    "kimi-code": HOME / ".kimi-code" / "sessions",
    "kimi-cli": HOME / ".kimi" / "sessions",
    "claude": HOME / ".claude" / "projects",
    "omp": HOME / ".omp" / "agent" / "history.db",
}

# Журнальные и синтетические сообщения kimi-code, которые не должны входить
# в индекс (system-reminder, skill-loaded и т.п. попадают в wire как user).
SYNTHETIC_PREFIXES = ("<system", "<skill-loaded", "<cron-fire")

# Шаблонный мусор, который не индексируем вообще: компактификация контекста,
# служебные вставки и слишком короткие сообщения.
_JUNK_RE = re.compile(r"context has been compacted", re.IGNORECASE)
_MIN_TEXT_LEN = 30


def _is_junk(text: str) -> bool:
    t = text.strip()
    if len(t) < _MIN_TEXT_LEN:
        return True
    if t.startswith(SYNTHETIC_PREFIXES):
        return True
    return bool(_JUNK_RE.search(t))


# --- Утилиты -----------------------------------------------------------------

def _ts_to_epoch(ts) -> float | None:
    """timestamp → epoch (сек). Отличаем мс от с по величине."""
    if isinstance(ts, bool) or ts is None:
        return None
    try:
        f = float(ts)
    except (TypeError, ValueError):
        return None
    if f > 1e11:  # миллисекунды
        f /= 1000.0
    return f


def _iso_to_epoch(s: str) -> float | None:
    if not isinstance(s, str) or not s:
        return None
    s2 = s.strip().replace("Z", "+00:00")
    try:
        from datetime import datetime
        return datetime.fromisoformat(s2).timestamp()
    except ValueError:
        return None


def _epoch_to_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_synthetic(text: str) -> bool:
    t = text.lstrip()
    return t.startswith(SYNTHETIC_PREFIXES)


def chunk_messages(messages: list[dict]) -> list[dict]:
    """Разбить сообщения на чанки ~CHUNK_MAX_CHARS (с перекрытием для длинных)."""
    chunks = []

    def split_long(text: str, role: str, ts) -> None:
        if len(text) <= CHUNK_MAX_CHARS:
            chunks.append({"role": role, "ts": ts, "text": text})
            return
        step = CHUNK_MAX_CHARS - CHUNK_OVERLAP_CHARS
        for i in range(0, len(text), step):
            piece = text[i:i + CHUNK_MAX_CHARS]
            if len(piece.strip()) < 20:
                continue
            chunks.append({"role": role, "ts": ts, "text": piece})

    for m in messages:
        t = _clean_text(m["text"])
        if not t or _is_junk(t):
            continue
        split_long(t, m["role"], m["ts"])
    return chunks


# --- Адаптеры агентов --------------------------------------------------------

def _kimi_code_parse_lines(lines: Iterator[str]) -> list[dict]:
    """kimi-code wire.jsonl (строки) → сообщения (user/assistant/tool).

    Index только context.append_message (user, без синтетики) +
    content.part text из loop-событий (потоковые ответы ассистента, в т.ч.
    финальные — append_message для assistant kimi-code не дублирует) +
    имена tool.call. Тела tool.result и think-цепочки пропускаем.
    Работает и на хвосте файла (окно с конца): первая частичная строка
    уже отброшена вызывающим.
    """
    msgs: list[dict] = []
    buf: list[str] = []
    buf_ts = None

    def flush_buf():
        nonlocal buf, buf_ts
        if buf:
            text = _clean_text("\n".join(buf))
            if text:
                msgs.append({"role": "assistant", "ts": buf_ts, "text": text})
        buf, buf_ts = [], None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = rec.get("type")
        if etype == "context.append_message":
            flush_buf()
            m = rec.get("message", {})
            if m.get("role") == "user":
                texts = [b.get("text", "") for b in m.get("content", [])
                         if isinstance(b, dict) and b.get("type") == "text"]
                text = _clean_text("\n".join(texts))
                if text and not _is_synthetic(text):
                    msgs.append({"role": "user", "ts": _ts_to_epoch(rec.get("time")), "text": text})
            elif m.get("role") == "assistant":
                texts = [b.get("text", "") for b in m.get("content", [])
                         if isinstance(b, dict) and b.get("type") == "text"]
                text = _clean_text("\n".join(texts))
                if text:
                    msgs.append({"role": "assistant", "ts": _ts_to_epoch(rec.get("time")), "text": text})
        elif etype == "context.append_loop_event":
            ev = rec.get("event", {})
            if ev.get("type") == "content.part":
                part = ev.get("part", {})
                if part.get("type") == "text":
                    buf.append(part.get("text", ""))
                    buf_ts = _ts_to_epoch(rec.get("time")) or buf_ts
            elif ev.get("type") == "tool.call":
                flush_buf()
                msgs.append({"role": "tool", "ts": _ts_to_epoch(rec.get("time")),
                             "text": f"tool: {ev.get('name', '?')}"})
            elif ev.get("type") == "tool.result":
                flush_buf()  # тело результата не индексируем
    flush_buf()
    return msgs


def _kimi_code_parse_wire(wire_path: Path) -> list[dict]:
    try:
        with wire_path.open(encoding="utf-8", errors="replace") as f:
            return _kimi_code_parse_lines(f)
    except OSError:
        return []


def kimi_code_sources(root: Path) -> Iterator[tuple[str, Path, dict]]:
    """(agent, src_path, session_meta) для каждого wire.jsonl kimi-code.

    Индексируем основную сессию (agents/main, role='main') и сабагентов
    (agents/agent-N, role='subagent') — единая политика: сессии сабагентов
    в индексе, с меткой role. session_id сабагента — <сессия>/<агент>.
    """
    for wire in sorted(root.glob("wd_*/session_*/agents/*/wire.jsonl")):
        agent_dir = wire.parent.name
        sdir = wire.parent.parent.parent  # wd_*/session_*/
        role = "main" if agent_dir == "main" else "subagent"
        state = {}
        st = sdir / "state.json"
        if st.exists():
            try:
                state = json.loads(st.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                state = {}
        meta = {
            "session_id": sdir.name if role == "main" else f"{sdir.name}/{agent_dir}",
            "title": state.get("title") or state.get("lastPrompt") or None,
            "work_dir": state.get("cwd"),
            "first_ts": _ts_to_epoch(state.get("createdAt")),
            "last_ts": _ts_to_epoch(state.get("updatedAt")),
            "role": role,
        }
        yield "kimi-code", wire, meta


def _kimi_cli_workdir_map() -> dict[str, str]:
    m: dict[str, str] = {}
    meta_file = HOME / ".kimi" / "kimi.json"
    if meta_file.exists():
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        for wd in data.get("work_dirs", []):
            path = wd.get("path", "")
            kaos = wd.get("kaos", "local")
            h = (hashlib.md5(path.encode()).hexdigest() if kaos == "local"
                 else f"{kaos}_{hashlib.md5(path.encode()).hexdigest()}")
            m[h] = path
    return m


def _kimi_cli_parse_lines(lines: Iterator[str]) -> list[dict]:
    """kimi-cli wire.jsonl (строки) → сообщения (агрегация TurnBegin..TurnEnd).

    TurnBegin (payload.user_input) → user; ContentPart (payload.text)
    между TurnBegin/TurnEnd аккумулируются в один assistant-чанк на ход
    (стриминг делится на много ContentPart). Работает и на хвосте файла.
    """
    msgs: list[dict] = []
    cur_assistant: list[str] = []
    cur_ts = None

    def flush():
        nonlocal cur_assistant, cur_ts
        if cur_assistant:
            text = _clean_text("\n".join(cur_assistant))
            if text:
                msgs.append({"role": "assistant", "ts": cur_ts, "text": text})
        cur_assistant, cur_ts = [], None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = rec.get("message", {})
        mtype = m.get("type")
        ts = _ts_to_epoch(rec.get("timestamp"))
        if mtype == "TurnBegin":
            flush()
            ui = m.get("payload", {}).get("user_input", [])
            texts = [p.get("text", "") for p in ui if isinstance(p, dict)]
            text = _clean_text("\n".join(texts))
            if text:
                msgs.append({"role": "user", "ts": ts, "text": text})
        elif mtype == "ContentPart":
            payload = m.get("payload", {})
            if isinstance(payload, dict) and payload.get("text"):
                cur_assistant.append(payload["text"])
                cur_ts = ts or cur_ts
        elif mtype == "TurnEnd":
            flush()
    flush()
    return msgs


def _kimi_cli_parse_wire(wire_path: Path) -> list[dict]:
    try:
        with wire_path.open(encoding="utf-8", errors="replace") as f:
            return _kimi_cli_parse_lines(f)
    except OSError:
        return []


def kimi_cli_sources(root: Path) -> Iterator[tuple[str, Path, dict]]:
    """(agent, src_path, session_meta): основная сессия + сабагенты.

    Основная — <hash>/<uuid>/wire.jsonl (role='main'); сабагенты —
    <uuid>/subagents/<id>/wire.jsonl (role='subagent', 182 файла на этой
    машине), мета — из meta.json (description/created_at/updated_at).
    session_id сабагента — <сессия>/<сабагент> (без коллизий с main).
    """
    hash_map = _kimi_cli_workdir_map()
    session_dirs: list[tuple[Path, str | None]] = []
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "state.json").exists() or (child / "wire.jsonl").exists():
                session_dirs.append((child, hash_map.get(child.name)))
            else:
                wd = hash_map.get(child.name)
                for sdir in child.iterdir():
                    if sdir.is_dir() and (sdir / "wire.jsonl").exists():
                        session_dirs.append((sdir, wd))
    for sdir, wd in session_dirs:
        state = {}
        st = sdir / "state.json"
        if st.exists():
            try:
                state = json.loads(st.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                state = {}
        title = state.get("custom_title")
        if title and title.startswith("<"):
            title = None  # в custom_title попадают вставки вида `<git-context> ...`
        meta = {
            "session_id": sdir.name,
            "title": title,
            "work_dir": wd,
            "first_ts": None,
            "last_ts": None,
            "role": "main",
        }
        yield "kimi-cli", sdir / "wire.jsonl", meta
        subs = sdir / "subagents"
        if not subs.is_dir():
            continue
        for swire in sorted(subs.glob("*/wire.jsonl")):
            sub_meta = {"description": None, "created_at": None, "updated_at": None}
            mj = swire.parent / "meta.json"
            if mj.exists():
                try:
                    sub_meta = json.loads(mj.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    sub_meta = {}
            sub_title = sub_meta.get("description")
            prompt = swire.parent / "prompt.txt"
            if not sub_title and prompt.exists():
                try:
                    first = prompt.read_text(encoding="utf-8",
                                             errors="replace").strip().splitlines()
                    sub_title = first[0][:120] if first else None
                except OSError:
                    pass
            yield "kimi-cli", swire, {
                "session_id": f"{sdir.name}/{swire.parent.name}",
                "title": sub_title,
                "work_dir": wd,
                "first_ts": _ts_to_epoch(sub_meta.get("created_at")),
                "last_ts": _ts_to_epoch(sub_meta.get("updated_at")),
                "role": "subagent",
            }


def _claude_parse_transcript(path: Path) -> tuple[list[dict], dict]:
    """claude .jsonl → (сообщения, session_meta)."""
    msgs: list[dict] = []
    meta: dict = {"session_id": path.stem, "work_dir": None, "first_ts": None, "last_ts": None}
    first_user = None
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rtype = rec.get("type")
                ts = _iso_to_epoch(rec.get("timestamp"))
                if meta["first_ts"] is None and ts:
                    meta["first_ts"] = ts
                if ts:
                    meta["last_ts"] = ts
                if meta["work_dir"] is None:
                    meta["work_dir"] = rec.get("cwd")
                if rec.get("sessionId"):
                    meta["session_id"] = rec["sessionId"]
                if rtype in ("user", "assistant"):
                    content = rec.get("message", {}).get("content", [])
                    if not isinstance(content, list):
                        content = []
                    texts, tools = [], []
                    for blk in content:
                        if not isinstance(blk, dict):
                            continue
                        bt = blk.get("type")
                        if bt == "text" and blk.get("text"):
                            texts.append(blk["text"])
                        elif bt == "tool_use" and blk.get("name"):
                            tools.append(blk["name"])
                    if rtype == "user":
                        joined = _clean_text("\n".join(texts))
                        if joined and first_user is None:
                            first_user = joined[:80]
                        # tool_result-блоки не индексируем (шум)
                        if joined:
                            msgs.append({"role": "user", "ts": ts, "text": joined})
                    else:
                        if texts:
                            msgs.append({"role": "assistant", "ts": ts,
                                         "text": _clean_text("\n".join(texts))})
                        for name in tools:
                            msgs.append({"role": "tool", "ts": ts, "text": f"tool: {name}"})
                elif rtype == "summary" and rec.get("summary"):
                    msgs.append({"role": "assistant", "ts": ts,
                                 "text": "[summary] " + _clean_text(rec["summary"])})
    except OSError:
        pass
    if meta["first_ts"] and meta["last_ts"] and meta["first_ts"] > meta["last_ts"]:
        meta["first_ts"], meta["last_ts"] = meta["last_ts"], meta["first_ts"]
    meta["title"] = first_user
    return msgs, meta


def claude_sources(root: Path) -> Iterator[tuple[str, Path, dict]]:
    if root.exists():
        for path in sorted(root.glob("**/*.jsonl")):
            if path.name == "history.jsonl":
                continue
            msgs, meta = _claude_parse_transcript(path)
            # кэш метаданных: пропарсенные сообщения уезжают с метой —
            # _parse_source переиспользует их (раньше cmd_index парсил тот же
            # .jsonl дважды: здесь и в _parse_source)
            meta["_msgs"] = msgs
            yield "claude", path, meta


def omp_sources(db_path: Path) -> Iterator[tuple[str, Path, dict]]:
    """omp history.db → (agent, src, meta): одна запись НА СЕССИЮ, не на строку
    истории. src — БД, читается только при индексации."""
    if not db_path.exists():
        return
    for meta in _omp_sessions(db_path):
        yield "omp", db_path, meta


def _omp_read(db_path: Path) -> Iterator[dict]:
    """Строки history.db omp (session_id, prompt, created_at, cwd) — как есть."""
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return
    try:
        rows = conn.execute(
            "SELECT session_id, prompt, created_at, cwd FROM history ORDER BY id"
        ).fetchall()
    except sqlite3.Error:
        rows = []
        conn.close()
        return
    conn.close()
    for session_id, prompt, created_at, cwd in rows:
        yield {
            "session_id": session_id or "?",
            "title": None,
            "work_dir": cwd,
            "first_ts": None,
            "last_ts": None,
            "_prompt": prompt,
            "_ts": _ts_to_epoch(created_at),
        }


def _omp_sessions(db_path: Path) -> list[dict]:
    """history.db → одна мета на сессию (агрегация строк по session_id).

    Метданные — из первой строки сессии (cwd), границы времени — min/max
    created_at; текст — сконкатенированные реплики (junk-строки отсеяны).
    Раньше каждая строка истории считалась отдельной «сессией»: 48 источников
    из 5 реальных, коллизии ключей omp::sid, реальные сессии терялись
    junk-фильтром.
    """
    by_sid: dict[str, list[dict]] = {}
    for r in _omp_read(db_path):
        by_sid.setdefault(r["session_id"], []).append(r)
    out = []
    for sid, rows in by_sid.items():
        ts = [r["_ts"] for r in rows if r["_ts"]]
        msgs = []
        for r in rows:
            text = _clean_text(r["_prompt"] or "")
            if text and not _is_junk(text):
                msgs.append({"role": "user", "ts": r["_ts"], "text": text})
        out.append({
            "session_id": sid,
            "title": None,
            "work_dir": rows[0]["work_dir"],
            "first_ts": min(ts) if ts else None,
            "last_ts": max(ts) if ts else None,
            "_msgs": msgs,
        })
    return out


def _parse_source(agent: str, src: Path, meta: dict | None = None) -> tuple[list[dict], dict]:
    """→ (messages, session_meta по сообщениям)."""
    meta = dict(meta or {})
    if agent == "kimi-code":
        msgs = _kimi_code_parse_wire(src)
    elif agent == "kimi-cli":
        msgs = _kimi_cli_parse_wire(src)
    elif agent == "claude":
        # переиспользование кэша метаданных claude_sources (meta["_msgs"]):
        # повторный разбор .jsonl только если кэша нет (cmd_tail, тесты)
        cached = meta.pop("_msgs", None)
        if cached is not None:
            msgs = cached
        else:
            msgs, m2 = _claude_parse_transcript(src)
            meta.update(m2)
    elif agent == "omp":
        # одна сессия приходит сверху (collect_sources/omp_sources); кэш _msgs
        # агрегирован в _omp_sessions, fallback — прямое чтение конкретной сессии
        sid = meta.get("session_id")
        msgs = meta.pop("_msgs", None)
        if msgs is None:
            msgs = [{"role": "user", "ts": r["_ts"],
                     "text": _clean_text(r["_prompt"] or "")}
                    for r in _omp_read(src) if r["session_id"] == sid]
        meta["msg_count"] = len(msgs)
        return msgs, meta
    else:  # pragma: no cover
        msgs = []
    ts = [m["ts"] for m in msgs if m.get("ts")]
    if ts:
        if meta.get("first_ts") in (None,):
            meta["first_ts"] = min(ts)
        meta["last_ts"] = max(ts)
    meta["msg_count"] = len(msgs)
    return msgs, meta


# --- Сбор источников ---------------------------------------------------------

def _fingerprint(path: Path) -> str:
    try:
        st = path.stat()
    except OSError:
        return ""
    return f"{st.st_mtime_ns}|{st.st_size}"


def collect_sources() -> list[dict]:
    """Все источники сессий: {agent, src_path, fingerprint, meta}."""
    out: list[dict] = []
    for agent, root in AGENT_ROOTS.items():
        if agent == "omp":
            if root.exists():
                # у omp N сессий в одном history.db — одна запись на сессию,
                # с синтетическим ключом agent::session_id (fingerprint — от БД)
                for _, src, meta in omp_sources(root):
                    out.append({
                        "agent": agent,
                        "src_path": str(root),
                        "key": f"omp::{meta['session_id']}",
                        "fingerprint": _fingerprint(root),
                        "meta": meta,
                    })
            continue
        if not root.exists():
            continue
        if agent == "kimi-code":
            gen = kimi_code_sources(root)
        elif agent == "kimi-cli":
            gen = kimi_cli_sources(root)
        else:
            gen = claude_sources(root)
        for ag, src, meta in gen:
            if not src.exists():
                continue
            out.append({
                "agent": ag,
                "src_path": str(src),
                "key": f"{ag}::{src}",
                "fingerprint": _fingerprint(src),
                "meta": meta,
            })
    return out


# --- БД ----------------------------------------------------------------------

def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _has_session_role(conn: sqlite3.Connection) -> bool:
    """Колонка sessions.role появилась в фазе 7: ro-соединение в старом индексе
    её не имеет (ALTER возможен только при записи — происходит при index)."""
    return "role" in _table_columns(conn, "sessions")


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Read-only подключение для search/status/list/tail.

    URI mode=ro + busy_timeout. БД при этом НЕ создаётся (вызывающие заранее
    проверили exists()). Fallback на обычное подключение с PRAGMA query_only:
    ro-открытие WAL-БД может не выйти (нет -shm / эксклюзивная блокировка
    пишущего процесса).
    """
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA busy_timeout=5000")
        init_vec_conn(conn)
        conn.execute("SELECT COUNT(*) FROM sqlite_master")  # прогрев открытия
        return conn
    except sqlite3.Error:
        if conn is not None:
            conn.close()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA query_only=ON")
        init_vec_conn(conn)
    except Exception:
        conn.close()
        raise
    return conn


def init_db(db_path: Path, read_only: bool = False) -> sqlite3.Connection:
    """rw (по умолчанию, для index/reindex) или ro (search/status/list/tail)."""
    if read_only:
        return _connect_readonly(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    init_vec_conn(conn)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            key TEXT PRIMARY KEY,
            agent TEXT NOT NULL,
            src_path TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            indexed_at REAL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            agent TEXT NOT NULL,
            session_id TEXT NOT NULL,
            src_path TEXT NOT NULL,
            title TEXT,
            work_dir TEXT,
            first_ts REAL,
            last_ts REAL,
            msg_count INTEGER NOT NULL DEFAULT 0,
            role TEXT NOT NULL DEFAULT 'main',
            PRIMARY KEY (agent, session_id)
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT NOT NULL,
            session_id TEXT NOT NULL,
            src_path TEXT NOT NULL,
            role TEXT NOT NULL,
            ord INTEGER NOT NULL DEFAULT 0,
            ts REAL,
            text TEXT NOT NULL,
            stemmed TEXT NOT NULL DEFAULT ''
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            stemmed, agent, session_id,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, stemmed, agent, session_id)
            VALUES (new.id, new.stemmed, new.agent, new.session_id);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, stemmed, agent, session_id)
            VALUES ('delete', old.id, old.stemmed, old.agent, old.session_id);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, stemmed, agent, session_id)
            VALUES ('delete', old.id, old.stemmed, old.agent, old.session_id);
            INSERT INTO chunks_fts(rowid, stemmed, agent, session_id)
            VALUES (new.id, new.stemmed, new.agent, new.session_id);
        END;
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)

    # Миграция фаз 4→7: колонка role (main/subagent) у sessions. Старый
    # индекс без неё доезжает сюда на первом rw-открытии (ALTER); ro-пути
    # (`_has_session_role`) до тех пор возвращают 'main' как default.
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN role TEXT NOT NULL DEFAULT 'main'")
    except sqlite3.OperationalError:
        pass  # колонка уже есть

    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)
    except Exception:
        pass

    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


# --- Блокировка перестроения (index/reindex) ----------------------------------

def _pid_alive(pid) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError, OverflowError):
        return False
    return True


def _read_lock(lock: Path) -> tuple[int | None, float]:
    try:
        pid_s, ts_s = lock.read_text(encoding="utf-8").strip().split("|", 1)
        return int(pid_s), float(ts_s)
    except (OSError, ValueError, IndexError):
        return None, 0.0


def _acquire_rebuild_lock(db_path: Path) -> Path | None:
    """Эксклюзивный маркер .rebuild.lock (pid|time) рядом с БД.

    Параллельные index/reindex двух MCP-серверов: создание O_EXCL атомарно,
    проигравший получает JSON-отказ (ok=false, locked=true) и выходит.
    Маркер-сирота (владелец мёртв И старше LOCK_STALE_SEC) перехватывается.
    """
    lock = db_path.parent / REBUILD_LOCK_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)  # fresh install: .index/ может не существовать
    if lock.exists():
        pid, _ = _read_lock(lock)
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            age = 0.0
        if _pid_alive(pid) or age < LOCK_STALE_SEC:
            print(json.dumps({
                "ok": False,
                "error": f"перестроение уже идёт: {lock.name} от pid {pid}, "
                         f"возраст {age / 60:.1f} мин — подожди и повтори",
                "locked": True,
            }, ensure_ascii=False))
            return None
        try:
            lock.unlink()
        except OSError:  # кто-то перехватил раньше
            return None
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(json.dumps({
            "ok": False,
            "error": f"перестроение уже идёт: {lock.name} создан параллельным процессом",
            "locked": True,
        }, ensure_ascii=False))
        return None
    except OSError as e:
        print(json.dumps({
            "ok": False,
            "error": f"не создать маркер {lock}: {e}",
            "locked": False,
        }, ensure_ascii=False))
        return None
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()}|{time.time()}")
    return lock


def _release_rebuild_lock(lock: Path) -> None:
    try:
        lock.unlink(missing_ok=True)
    except OSError:
        pass


def _meta_get(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )


# --- Индексация --------------------------------------------------------------

def _prune_chunks(conn: sqlite3.Connection, sources: list[dict]) -> None:
    """Удалить чанки/сессии пропавших источников.

    У всех агентов кроме omp один src_path на сессию — чистим по src_path.
    У omp N сессий живут в одном history.db (общий src_path), поэтому чанки
    исчезнувших omp-сессий чистим по session_id: сам путь остаётся живым,
    пока существует history.db.
    """
    conn.execute(
        "DELETE FROM chunks WHERE agent != 'omp' AND src_path NOT IN "
        "(SELECT src_path FROM sources)"
    )
    live_omp = {s["meta"]["session_id"] for s in sources if s["agent"] == "omp"}
    dead_omp = [
        sid for (sid,) in conn.execute(
            "SELECT DISTINCT session_id FROM chunks WHERE agent = 'omp'")
        if sid not in live_omp
    ]
    for i in range(0, len(dead_omp), 500):  # лимит хост-параметров SQLite
        batch = dead_omp[i:i + 500]
        ph = ",".join("?" * len(batch))
        conn.execute(
            f"DELETE FROM chunks WHERE agent = 'omp' AND session_id IN ({ph})",
            batch,
        )
    conn.execute("DELETE FROM chunks_vec WHERE id NOT IN (SELECT id FROM chunks)")
    conn.execute(
        "DELETE FROM sessions WHERE (agent, session_id) NOT IN "
        "(SELECT agent, session_id FROM chunks GROUP BY agent, session_id)"
    )


def _wipe_source(conn: sqlite3.Connection, agent: str, session_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE agent=? AND session_id=?", (agent, session_id))
    conn.execute("DELETE FROM chunks_vec WHERE id NOT IN (SELECT id FROM chunks)")
    conn.execute("DELETE FROM sessions WHERE agent=? AND session_id=?", (agent, session_id))


def _store_parsed(conn: sqlite3.Connection, src: dict, msgs: list[dict], meta: dict,
                  chunks: list[dict] | None = None) -> tuple[int, str]:
    """Записать уже распарсенную сессию в БД (чанки + эмбеддинги + мета).

    chunks=None — нарезать из msgs прямо здесь; cmd_index передаёт готовые
    (чанковка один раз на источник: бюджет остатка считает по ним же).
    Порядок: (a) чанки собираются в память; (b) эмбеддинги считаются ДО
    первых записей — при сбое embedding источник не тронут (остаётся stale
    и переиндексируется в следующий заход); (c) при успехе — одна атомарная
    транзакция (SAVEPOINT): wipe старых чанков/строки сессии + вставка
    новых + commit.
    """
    agent = src["agent"]
    path = Path(src["src_path"])
    session_id = meta.get("session_id") or path.stem or src["key"]

    if chunks is None:
        chunks = chunk_messages(msgs)
    embeddings = embed_texts([c["text"] for c in chunks]) if chunks else []

    conn.execute("SAVEPOINT _store_parsed")
    try:
        _wipe_source(conn, agent, session_id)
        for i, c in enumerate(chunks):
            cur = conn.execute(
                "INSERT INTO chunks(agent, session_id, src_path, role, ord, ts, text, stemmed) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (agent, session_id, str(path), c["role"], i, c.get("ts"),
                 c["text"], stem_text(c["text"])),
            )
            cid = cur.lastrowid
            conn.execute(
                "INSERT INTO chunks_vec(id, embedding) VALUES(?, ?)",
                (cid, serialize_f32(embeddings[i])),
            )

        ts = [c["ts"] for c in chunks if c.get("ts")]
        first_ts, last_ts = meta.get("first_ts"), meta.get("last_ts")
        if first_ts is None and ts:
            first_ts = min(ts)
        if last_ts is None and ts:
            last_ts = max(ts)
        conn.execute(
            "INSERT OR REPLACE INTO sessions(agent, session_id, src_path, title, work_dir, first_ts, last_ts, msg_count, role) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (agent, session_id, str(path), meta.get("title"), meta.get("work_dir"),
             first_ts, last_ts, len(chunks), meta.get("role", "main")),
        )
        conn.execute(
            "INSERT OR REPLACE INTO sources(key, agent, src_path, fingerprint, chunk_count, indexed_at) "
            "VALUES(?,?,?,?,?,?)",
            (src["key"], agent, str(path), src["fingerprint"], len(chunks), time.time()),
        )
    except Exception:
        conn.execute("ROLLBACK TO _store_parsed")
        conn.execute("RELEASE _store_parsed")
        raise
    conn.execute("RELEASE _store_parsed")
    conn.commit()
    return len(chunks), session_id


def cmd_index(args, _lock_held: bool = False, _print_result: bool = True) -> dict:
    """Инкрементальная (до)индексация с бюджетом. Сериализована маркером
    .rebuild.lock (два MCP-сервера), _lock_held=True — маркер держит
    вызывающий (cmd_reindex).

    Возвращает payload-словарь; печатает его только при _print_result=True
    (cmd_reindex глушит внутреннюю печать и печатает свой итог).
    Маркер освобождается ДО печати JSON: MCP-обёртка (run_cli) убивает
    процесс сразу после чтения JSON — освобождение после печати оставляло
    сиротский .rebuild.lock на каждый вызов (баг фазы 7 аудита р3).
    """
    db_path = Path(args.db).resolve()
    budget = max(1, int(args.max_new_chunks))
    lock = None
    if not _lock_held:
        lock = _acquire_rebuild_lock(db_path)
        if lock is None:
            return {}
    try:
        conn = init_db(db_path)

        sources = collect_sources()
        stored = {r[0]: r[1] for r in conn.execute("SELECT key, fingerprint FROM sources")}

        # Удаляем пропавшие
        for key in stored:
            if key not in {s["key"] for s in sources}:
                conn.execute("DELETE FROM sources WHERE key=?", (key,))
        _prune_chunks(conn, sources)

        # Изменённые/новые — сначала свежие (по mtime источника, desc)
        stale = [s for s in sources if stored.get(s["key"]) != s["fingerprint"]]
        stale.sort(key=lambda s: _src_mtime(s), reverse=True)

        # Парсим и чанкуем заранее (оба прохода один раз на источник;
        # раньше chunk_messages гонялся повторно и здесь, и в _store_parsed)
        planned: list[tuple[dict, list[dict], list[dict], dict]] = []
        for s in stale:
            try:
                msgs, meta = _parse_source(s["agent"], Path(s["src_path"]), s["meta"])
                planned.append((s, msgs, chunk_messages(msgs), meta))
            except Exception as e:
                print(f"parse failed {s['agent']} {s['src_path']}: {e}", file=sys.stderr)

        indexed_chunks = 0
        remaining_chunks = 0
        indexed_sources = 0
        error = None

        for idx, (s, msgs, chunks, meta) in enumerate(planned):
            nchunks = len(chunks)
            # Мягкий бюджет: превышен — источник целиком на следующий заход
            if indexed_chunks >= budget:
                remaining_chunks += nchunks
                continue
            try:
                n, sid = _store_parsed(conn, s, msgs, meta, chunks)
                indexed_chunks += n
                indexed_sources += 1
                print(f"indexed {s['agent']} {sid}: {n} chunks", file=sys.stderr)
            except (httpx.HTTPError, OSError) as e:
                error = f"embedding недоступен: {e}. Повтори позже (keyword-поиск работает и без модели)."
                print(error, file=sys.stderr)
                # текущий и все необработанные источники остаются на следующий заход
                remaining_chunks += nchunks + sum(
                    len(ch) for _, _, ch, _ in planned[idx + 1:])
                break
            except Exception as e:  # не дать одному битому файлу уронить весь проход
                print(f"failed {s['agent']} {s['src_path']}: {e}", file=sys.stderr)
                continue

        if error is None:
            _meta_set(conn, "last_index_at", time.time())
        conn.commit()

        payload = {
            "ok": error is None,
            "error": error,
            "indexed_sources": indexed_sources,
            "new_chunks": indexed_chunks,
            "remaining_chunks": remaining_chunks,
            "total_sources": len(sources),
            "chunks_total": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "last_index_at": _epoch_to_iso(_meta_get(conn, "last_index_at")),
        }
        conn.close()
    finally:
        if lock is not None:
            _release_rebuild_lock(lock)
    # печать ПОСЛЕ finally: маркер уже освобождён — kill из run_cli сразу
    # за чтением JSON не оставляет сиротский .rebuild.lock
    if _print_result:
        print(json.dumps(payload, ensure_ascii=False))
    return payload


def _src_mtime(s: dict) -> float:
    try:
        return Path(s["src_path"]).stat().st_mtime
    except OSError:
        return 0.0


# --- Поиск -------------------------------------------------------------------

def _search_bm25(conn, query: str, limit: int, agent: str | None, since: float | None,
                 workdir: str | None) -> list[dict]:
    stems = stem_query(query)
    if not stems:
        return []
    # column-filter: match только по тексту чанков (stemmed), иначе агент/
    # session_id FTS-колонки дают phantom-хиты (чат «grafana» матчит агента
    # по имени и т.п.)
    fts_query = fts_column_query(stems, "stemmed")
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    except Exception:
        return []
    return _filter_rows(conn, rows, agent, since, workdir, "bm25")


def _search_vector(conn, query: str, limit: int, agent: str | None, since: float | None,
                   workdir: str | None) -> list[dict]:
    q = embed_texts([query])[0]
    rows = conn.execute(
        "SELECT id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(q), limit),
    ).fetchall()
    out = []
    for cid, dist in rows:
        out.append({"chunk_id": cid, "score": 1.0 - (dist / 2.0), "source": "vector"})
    return _filter_by_id(conn, out, agent, since, workdir)


def _filter_rows(conn, rows, agent, since, workdir, source) -> list[dict]:
    out = []
    for rowid, rank in rows:
        out.append({"chunk_id": rowid, "score": 1.0 / (1.0 + abs(rank)), "source": source})
    return _filter_by_id(conn, out, agent, since, workdir)


def _filter_by_id(conn, items: list[dict], agent: str | None, since: float | None,
                  workdir: str | None) -> list[dict]:
    """Фильтрация кандидатов (agent/since/workdir) ДО среза top_k.

    work_dir лежит в sessions.work_dir — тянем одним batched JOIN-запросом
    (вместо SELECT на каждый чанк). Сравнение workdir-подстроки — в Python:
    SQL lower() не знает кириллицу, а фильтр регистронезависимый.
    """
    if not items or (agent is None and since is None and not workdir):
        return items
    ids = [it["chunk_id"] for it in items]
    allowed = set()
    for i in range(0, len(ids), 500):  # лимит хост-параметров SQLite
        batch = ids[i:i + 500]
        ph = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT c.id, c.agent, c.ts, s.work_dir FROM chunks c "
            f"LEFT JOIN sessions s ON s.agent = c.agent AND s.session_id = c.session_id "
            f"WHERE c.id IN ({ph})",
            batch,
        ).fetchall()
        for cid, ag, ts, wd in rows:
            if agent and ag != agent:
                continue
            if since is not None and (ts is None or ts < since):
                continue
            if workdir and (not wd or workdir.lower() not in wd.lower()):
                continue
            allowed.add(cid)
    return [it for it in items if it["chunk_id"] in allowed]


def cmd_search(args) -> None:
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(json.dumps({"error": "index not found — сначала session_reindex"}, ensure_ascii=False))
        return
    conn = _connect_readonly(db_path)
    query, mode = args.query, args.mode
    # Кламп top_k (находка аудита р4): top_k <= 0 давал LIMIT < 0 = «все
    # строки» в SQLite + мусорный срез results[:top_k] (паритет с skills-search)
    top_k = max(1, int(args.top_k))
    # argparse ограничивает choices только в CLI; программный вызов с чужим
    # mode раньше молча возвращал пустой результат (фаза 1 аудита р3)
    if mode not in _MODES:
        conn.close()
        raise ValueError(f"unknown search mode: {mode!r} (expected {'|'.join(_MODES)})")
    agent = args.agent
    since = _iso_to_epoch(args.since) if args.since else None
    workdir = args.workdir or None

    results: list[dict] = []
    # При фильтрах (агент/воркдир/дата) пул кандидатов до фильтра должен быть
    # большим — иначе топ-кандидаты вне фильтра вытеснят всё. Все фильтры
    # применяются в _filter_by_id ДО среза [:top_k] (раньше workdir
    # фильтровался уже после среза и недозаполнял выдачу).
    filtered = bool(agent or workdir or since is not None)
    pool = top_k * 5 if not filtered else max(500, top_k * 50)
    if mode in ("hybrid", "keyword"):
        results.extend(_search_bm25(conn, query, pool, agent, since, workdir))
    if mode in ("hybrid", "semantic"):
        try:
            results.extend(_search_vector(conn, query, pool, agent, since, workdir))
        except (httpx.HTTPError, RuntimeError):
            if mode == "semantic":
                conn.close()
                print(json.dumps(
                    {"error": "embedding недоступен — используй --mode keyword", "results": []},
                    ensure_ascii=False))
                return

    if mode == "hybrid":
        results = _rrf_fuse(results)
        if getattr(args, "rerank", False):
            # пул реранкера = max(20, top_k), иначе большие --top-k молча обрезались до 20
            results = _rerank(query, results, conn, top_n=max(RERANKER_TOP_N, top_k))
    else:
        seen: dict[int, dict] = {}
        for r in results:
            cid = r["chunk_id"]
            if cid not in seen or r["score"] > seen[cid]["score"]:
                seen[cid] = r
        results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    results = results[:top_k]
    output = []
    # Один batched JOIN вместо SELECT на каждый чанк + SELECT на каждую
    # сессию (N+1): чанки и sessions тянутся по IN (...), порядок — исходный.
    by_id: dict[int, tuple] = {}
    ids = [r["chunk_id"] for r in results]
    for i in range(0, len(ids), 500):  # лимит хост-параметров SQLite
        batch = ids[i:i + 500]
        ph = ",".join("?" * len(batch))
        role_col = "s.role" if _has_session_role(conn) else "'main' AS role"
        rows = conn.execute(
            f"SELECT c.id, c.agent, c.session_id, c.src_path, c.role, c.ts, c.text, "
            f"s.agent AS s_agent, s.title, s.work_dir, s.first_ts, s.last_ts, {role_col} "
            f"FROM chunks c LEFT JOIN sessions s "
            f"ON s.agent = c.agent AND s.session_id = c.session_id "
            f"WHERE c.id IN ({ph})",
            batch,
        ).fetchall()
        for row in rows:
            by_id[row[0]] = row
    for r in results:
        row = by_id.get(r["chunk_id"])
        if not row:
            continue
        # s_agent (row[7]) NOT NULL только если строка сессии существует
        sess = {"title": row[8], "work_dir": row[9],
                "first_ts": row[10], "last_ts": row[11]} if row[7] is not None else None
        # workdir-фильтр уже применён в _filter_by_id до среза top_k
        output.append({
            "agent": row[1],
            "session_id": row[2],
            "title": sess["title"] if sess else None,
            "work_dir": sess["work_dir"] if sess else None,
            "session_start": _epoch_to_iso(sess["first_ts"]) if sess else None,
            "session_end": _epoch_to_iso(sess["last_ts"]) if sess else None,
            "is_subagent": row[12] == "subagent",
            "role": row[4],
            "ts": _epoch_to_iso(row[5]),
            "score": round(r["score"], 4),
            "snippet": row[6][:300].replace("\n", " "),
            "src_path": row[3],
        })
    print(json.dumps({
        "results": output,
        "mode": mode,
        "last_indexed_at": _epoch_to_iso(_meta_get(conn, "last_index_at")),
        "indexed_sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
    }, ensure_ascii=False))
    conn.close()


def cmd_status(args) -> None:
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(json.dumps({"error": "index not found", "exists": False}, ensure_ascii=False))
        return
    conn = _connect_readonly(db_path)
    current = collect_sources()
    stored = {r[0]: r[1] for r in conn.execute("SELECT key, fingerprint FROM sources")}
    new_changed = [s for s in current if stored.get(s["key"]) != s["fingerprint"] and s["key"] in stored]
    brand_new = [s for s in current if s["key"] not in stored]
    missing = [k for k in stored if k not in {s["key"] for s in current}]
    chunks_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    # Здоровье (фаза 7): источники-файлы без единого чанка (молчаливые потери
    # парсинга) и рассинхрон суммы учтённых чанков с фактическим числом.
    files_no_chunks = conn.execute(
        "SELECT COUNT(*) FROM sources WHERE chunk_count=0").fetchone()[0]
    recorded = conn.execute(
        "SELECT COALESCE(SUM(chunk_count), 0) FROM sources").fetchone()[0]
    out = {
        "exists": True,
        "last_index_at": _epoch_to_iso(_meta_get(conn, "last_index_at")),
        "total_sources": len(current),
        "new_sources": len(brand_new),
        "changed_sources": len(new_changed),
        "missing_sources": len(missing),
        "chunks_total": chunks_total,
        "vectors_total": conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0],
        "sessions_total": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        # GROUP BY вместо COUNT(*) на каждого агента (фаза 1 аудита р3)
        "by_agent": (lambda counts: {a: counts.get(a, 0) for a in sorted(AGENT_ROOTS)})(
            dict(conn.execute("SELECT agent, COUNT(*) FROM sessions GROUP BY agent"))),
        "files_no_chunks": files_no_chunks,
        "files_chunks_mismatch": abs(chunks_total - recorded),
        "locked": (db_path.parent / REBUILD_LOCK_NAME).exists(),
        "db_path": str(db_path),
    }
    if _has_session_role(conn):
        by_role = dict(conn.execute("SELECT role, COUNT(*) FROM sessions GROUP BY role"))
        out["sessions_by_role"] = {r: by_role.get(r, 0) for r in ("main", "subagent")}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    conn.close()


def cmd_list(args) -> None:
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(json.dumps({"error": "index not found"}, ensure_ascii=False))
        return
    conn = _connect_readonly(db_path)
    if getattr(args, "empty", False):
        # Источники-файлы без единого чанка (молчаливые потери парсинга или
        # сессии без индексируемых сообщений): status показывает их ЧИСЛО,
        # list --empty — какие именно (фаза 8 аудита р4).
        rows = conn.execute(
            "SELECT agent, src_path, key, indexed_at FROM sources "
            "WHERE chunk_count = 0 ORDER BY agent, src_path"
        ).fetchall()
        print(json.dumps(
            {"empty_sources": [
                {"agent": a, "src_path": p, "key": k, "indexed_at": t}
                for a, p, k, t in rows]},
            ensure_ascii=False))
        conn.close()
        return
    role_col = "role" if _has_session_role(conn) else "'main' AS role"
    q = ("SELECT agent, session_id, title, work_dir, first_ts, last_ts, msg_count, "
         f"{role_col} FROM sessions")
    if args.agent:
        q += " WHERE agent=?"
        rows = conn.execute(q, (args.agent,))
    else:
        rows = conn.execute(q + " ORDER BY agent, last_ts DESC")
    out = []
    for agent, sid, title, wd, fts, lts, mc, role in rows:
        out.append({
            "agent": agent, "session_id": sid, "title": title, "work_dir": wd,
            "first_ts": _epoch_to_iso(fts), "last_ts": _epoch_to_iso(lts),
            "chunks": mc, "is_subagent": role == "subagent",
        })
    print(json.dumps({"sessions": out}, ensure_ascii=False))
    conn.close()


def _locate_session(conn, session_id: str, agent: str | None):
    """→ (agent, src_path, role). role='main' в индексах без колонки role."""
    role_col = "role" if _has_session_role(conn) else "'main' AS role"
    q = f"SELECT agent, src_path, {role_col} FROM sessions WHERE session_id=?"
    rows = conn.execute(q, (session_id,)).fetchall()
    if agent:
        rows = [r for r in rows if r[0] == agent]
    if not rows:
        return None, None, "main"
    return rows[0][0], Path(rows[0][1]), rows[0][2]


# --- tail: чтение больших wire.jsonl с конца ---------------------------------

def _read_wire_tail_lines(path: Path, max_bytes: int) -> list[str]:
    """Последние ~max_bytes файла построчно.

    Окно начинается с середины строки — первая (частичная) строка
    отбрасывается. Файл целиком не читается (tail на 282-МБ wire)."""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        chunk = f.read()
    lines = chunk.decode("utf-8", errors="replace").splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # первая строка окна обрезана
    return lines


def _tail_msgs_from_wire(agent: str, path: Path, n: int) -> list[dict]:
    """Последние N сообщений большого wire.jsonl без полного чтения файла.

    Окно с конца растёт (n * TAIL_AVG_MSG_BYTES, ×4 за итерацию, потолок
    TAIL_WINDOW_MAX), пока в нём не наберётся N сообщений — единственный
    случай неполного хвоста: сообщения настолько длинные, что N штук не
    влезают в потолок окна.
    """
    parse_lines = (_kimi_code_parse_lines if agent == "kimi-code"
                   else _kimi_cli_parse_lines)
    window = max(TAIL_BLOCK, n * TAIL_AVG_MSG_BYTES)
    msgs: list[dict] = []
    while True:
        lines = _read_wire_tail_lines(path, window)
        msgs = parse_lines(lines)
        if len(msgs) >= n or window >= TAIL_WINDOW_MAX:
            return msgs
        window = min(TAIL_WINDOW_MAX, window * 4)


def cmd_tail(args) -> None:
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(json.dumps({"error": "index not found — сначала session_reindex"}, ensure_ascii=False))
        return
    conn = _connect_readonly(db_path)
    agent, src, role = _locate_session(conn, args.session_id, args.agent)
    conn.close()
    if not src:
        print(json.dumps(
            {"error": f"session {args.session_id} не найден в индексе — сделай session_reindex"},
            ensure_ascii=False))
        return

    if agent == "omp":
        msgs = [{"role": "user", "ts": r["_ts"], "text": _clean_text(r["_prompt"] or "")}
                for r in _omp_read(src) if r["session_id"] == args.session_id]
    elif agent in ("kimi-code", "kimi-cli"):
        try:
            big = src.stat().st_size > TAIL_BIG_FILE
        except OSError:
            big = False
        if big:
            # большие wire.jsonl (kimi-cli до 282 МБ): читаем только окно
            # с конца; семантика вывода та же, что у полного разбора
            msgs = _tail_msgs_from_wire(agent, src, max(1, args.n))
        else:
            msgs, _ = _parse_source(agent, src, {"session_id": args.session_id})
    else:
        msgs, _ = _parse_source(agent, src, {"session_id": args.session_id})
    msgs = [m for m in msgs if m["role"] in ("user", "assistant") and m["text"]]
    tail = msgs[-max(1, args.n):]
    print(json.dumps({
        "session_id": args.session_id,
        "agent": agent,
        "src_path": str(src),
        "is_subagent": role == "subagent",
        "messages": [
            {"role": m["role"], "ts": _epoch_to_iso(m["ts"]),
             "text": m["text"][:800]}
            for m in tail
        ],
    }, ensure_ascii=False, indent=2))


def cmd_reindex(args) -> dict:
    """Полная пересборка: сборка во временный файл + атомарная замена.

    Старый индекс не удаляется до полного успеха пересборки — упавший
    embedding-вызов не оставляет поиск без работающего индекса. Сериализована
    маркером .rebuild.lock (вместе с index).

    Бюджет max_new_chunks — защита от MCP-таймаута, но частичная временная
    БД НЕ заменяет основной индекс (фаза 7 аудита р3: прежде os.replace
    ставил неполный индекс молча). Итоговый JSON печатается ПОСЛЕ замены и
    освобождения маркера: run_cli убивает процесс сразу за чтением JSON."""
    db_path = Path(args.db).resolve()
    lock = _acquire_rebuild_lock(db_path)
    if lock is None:
        return
    try:
        tmp_db = Path(str(db_path) + ".tmp")
        _drop_db_files(tmp_db)
        tmp_args = argparse.Namespace(db=str(tmp_db), max_new_chunks=args.max_new_chunks)
        try:
            payload = cmd_index(tmp_args, _lock_held=True, _print_result=False)
        except BaseException:
            _drop_db_files(tmp_db)
            print(f"reindex failed — старый индекс не тронут: {db_path}",
                  file=sys.stderr)
            raise
        if payload.get("remaining_chunks"):
            # неполная временная БД: замена уничтожила бы индекс — откат
            _drop_db_files(tmp_db)
            payload.update({
                "ok": False,
                "error": f"бюджет {args.max_new_chunks} чанков исчерпан, "
                         f"осталось {payload['remaining_chunks']} — подними "
                         f"max_chunks или запускай CLI напрямую",
                "reindexed": False,
            })
        else:
            for suffix in ("-wal", "-shm"):  # хвосты WAL закрытой временной БД
                p = Path(str(tmp_db) + suffix)
                if p.exists():
                    p.unlink()
            _drop_db_files(db_path)
            os.replace(tmp_db, db_path)
            payload["reindexed"] = True
            print(f"reindexed {db_path}", file=sys.stderr)
    finally:
        _release_rebuild_lock(lock)
    print(json.dumps(payload, ensure_ascii=False))
    return payload


# --- main --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(prog="session-search",
                                description="Hybrid поиск по сессиям AI-агентов")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="Инкрементальная индексация (с бюджетом)")
    pi.add_argument("--max-new-chunks", type=int, default=DEFAULT_BUDGET)
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="Поиск по индексу")
    ps.add_argument("query")
    ps.add_argument("--agent", default=None)
    ps.add_argument("--workdir", default=None, help="фильтр по work_dir (подстрока)")
    ps.add_argument("--mode", choices=_MODES, default="hybrid")
    ps.add_argument("--rerank", action="store_true",
                    help="применить BGE-reranker к топу hybrid (на коротких диалоговых чанках нестабилен, по умолчанию выкл)")
    ps.add_argument("--top-k", type=int, default=10)
    ps.add_argument("--since", default=None, help="ISO-дата (YYYY-MM-DD)")
    ps.set_defaults(func=cmd_search)

    pst = sub.add_parser("status", help="Статус индекса (без эмбеддингов)")
    pst.set_defaults(func=cmd_status)

    pr = sub.add_parser("reindex", help="Полная пересборка с нуля (tmp + атомарная замена)")
    pr.add_argument("--max-new-chunks", type=int, default=DEFAULT_BUDGET)
    pr.set_defaults(func=cmd_reindex)

    pl = sub.add_parser("list", help="Список сессий в индексе")
    pl.add_argument("--agent", default=None)
    pl.add_argument("--empty", action="store_true",
                    help="вместо сессий — источники без единого чанка "
                         "(тихие потери парсинга)")
    pl.set_defaults(func=cmd_list)

    pt = sub.add_parser("tail", help="Хвост сессии из источника (всегда свежий)")
    pt.add_argument("session_id")
    pt.add_argument("--agent", default=None)
    pt.add_argument("-n", type=int, default=20)
    pt.set_defaults(func=cmd_tail)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
