#!/usr/bin/env python3
"""project-search — семантический поиск по рабочему дереву проекта.

Гибридный (BM25 + vector) поиск по корпусу проекта:
  - kind=skill: проектные скилы `.agents/skills/**` (md + references);
  - kind=docs:  остальные markdown (AGENTS.md, README, documentation/, ...);
  - kind=code:  комментарии исходников (код целиком не индексируем —
                точный поиск по коду делает grep агента).

SQLite FTS5 + sqlite-vec + BGE-M3; примитивы эмбеддинга/стемминга/RRF/
реранка — общие с skills-search/session-search (search_core.py). Чанкер
markdown — chunker_md.py. Индекс живёт внутри дерева:
`<root>/.agents/.index/index.db` (gitignored), переживает переезды дерева.

Никаких демонов: индекс обновляется только явной командой («агент решает»).

Команды:
    project-search index     [--root PATH] [--max-new-chunks N] [--json]
    project-search search    QUERY [--top-k N] [--mode M] [--kind K]
                                   [--no-rerank] [--abstain|--no-abstain]
    project-search status    [--root PATH]
    project-search reindex   [--root PATH] [--max-new-chunks N]
    project-search sync-copy [--root PATH]   # обновить копию в дереве из канона

root резолвится: флаг --root → env PROJECT_ROOT → вверх от cwd до каталога
с AGENTS.md/.agents. Индекс — <root>/.agents/.index/index.db (--db перекрывает).

Конкурентность: search/status — read-only подключение; index/reindex —
сериализованы маркером .rebuild.lock рядом с БД; reindex собирает индекс во
временный файл и атомарно заменяет (падение embedding-модели не теряет
старый индекс).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import httpx
from snowballstemmer import stemmer as _snowball_stemmer

# При запуске из self-contained копии в дереве проекта модули лежат рядом;
# в базе знаний (tools/) — тоже рядом. Путь не поднимаем.
from search_core import (
    EMBEDDING_DIM,
    RERANKER_TOP_N,
    RRF_K,
    absent_by_cosine,
    embed_texts,
    fts_column_query,
    init_vec_conn,
    low_confidence,
    rerank_from_db as _rerank,
    rrf_fuse as _rrf_fuse,
    serialize_f32,
    stem_query,
    stem_text,
)

from chunker_md import CHUNK_SIZE, chunk_markdown, parse_frontmatter  # noqa: F401

# На Windows при перенаправлении stdout в pipe кодировка может быть cp1251 —
# фиксируем UTF-8, чтобы JSON с кириллицей не ломался на стороне MCP-обёртки.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Конфигурация -------------------------------------------------------------

CHUNK_MAX_CHARS = CHUNK_SIZE          # 1500
DEFAULT_BUDGET = 300                  # новых чанков за вызов из MCP
MAX_EMBED_CHARS = 10000              # потолок длины текста на embed
KINDS = ("skill", "docs", "code")
VALID_KINDS = KINDS + ("all",)

# Каталоги, исключаемые из обхода всегда (в дополнение к <root>/.ignore)
SKIP_DIR_NAMES = {
    ".git", ".venv", "__pycache__", "node_modules", ".pytest_cache",
    "out", "bin", "build", "build_logs", ".search", ".graph", ".index",
}
SKIP_FILES = {".env", "package-lock.json", "yarn.lock"}

# Реестр «расширение → язык комментариев» (kind=code). CMakeLists.txt —
# по имени файла, для остальных .txt язык не определён (не индексируем).
LANG_BY_EXT = {
    ".c": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp",
    ".h": "cpp", ".hpp": "cpp",
    ".py": "py",
    ".sql": "sql",
    ".yml": "yml", ".yaml": "yml", ".ini": "hash", ".cfg": "hash",
    ".cmake": "cmake",
    ".ps1": "ps1", ".sh": "hash",
}

# Блокировка перестроения (index/reindex): маркер .rebuild.lock рядом с БД.
REBUILD_LOCK_NAME = ".rebuild.lock"
LOCK_STALE_SEC = 1800

# Мёрж двух источников скилов (global + project) — лёгкий приоритет
# проектных: они живые каноны. Калибруется на первом прогоне.
PROJECT_BOOST = 1.08


# --- Разрешение корня проекта -------------------------------------------------

def _is_project_root(path: Path) -> bool:
    return (path / "AGENTS.md").exists() or (path / ".agents").is_dir()


def discover_root(start: Path | str = None) -> Path | None:
    """Вверх от start (дефолт — cwd) до каталога с AGENTS.md/.agents."""
    cur = Path(start).resolve() if start else Path.cwd()
    if cur.is_file():
        cur = cur.parent
    for p in (cur, *cur.parents):
        if _is_project_root(p):
            return p
    return None


def resolve_root(args_root: str | None) -> Path:
    """--root → env PROJECT_ROOT → вверх от cwd. Нет корня — ошибка."""
    if args_root:
        root = Path(args_root).resolve()
        if not _is_project_root(root):
            print(f"Error: {root} не похож на корень проекта "
                  "(нет AGENTS.md/.agents)", file=sys.stderr)
            sys.exit(1)
        return root
    env = os.environ.get("PROJECT_ROOT")
    if env:
        root = Path(env).resolve()
        if _is_project_root(root):
            return root
    root = discover_root()
    if root is not None:
        return root
    print("Error: корень проекта не найден (--root, PROJECT_ROOT или "
          "запуск из дерева с AGENTS.md/.agents)", file=sys.stderr)
    sys.exit(1)


def db_path_for(root: Path, args_db: str | None = None) -> Path:
    if args_db:
        return Path(args_db).resolve()
    return root / ".agents" / ".index" / "index.db"


# --- Правила .ignore ----------------------------------------------------------

def parse_ignore(root: Path) -> tuple[list[str], list[str]]:
    """<root>/.ignore → (dir_prefixes, file_patterns).

    Строки с ведущими `**/` и хвостом `/**` сворачиваются в префикс каталога
    (например `**/out/**` → `out`: prune всей ветки при обходе). Остальные —
    fnmatch-паттерны для файлов (по rel-пути). Комментарии (#) и пустые —
    пропуск.
    """
    ign = root / ".ignore"
    dirs: list[str] = []
    files: list[str] = []
    if not ign.exists():
        return dirs, files
    try:
        src = ign.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return dirs, files
    for raw in src:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace("\\", "/").lstrip("./")
        while line.startswith("**/"):
            line = line[3:]
        if line.endswith("/**"):
            line = line[:-3].rstrip("/")
            if line:
                dirs.append(line)
        elif line.endswith("/"):
            dirs.append(line.rstrip("/"))
        else:
            files.append(line)
    return dirs, files


def _dir_excluded(rel: str, prefixes: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    for p in prefixes:
        if rel == p or rel.startswith(p + "/"):
            return True
    return False


def _file_excluded(rel: str, patterns: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    base = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(base, p)
               for p in patterns)


# --- Сканер корпуса -----------------------------------------------------------

def _kind_for(rel: str, ext: str, filename: str) -> str | None:
    """kind файла по rel-пути/расширению; None — не индексируем."""
    if ext == ".md":
        parts = rel.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] == ".agents" and parts[1] == "skills":
            return "skill"
        return "docs"
    if filename == "CMakeLists.txt":
        return "code"
    if fnmatch.fnmatch(filename, "CMakeLists.txt"):
        return "code"
    if ext in LANG_BY_EXT:
        return "code"
    return None


def _project_skill_owners(root: Path) -> dict[str, str]:
    """rel-префикс каталога скила → каноническое имя скила (frontmatter)."""
    owners: dict[str, str] = {}
    skills_dir = root / ".agents" / "skills"
    if not skills_dir.is_dir():
        return owners
    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir():
            continue
        sk = d / "SKILL.md"
        name = d.name
        if sk.exists():
            try:
                fm, _ = parse_frontmatter(sk.read_text(encoding="utf-8"))
                name = fm.get("name", d.name)
            except (OSError, UnicodeDecodeError):
                pass
        rel = str(d.relative_to(root)).replace("\\", "/")
        owners[rel] = str(name)
    return owners


def _project_skill_meta(root: Path) -> dict[str, dict]:
    """rel-каталог скила -> {name, scope, category, status, updated}."""
    meta: dict[str, dict] = {}
    skills_dir = root / ".agents" / "skills"
    if not skills_dir.is_dir():
        return meta
    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir():
            continue
        sk = d / "SKILL.md"
        fm: dict = {}
        if sk.exists():
            try:
                fm, _ = parse_frontmatter(sk.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                fm = {}
        rel = str(d.relative_to(root)).replace("\\", "/")
        meta[rel] = {
            "name": str(fm.get("name", d.name)),
            "scope": fm.get("scope"),
            "category": fm.get("category"),
            "status": fm.get("status", "active"),
            "updated": fm.get("updated"),
        }
    return meta


def _project_skill_pred(meta: dict[str, dict], scope, category, skill,
                        include_archived: bool = False):
    """Предикат file_path для скил-фильтров проектного индекса.

    Возвращает (sql, params) с GLOB-принадлежностью каталогам скилов;
    None — активные фильтры дали пустое множество (несовместимы).
    Аналогично tools/skills-search.py::filter_skill_dirs + _glob_predicate.
    """
    active = scope is not None or category is not None or skill is not None
    if not active:
        # без активных скил-фильтров предикат не ограничивает корпус
        # (иначе поиск по проекту молча сужался бы до .agents/skills/*)
        return None
    dirs: list[str] = []
    for d, m in meta.items():
        if m["status"] == "archived" and not include_archived:
            continue
        if skill is not None and skill not in (d, m["name"]):
            continue
        if scope is not None and m["scope"] != scope:
            continue
        if category is not None and m["category"] != category:
            continue
        dirs.append(d)
    if active and not dirs:
        return None
    if not dirs:
        return None
    dirs = sorted(dirs)
    sql = "(" + " OR ".join(f"file_path GLOB ?" for _ in dirs) + ")"
    return sql, [f"{d}/*" for d in dirs]


def _and_preds(*parts) -> tuple[str, list] | None:
    parts = [p for p in parts if p is not None and p[0] != "1=1"]
    if not parts:
        return ("1=1", [])
    return (" AND ".join(p[0] for p in parts),
            [param for p in parts for param in p[1]])


def _owner_for_skill_file(rel: str, owners: dict[str, str]) -> str:
    rel = rel.replace("\\", "/")
    for prefix, name in owners.items():
        if rel == prefix + "/SKILL.md" or rel.startswith(prefix + "/"):
            return name
    return "(root)"


def collect_files(root: Path) -> list[tuple[Path, str, str]]:
    """Обход корпуса: [(full_path, rel_path, kind)], .ignore + SKIP учтены."""
    dir_prefixes, file_patterns = parse_ignore(root)
    out: list[tuple[Path, str, str]] = []
    for dirpath, dirs, files in os.walk(root):
        rel_dir = str(Path(dirpath).relative_to(root)).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""
        kept = []
        for d in dirs:
            drel = f"{rel_dir}/{d}" if rel_dir else d
            if d in SKIP_DIR_NAMES or _dir_excluded(drel, dir_prefixes):
                continue
            if (Path(dirpath) / d).is_symlink():
                continue
            kept.append(d)
        dirs[:] = sorted(kept)
        for f in sorted(files):
            if f in SKIP_FILES:
                continue
            full = Path(dirpath) / f
            if full.is_symlink():
                continue
            rel = str(full.relative_to(root)).replace("\\", "/")
            if _file_excluded(rel, file_patterns):
                continue
            kind = _kind_for(rel, full.suffix.lower(), f)
            if kind is not None:
                out.append((full, rel, kind))
    return out


# --- Экстракция комментариев --------------------------------------------------

def _find_next_code_line(lines: list[str], after: int,
                         skip_prefixes: tuple = ("#",)) -> str:
    """Первая значимая непустая строка после `after` (сигнатура сущности).

    Строки-комментарии других языков по префиксам skip_prefixes пропускаются
    (`#` для py/yml/hash, `//` для cpp, `--` для sql), чтобы сигнатурой не
    стал соседний комментарий.
    """
    for j in range(after, len(lines)):
        s = lines[j].strip()
        if not s:
            continue
        if any(s.startswith(p) for p in skip_prefixes):
            continue
        return s
    return ""


def _strip_trailing_comment(sig: str, lang: str) -> str:
    """Убрать хвостовой комментарий из строки-сигнатуры."""
    if not sig:
        return ""
    if lang in ("cpp", "sql"):
        sig = re.sub(r"/\*.*?\*/", "", sig)
    if lang == "cpp" and "//" in sig:
        sig = sig.split("//", 1)[0].rstrip()
    elif lang == "sql" and "--" in sig:
        sig = sig.split("--", 1)[0].rstrip()
    elif lang == "yml":
        pos = _find_hash_comment(sig)
        if pos != -1:
            sig = sig[:pos].rstrip()
    return sig


class _CScanner:
    """Сканер комментариев C-семейства: `//` и `/* */` с учётом строк/char.

    Препроцессорные директивы (#if 0 и пр.) — код: в блоки не попадают;
    текстовые комментарии внутри них извлекаются как и везде.
    """

    def __init__(self, lines: list[str]):
        self.lines = lines
        self.blocks: list[dict] = []

    def run(self) -> list[dict]:
        in_block = False
        block_lines: list[str] = []
        block_start = 0
        block_prefix = ""
        for idx, line in enumerate(self.lines):
            i = 0
            n = len(line)
            if in_block:
                end = line.find("*/")
                if end == -1:
                    block_lines.append(line)
                    continue
                block_lines.append(line[:end])
                self._append(block_start, block_lines, block_prefix, idx,
                             _strip_trailing_comment(
                                 _find_next_code_line(self.lines, idx + 1, ("//",)),
                                 "cpp"))
                in_block = False
                i = end + 2
            string = ""
            while i < n:
                ch = line[i]
                if string:
                    if ch == "\\":
                        i += 2
                        continue
                    if ch == string:
                        string = ""
                    i += 1
                    continue
                nxt = line[i:i + 2]
                if nxt == "//":
                    # inline-комментарий: префикс строки — сигнатура
                    prefix = line[:i].strip()
                    if prefix.endswith("*/"):  # хвост закрытого блока выше
                        prefix = ""
                    self._append(idx + 1, [line[i + 2:]], prefix, idx + 1,
                                 _strip_trailing_comment(
                                     _find_next_code_line(self.lines, idx + 1, ("//",)),
                                     "cpp"))
                    break
                if nxt == "/*":
                    in_block = True
                    block_start = idx + 1
                    block_prefix = line[:i].strip()
                    block_lines = [line[i + 2:] if "*/" not in line[i + 2:]
                                   else line[i + 2:line.find("*/", i + 2)]]
                    if "*/" in line[i + 2:]:
                        self._append(block_start, block_lines, block_prefix,
                                     idx + 1,
                                     _strip_trailing_comment(
                                         _find_next_code_line(self.lines, idx + 1, ("//",)),
                                         "cpp"))
                        in_block = False
                    break
                if ch == '"' or ch == "'":
                    string = ch
                    i += 1
                    continue
                i += 1
        if in_block:  # незакрытый блок — до конца файла
            self._append(block_start, block_lines, block_prefix,
                         len(self.lines), "")
        return self.blocks

    def _append(self, start: int, lines: list[str], prefix: str, end: int,
                sig: str) -> None:
        text = "\n".join(lines).strip()
        if not text:
            return
        self.blocks.append({
            "start": start, "end": end, "text": text,
            "sig": prefix or sig,
        })


def _py_scanner(lines: list[str]) -> list[dict]:
    """Python: `#` и модульные docstrings `'''`/`\"\"\"` (строки не считаем)."""
    blocks: list[dict] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith(("#", "'''", '"""')):
            if stripped.startswith("#"):
                text = stripped[1:].strip()
                if text:
                    blocks.append({"start": i + 1, "end": i + 1, "text": text,
                                   "sig": _find_next_code_line(lines, i + 1, ("#",))})
                i += 1
                continue
            marker = stripped[:3]
            closed = stripped.endswith(marker) and len(stripped) > 3
            buf = [stripped[3:-3] if closed else stripped[3:]]
            j = i + 1
            while j < n and not closed:
                ln = lines[j]
                if ln.rstrip().endswith(marker):
                    closed = True
                    buf.append(ln.rstrip()[:-3])
                else:
                    buf.append(ln)
                j += 1
            text = "\n".join(buf).strip()
            if text:
                blocks.append({"start": i + 1, "end": j, "text": text,
                               "sig": _find_next_code_line(lines, j, ("#",))})
            i = j
            continue
        # строки с обычными кавычками: просто пропускаем (шум-фильтр `#` внутри
        # строк не отсекается — осознанное ограничение не-AST-сканера)
        if "#" in lines[i]:
            pos = lines[i].find("#")
            text = lines[i][pos + 1:].strip()
            if text:
                blocks.append({"start": i + 1, "end": i + 1, "text": text,
                               "sig": (lines[i][:pos].strip() or
                                       _find_next_code_line(lines, i + 1, ("#",)))})
        i += 1
    return blocks


def _sql_scanner(lines: list[str]) -> list[dict]:
    """SQL: `--` и `/* */`; строки '...' не считаем комментариями."""
    blocks: list[dict] = []
    in_block = False
    buf: list[str] = []
    bstart = 0
    for idx, line in enumerate(lines):
        i = 0
        n = len(line)
        if in_block:
            end = line.find("*/")
            if end == -1:
                buf.append(line)
                continue
            buf.append(line[:end])
            text = "\n".join(buf).strip()
            if text:
                blocks.append({"start": bstart, "end": idx + 1, "text": text,
                               "sig": _strip_trailing_comment(
                                   _find_next_code_line(lines, idx + 1, ("--",)),
                                   "sql")})
            buf = []
            in_block = False
            i = end + 2
        string = False
        while i < n:
            ch = line[i]
            if string:
                if ch == "'":
                    string = False
                i += 1
                continue
            nxt = line[i:i + 2]
            if nxt == "--":
                text = line[i + 2:].strip()
                if text:
                    blocks.append({"start": idx + 1, "end": idx + 1,
                                   "text": text,
                                   "sig": (line[:i].strip() or _strip_trailing_comment(
                                           _find_next_code_line(lines, idx + 1, ("--",)),
                                           "sql"))})
                break
            if nxt == "/*":
                in_block = True
                bstart = idx + 1
                tail = line[i + 2:]
                if "*/" in tail:
                    text = tail[:tail.find("*/")].strip()
                    if text:
                        blocks.append({"start": idx + 1, "end": idx + 1,
                                       "text": text,
                                       "sig": (line[:i].strip() or _strip_trailing_comment(
                                               _find_next_code_line(lines, idx + 1, ("--",)),
                                               "sql"))})
                    in_block = False
                else:
                    buf = [tail]
                break
            if ch == "'":
                string = True
            i += 1
    if in_block:
        text = "\n".join(buf).strip()
        if text:
            blocks.append({"start": bstart, "end": len(lines), "text": text,
                           "sig": ""})
    return blocks


def _hash_scanner(lines: list[str]) -> list[dict]:
    """Языки с `#` (sh/ini): комментарий от # до конца строки."""
    blocks: list[dict] = []
    for idx, line in enumerate(lines):
        if "#" not in line:
            continue
        pos = line.find("#")
        text = line[pos + 1:].strip()
        if not text:
            continue
        blocks.append({"start": idx + 1, "end": idx + 1, "text": text,
                       "sig": (line[:pos].strip() or
                               _find_next_code_line(lines, idx + 1, ("#",)))})
    return blocks


def _find_hash_comment(line: str) -> int:
    """Индекс `#`, открывающего комментарий: пробел перед ним вне кавычек."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] == " "):
            return i
    return -1


def _yml_scanner(lines: list[str]) -> list[dict]:
    """YAML: `#` в начале строки или после пробела (спецификация YAML),
    вне одиночных/двойных кавычек."""
    blocks: list[dict] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            pos = _find_hash_comment(line)
            if pos == -1:
                continue
            text = line[pos + 1:].strip()
            prefix = line[:pos].strip()
        else:
            text = stripped[1:].strip()
            prefix = ""
        if not text:
            continue
        blocks.append({"start": idx + 1, "end": idx + 1, "text": text,
                       "sig": _strip_trailing_comment(
                           prefix or _find_next_code_line(lines, idx + 1, ("#",)),
                           "yml")})
    return blocks


def _ps1_scanner(lines: list[str]) -> list[dict]:
    """PowerShell: `#` и блоки `<# #>`."""
    text_all = "\n".join(lines)
    blocks: list[dict] = []

    def _sig(pos: int) -> str:
        for j in range(pos, len(lines)):
            s = lines[j].strip()
            if s and not s.startswith("#"):
                return s
        return ""

    for m in re.finditer(r"<#(.*?)#>", text_all, re.S):
        start = text_all.count("\n", 0, m.start()) + 1
        t = m.group(1).strip()
        if t:
            blocks.append({"start": start, "end": start + t.count("\n"),
                           "text": t, "sig": _sig(start)})
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("<#"):
            continue
        pos = line.find("#")
        if pos == -1:
            continue
        text = line[pos + 1:].strip()
        if not text:
            continue
        blocks.append({"start": idx + 1, "end": idx + 1, "text": text,
                       "sig": (line[:pos].strip() or
                               _find_next_code_line(lines, idx + 1, ("#",)))})
    return blocks


def extract_comments(lang: str, text: str) -> list[dict]:
    """Список коммент-блоков: {start, end, text, sig}."""
    lines = text.split("\n")
    if lang == "cpp":
        return _CScanner(lines).run()
    if lang == "py":
        return _py_scanner(lines)
    if lang == "sql":
        return _sql_scanner(lines)
    if lang == "yml":
        return _yml_scanner(lines)
    if lang == "ps1":
        return _ps1_scanner(lines)
    return _hash_scanner(lines)  # hash, cmake (cmake: `#` вне кавычек — приближение)


def _blocks_to_chunks(rel: str, blocks: list[dict]) -> list[dict]:
    """Коммент-блоки → чанки ~CHUNK_MAX_CHARS; блок не разрезается,
    накопление с переносом последнего блока (перекрытие)."""
    chunks: list[dict] = []
    current: list[dict] = []
    size = 0
    for b in blocks:
        blen = len(b["text"])
        if size + blen + 2 > CHUNK_MAX_CHARS and current:
            chunks.append(_join_block(current, rel))
            carry = current[-1:] if len(current[-1]["text"]) <= CHUNK_MAX_CHARS else []
            current = carry
            size = sum(len(x["text"]) + 2 for x in current)
        current.append(b)
        size += blen + 2
    if current:
        chunks.append(_join_block(current, rel))
    return chunks


def _join_block(blocks: list[dict], rel: str) -> dict:
    text = "\n\n".join(b["text"] for b in blocks)
    start = blocks[0]["start"]
    sig = next((b["sig"] for b in blocks if b["sig"]), "")
    prefix = f"[{rel}:#{start}]"
    if sig:
        prefix += f" {sig}"
    return {
        "text": text,
        "file": rel,
        "line": start,
        "heading": "",
        "text_for_embedding": f"{prefix}\n\n{text}",
        "_sig": sig,
    }


def _chunks_for_file(full_path: Path, rel: str, kind: str,
                     skill_owner: str) -> list[dict]:
    """Чанки одного файла корпуса (md — чанкером, code — коммент-блоками)."""
    try:
        content = full_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if kind in ("skill", "docs"):
        owner = skill_owner if kind == "skill" else rel
        return chunk_markdown(content, rel, owner=owner)
    blocks = extract_comments(LANG_BY_EXT.get(full_path.suffix.lower(), "hash"),
                              content)
    if full_path.name == "CMakeLists.txt":
        blocks = extract_comments("hash", content)
    return _blocks_to_chunks(rel, blocks)


# --- БД ----------------------------------------------------------------------

def init_db(db_path: Path, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        return _connect_readonly(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    init_vec_conn(conn, strict=True)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            indexed_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            kind TEXT NOT NULL,
            skill TEXT NOT NULL DEFAULT '',
            heading TEXT NOT NULL DEFAULT '',
            line INTEGER,
            part INTEGER NOT NULL DEFAULT 0,
            text TEXT NOT NULL,
            stemmed_text TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            stemmed_text, skill, heading,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, stemmed_text, skill, heading)
            VALUES (new.id, new.stemmed_text, new.skill, new.heading);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, stemmed_text, skill, heading)
            VALUES ('delete', old.id, old.stemmed_text, old.skill, old.heading);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, stemmed_text, skill, heading)
            VALUES ('delete', old.id, old.stemmed_text, old.skill, old.heading);
            INSERT INTO chunks_fts(rowid, stemmed_text, skill, heading)
            VALUES (new.id, new.stemmed_text, new.skill, new.heading);
        END;
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    return conn


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA busy_timeout=5000")
        init_vec_conn(conn)
        conn.execute("SELECT COUNT(*) FROM sqlite_master")
        return conn
    except sqlite3.Error:
        if conn is not None:
            conn.close()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA query_only=ON")
    init_vec_conn(conn)
    return conn


def _meta_get(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )


# --- Блокировка перестроения --------------------------------------------------

def _pid_alive(pid) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError, OverflowError):
        return False
    return True


def _acquire_rebuild_lock(db_path: Path) -> Path | None:
    lock = db_path.parent / REBUILD_LOCK_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)  # первый запуск в свежем дереве
    if lock.exists():
        try:
            pid_s, _ = lock.read_text(encoding="utf-8").strip().split("|", 1)
            pid = int(pid_s)
        except (OSError, ValueError):
            pid = None
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            age = 0.0
        if _pid_alive(pid) or age < LOCK_STALE_SEC:
            print(json.dumps({
                "ok": False,
                "locked": True,
                "error": f"перестроение уже идёт (pid {pid}, "
                         f"возраст {age/60:.1f} мин) — подожди и повтори",
            }, ensure_ascii=False))
            return None
        try:
            lock.unlink()
        except OSError:
            return None
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except (FileExistsError, OSError):
        print(json.dumps({"ok": False, "locked": True,
                          "error": "перестроение уже идёт (параллельный процесс)"},
                         ensure_ascii=False))
        return None
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()}|{time.time()}")
    return lock


def _release_rebuild_lock(lock: Path) -> None:
    try:
        lock.unlink(missing_ok=True)
    except OSError:
        pass


# --- Индексация ---------------------------------------------------------------

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _wipe_file(conn: sqlite3.Connection, rel: str) -> None:
    conn.execute("DELETE FROM chunks_vec WHERE id IN "
                 "(SELECT id FROM chunks WHERE file_path = ?)", (rel,))
    conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))
    conn.execute("DELETE FROM files WHERE path = ?", (rel,))


def cmd_index(args) -> None:
    root = resolve_root(getattr(args, "root", None))
    db_path = db_path_for(root, getattr(args, "db", None))
    budget = max(1, int(getattr(args, "max_new_chunks", DEFAULT_BUDGET)))
    as_json = getattr(args, "json", False)

    lock = _acquire_rebuild_lock(db_path)
    if lock is None:
        sys.exit(1)
    try:
        conn = init_db(db_path)
        _meta_set(conn, "root", str(root))
        owners = _project_skill_owners(root)
        files = collect_files(root)
        rels = {rel for _, rel, _ in files}
        stored = {r[0]: (r[1], r[2]) for r in
                  conn.execute("SELECT path, kind, content_hash FROM files")}
        for rel in list(stored):
            if rel not in rels:
                _wipe_file(conn, rel)
        conn.commit()

        stale = []
        for full, rel, kind in files:
            try:
                content = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            h = _content_hash(content)
            if stored.get(rel) != (kind, h):
                stale.append((full, rel, kind, h))
        stale.sort(key=lambda t: t[0].stat().st_mtime, reverse=True)
        for full, rel, kind, h in stale:
            print(f"changed: {rel}", file=sys.stderr)

        indexed_chunks = 0
        remaining_chunks = 0
        indexed_files = 0
        error = None

        # Фаза A: распарсить и нарезать всё, что помещается в бюджет.
        # Эмбеддинги — ОДНИМ батчем на весь заход (единый вызов embed_texts):
        # прежний вызов на файл давал HTTP-запрос на каждый 1-чанковый файл
        # (тысячи запросов — минуты вместо секунд на маленьких комментариях).
        work: list[tuple] = []
        for full, rel, kind, h in stale:
            owner = (_owner_for_skill_file(rel, owners)
                     if kind == "skill" else "")
            chunks = _chunks_for_file(full, rel, kind, owner)
            if indexed_chunks + len(chunks) > budget:
                remaining_chunks += len(chunks)
                continue
            if not chunks:
                continue
            work.append((full, rel, kind, h, chunks))
            indexed_chunks += len(chunks)

        # Фаза B: эмбеддинги одним запросом. Гигантские неразрезанные
        # чанки (один md-блок > CHUNK_SIZE оставляется целиком) режутся для
        # ЭМБЕДДИНГА до MAX_EMBED_CHARS — иначе BGE-M3 (лимит 8192 токенов)
        # отвечает 400 на весь батч. В БД чанк хранится целиком: BM25,
        # rerank и сниппеты работают по полному тексту.
        all_texts = [t if len(t) <= MAX_EMBED_CHARS else t[:MAX_EMBED_CHARS]
                     for _, _, _, _, chunks in work for c in chunks
                     for t in (c["text_for_embedding"],)]
        try:
            embeddings = embed_texts(all_texts) if all_texts else []
        except (httpx.HTTPError, OSError) as e:
            error = (f"embedding недоступен: {e}. Повтори позже "
                     "(keyword-поиск работает и без модели).")
            print(error, file=sys.stderr)
            embeddings = []
        if embeddings and len(embeddings) != len(all_texts):
            error = (f"embedding count mismatch ({len(embeddings)} vs "
                     f"{len(all_texts)}) — индексация захода отменена")
            print(f"failed: {error}", file=sys.stderr)
            embeddings = []

        # Фаза C: атомарная запись по файлам
        pos = 0
        for full, rel, kind, h, chunks in work:
            if embeddings:
                emb = embeddings[pos:pos + len(chunks)]
                pos += len(chunks)
            else:
                emb = []
            with conn:
                if emb:
                    _wipe_file(conn, rel)
                    conn.execute(
                        "INSERT INTO files(path, kind, content_hash, indexed_at) "
                        "VALUES(?,?,?,?)", (rel, kind, h, time.time()))
                    for c, vec in zip(chunks, emb):
                        stem_for_fts = _project_stem_text(c["text"])
                        if kind == "code" and c.get("_sig"):
                            stem_for_fts = stem_for_fts + "\n" + _project_stem_text(c["_sig"])
                        cur = conn.execute(
                            "INSERT INTO chunks(file_path, kind, skill, heading, line, "
                            "part, text, stemmed_text, content_hash) "
                            "VALUES(?,?,?,?,?,?,?,?,?)",
                            (c["file"], kind, c.get("skill", ""), c.get("heading", ""),
                             c.get("line"), c.get("part", 0), c["text"],
                             stem_for_fts, h))
                        conn.execute(
                            "INSERT INTO chunks_vec(id, embedding) VALUES(?,?)",
                            (cur.lastrowid, serialize_f32(vec)))
                else:
                    if indexed_chunks == 0 and error is not None:
                        # embedding упал для всего захода: файлы остаются
                        # stale и будут переиндексированы следующим вызовом
                        pass
            indexed_files += 1
        if indexed_chunks:
            print(f"indexed {indexed_files} files, {indexed_chunks} chunks",
                  file=sys.stderr)
        if error is not None:
            remaining_chunks = indexed_chunks

        if error is None:
            _meta_set(conn, "last_index_at", time.time())
        conn.commit()
        stats = {
            "ok": error is None,
            "error": error,
            "root": str(root),
            "db": str(db_path),
            "indexed_files": indexed_files,
            "new_chunks": indexed_chunks,
            "remaining_chunks": remaining_chunks,
            "total_files": len(files),
            "chunks_total": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "last_index_at": _meta_get(conn, "last_index_at"),
        }
        conn.close()
    finally:
        _release_rebuild_lock(lock)
    if as_json:
        print(json.dumps(stats, ensure_ascii=False))
    else:
        print(f"Done: {indexed_files} files, {indexed_chunks} chunks "
              f"(remaining: {remaining_chunks}, total: {stats['total_files']}).")


# --- Поиск --------------------------------------------------------------------

def parse_kinds(value: str | None) -> tuple[str, ...]:
    if not value or value == "all":
        return KINDS
    parts = [v.strip() for v in value.split(",") if v.strip()]
    known = tuple(k for k in parts if k in KINDS)
    return known or KINDS


_CYR_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_SB_EN = _snowball_stemmer("english")
_SB_RU = _snowball_stemmer("russian")


def _stem_part(word: str) -> str:
    """Стебль одной части слова (без '_'): Russian — кириллица, English — ASCII."""
    word = word.lower()
    if _CYR_RE.search(word):
        return _SB_RU.stemWord(word)
    if word.isascii():
        return _SB_EN.stemWord(word)
    return word


def _project_stem_text(text: str) -> str:
    """stem_text с разбивкой идентификаторов по '_'.

    Snowball стеммит цельное ASCII-слово с подчёркиванием по-разному в
    зависимости от хвоста: 'direction_restore' -> 'direction_restor', а
    'direction_restore_t' не трогает вообще — термы индекса и запроса
    расходились, keyword-поиск по именам сущностей не находил ничего.
    Части разбиваются, стеммятся по отдельности и склеиваются пробелом
    (токенизатор FTS раскладывает их на отдельные токены).

    Прочие токены — как в search_core.stem_text (пунктуация сохраняется).
    """
    tokens = re.findall(r"\w+|\W+", text, re.UNICODE)
    result = []
    for token in tokens:
        if not token or not token[0].isalnum():
            result.append(token)
            continue
        if "_" in token:
            parts = [_stem_part(p) for p in token.split("_")]
            parts = [p for p in parts if len(p) >= 2]
            result.append(" ".join(parts))
        else:
            result.append(_stem_part(token))
    return "".join(result)


def _project_stem_query(query: str) -> list:
    """Слова запроса для keyword-ветки — в той же норме, что _project_stem_text."""
    stems = []
    for word in re.findall(r"\w{2,}", query, re.UNICODE):
        if "_" in word:
            for part in word.split("_"):
                if len(part) >= 2:
                    stems.append(_stem_part(part))
        elif _CYR_RE.search(word) or word.isascii():
            stems.append(_stem_part(word))
    return stems


def _kind_pred(kinds: tuple[str, ...], alias: str) -> tuple[str, list]:
    if set(kinds) == set(KINDS):
        return ("1=1", [])
    marks = ",".join("?" * len(kinds))
    return (f"{alias}.kind IN ({marks})", list(kinds))


def _search_bm25(conn, query: str, limit: int, pred) -> list[dict]:
    stems = _project_stem_query(query)
    if not stems:
        return []
    fts_query = fts_column_query(stems, "stemmed_text")
    sql = ("SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ?")
    params: list = [fts_query]
    psql, pparams = pred
    if psql != "1=1":
        sql += " AND rowid IN (SELECT id FROM chunks WHERE " + psql + ")"
        params += pparams
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        print(f"Warning: BM25-запрос не выполнен ({e}) — keyword-ветка пуста",
              file=sys.stderr)
        return []
    # bm25() в FTS5: чем меньше значение (отрицательное), тем лучше совпадение.
    # score = -rank — прямой перенос «лучше — выше»; 1/(1+|rank|) инвертировал
    # порядок (ближайший к нулю rank пролезал в топ), ломая keyword-ветку.
    return [{"chunk_id": r, "score": -rank, "source": "bm25"}
            for r, rank in rows]


def _search_vector(conn, query: str, limit: int, pred, knn_k=None) -> list[dict]:
    q = embed_texts([query])[0]
    k = knn_k if knn_k is not None else limit
    psql, pparams = pred
    if psql != "1=1":
        sql = ("SELECT v.id, v.distance FROM chunks_vec v "
               "JOIN chunks c ON c.id = v.id "
               f"WHERE v.embedding MATCH ? AND v.k = ? AND {psql}")
        params = [serialize_f32(q), k] + pparams
    else:
        sql = ("SELECT id, distance FROM chunks_vec "
               "WHERE embedding MATCH ? AND k = ?")
        params = [serialize_f32(q), k]
    rows = conn.execute(sql + " ORDER BY distance", params).fetchall()
    return [{"chunk_id": cid, "score": 1.0 - (d / 2.0), "source": "vector"}
            for cid, d in rows[:limit]]


def search_index(conn, query: str, top_k: int, mode: str, kinds: tuple[str, ...],
                 rerank_enabled: bool = True,
                 extra_preds: tuple | None = None) -> list[dict]:
    """Гибридный поиск по индексу (по следам skills-search).

    extra_preds — (pred_bm25, pred_vec), дополнительные SQL-предикаты на
    file_path (скил-фильтры --scope/--category/--skill), комбинируются AND
    с kind-предикатами.
    """
    base = _kind_pred(kinds, "c")
    base_bm = _kind_pred(kinds, "chunks")
    if extra_preds is not None:
        extra_bm, extra_vec = extra_preds
        pred = _and_preds(base, (extra_vec[0].replace("file_path", "c.file_path"),
                                 extra_vec[1]) if extra_vec else None)
        pred_bm = _and_preds(base_bm, extra_bm)
    else:
        pred, pred_bm = base, base_bm
    results: list[dict] = []
    if mode in ("hybrid", "keyword"):
        results.extend(_search_bm25(conn, query, top_k * 3, pred_bm))
    if mode in ("hybrid", "semantic"):
        try:
            knn_k = None
            if pred[0] != "1=1":
                n_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                # sqlite-vec: k <= 4096. Чем шире выборка ближайших, тем меньше
                # шансов потерять релевантные чанки за kind-фильтром по JOIN.
                knn_k = min(n_total, 4096, max(2000, top_k * 3 * 100))
            results.extend(_search_vector(conn, query, top_k * 3, pred, knn_k))
        except (httpx.HTTPError, OSError, RuntimeError) as e:
            if mode == "semantic":
                raise RuntimeError(
                    f"BGE-M3 недоступна ({e}) — используйте --mode keyword") from e
            print(f"Warning: BGE-M3 недоступна ({e}) — fallback на BM25",
                  file=sys.stderr)
    if mode == "hybrid":
        results = _rrf_fuse(results, k=RRF_K)
        if rerank_enabled:
            results = _rerank(query, results, conn, top_n=max(RERANKER_TOP_N, top_k))
    else:
        seen = {}
        for r in results:
            cid = r["chunk_id"]
            if cid not in seen or r["score"] > seen[cid]["score"]:
                seen[cid] = r
        results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _top1_cosine(conn, query: str) -> float | None:
    """Косинус top-1 сырой векторной ветки (без kind-фильтра) — абстеншен."""
    try:
        vec = _search_vector(conn, query, 1, ("1=1", []))
    except (httpx.HTTPError, OSError, RuntimeError):
        return None
    return vec[0]["score"] if vec else None


def cmd_search(args) -> None:
    db_path = db_path_for(resolve_root(getattr(args, "root", None)),
                          getattr(args, "db", None))
    if not db_path.exists():
        print(json.dumps({"error": "project index not found; сначала "
                                   "`project-search index`", "exists": False},
                         ensure_ascii=False))
        sys.exit(1)
    conn = _connect_readonly(db_path)
    top_k = max(1, int(args.top_k))
    mode = args.mode
    if mode not in ("hybrid", "semantic", "keyword"):
        conn.close()
        raise ValueError(f"unknown search mode: {mode!r}")
    kinds = parse_kinds(getattr(args, "kind", None))
    rerank_enabled = getattr(args, "rerank", True)

    root = resolve_root(getattr(args, "root", None))
    meta = _project_skill_meta(root)
    skill_pred = _project_skill_pred(
        meta, getattr(args, "scope", None), getattr(args, "category", None),
        getattr(args, "skill", None), getattr(args, "include_archived", False))
    if skill_pred is None and (getattr(args, "scope", None) is not None
                               or getattr(args, "category", None) is not None
                               or getattr(args, "skill", None) is not None):
        conn.close()
        print("[]")
        return
    extra = (skill_pred, skill_pred) if skill_pred is not None else None

    try:
        results = search_index(conn, args.query, top_k, mode, kinds, rerank_enabled, extra)
    except RuntimeError as e:
        conn.close()
        print(json.dumps({"error": str(e), "results": []}, ensure_ascii=False))
        sys.exit(1)

    output = []
    ids = [r["chunk_id"] for r in results]
    by_id = {}
    for i in range(0, len(ids), 500):
        ph = ",".join("?" * len(ids[i:i + 500]))
        if not ids[i:i + 500]:
            continue
        for row in conn.execute(
                f"SELECT id, file_path, kind, skill, heading, line, text "
                f"FROM chunks WHERE id IN ({ph})", ids[i:i + 500]):
            by_id[row[0]] = row
    for r in results:
        row = by_id.get(r["chunk_id"])
        if not row:
            continue
        output.append({
            "file": row[1],
            "kind": row[2],
            "skill": row[3] or None,
            "heading": row[4] or None,
            "line": row[5],
            "score": round(r["score"], 4),
            "snippet": row[6][:300].replace("\n", " "),
        })

    abstain = getattr(args, "abstain", True)
    low_conf = False
    if abstain:
        cos_top1 = _top1_cosine(conn, args.query) if mode in ("hybrid", "semantic") else None
        sig = absent_by_cosine(cos_top1)
        low_conf = (sig if sig is not None
                    else low_confidence([o["score"] for o in output]))
        if low_conf:
            print("УВЕРЕННОГО СОВПАДЕНИЯ В ИНДЕКСЕ ПРОЕКТА НЕТ (порог "
                  "абстеншена). Слабые кандидаты ниже — только как ориентир.",
                  file=sys.stderr)
        for o in output:
            o["low_confidence"] = low_conf
    print(json.dumps(output, ensure_ascii=False, indent=2))
    conn.close()


def cmd_status(args) -> None:
    db_path = db_path_for(resolve_root(getattr(args, "root", None)),
                          getattr(args, "db", None))
    if not db_path.exists():
        print(json.dumps({"error": "project index not found", "exists": False},
                         ensure_ascii=False))
        return
    conn = _connect_readonly(db_path)
    n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_vec = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    by_kind = dict(conn.execute(
        "SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall())
    print(json.dumps({
        "exists": True,
        "db_path": str(db_path),
        "root": _meta_get(conn, "root"),
        "files": n_files,
        "chunks": n_chunks,
        "vectors": n_vec,
        "by_kind": by_kind,
        "db_size_mb": round(db_path.stat().st_size / 1024 / 1024, 1),
        "last_index_at": _meta_get(conn, "last_index_at"),
        "locked": (db_path.parent / REBUILD_LOCK_NAME).exists(),
    }, ensure_ascii=False, indent=2))
    conn.close()


def cmd_reindex(args) -> None:
    root = resolve_root(getattr(args, "root", None))
    db_path = db_path_for(root, getattr(args, "db", None))
    # Лок захватывает сам cmd_index (по пути tmp-базы — тот же .rebuild.lock
    # в каталоге индекса); отдельный захват здесь давал взаимный deadlock.
    tmp_db = Path(str(db_path) + ".tmp")
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(tmp_db) + suffix)
        if p.exists():
            p.unlink()
    tmp_args = argparse.Namespace(
        root=str(root), db=str(tmp_db), json=False,
        max_new_chunks=int(getattr(args, "max_new_chunks", 1 << 30)))
    try:
        cmd_index(tmp_args)
    except BaseException:
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(tmp_db) + suffix)
            if p.exists():
                p.unlink()
        print(f"reindex failed — старый индекс не тронут: {db_path}",
              file=sys.stderr)
        raise
    for suffix in ("-wal", "-shm"):
        p = Path(str(tmp_db) + suffix)
        if p.exists():
            p.unlink()
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    os.replace(tmp_db, db_path)
    print(f"Reindexed: {db_path}")


# --- Копия в дерево проекта ---------------------------------------------------

COPY_TARGETS = ("project_search.py", "search_core.py", "chunker_md.py")
REQUIREMENTS_TXT = "sqlite-vec\nhttpx>=0.27\nsnowballstemmer>=2.2\n"


def cmd_sync_copy(args) -> None:
    root = resolve_root(getattr(args, "root", None))
    dest_dir = root / ".agents" / "skills" / "project-search" / "scripts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parent
    report = []
    for name in COPY_TARGETS:
        src = src_dir / name
        dst = dest_dir / name
        src_bytes = src.read_bytes()
        action = "unchanged"
        if dst.exists() and dst.read_bytes() == src_bytes:
            pass
        else:
            dst.write_bytes(src_bytes)
            action = "synced"
        report.append({"file": name, "action": action})
    req = dest_dir.parent / "requirements.txt"
    if not req.exists() or req.read_text(encoding="utf-8") != REQUIREMENTS_TXT:
        req.write_text(REQUIREMENTS_TXT, encoding="utf-8")
        report.append({"file": "requirements.txt", "action": "synced"})
    else:
        report.append({"file": "requirements.txt", "action": "unchanged"})
    print(json.dumps({"dest": str(dest_dir), "files": report},
                     ensure_ascii=False, indent=2))


# --- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="project-search",
        description="Гибридный поиск по дереву проекта (скилы + docs + "
                    "комментарии исходников)")
    parser.add_argument("--root", default=None,
                        help="корень проекта (иначе PROJECT_ROOT / вверх от cwd)")
    parser.add_argument("--db", default=None,
                        help="путь к индексу (по умолчанию <root>/.agents/.index/index.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="инкрементальная доиндексация с бюджетом")
    pi.add_argument("--root", default=None, help=argparse.SUPPRESS)
    pi.add_argument("--db", default=None, help=argparse.SUPPRESS)
    pi.add_argument("--max-new-chunks", type=int, default=DEFAULT_BUDGET)
    pi.add_argument("--json", action="store_true")

    ps = sub.add_parser("search", help="поиск по индексу проекта")
    ps.add_argument("query")
    ps.add_argument("--root", default=None, help=argparse.SUPPRESS)
    ps.add_argument("--db", default=None, help=argparse.SUPPRESS)
    ps.add_argument("--top-k", type=int, default=10)
    ps.add_argument("--mode", choices=["hybrid", "semantic", "keyword"],
                    default="hybrid")
    ps.add_argument("--kind", default=None,
                    help="skill|docs|code|all (или списком через запятую); "
                         "по умолчанию — все виды")
    ps.add_argument("--no-rerank", dest="rerank", action="store_false", default=True)
    ps.add_argument("--abstain", dest="abstain", action="store_true", default=True)
    ps.add_argument("--no-abstain", dest="abstain", action="store_false")
    ps.add_argument("--scope", default=None,
                    help="фильтр по scope проектных скилов (personal|work|shared)")
    ps.add_argument("--category", default=None,
                    help="фильтр по category проектных скилов")
    ps.add_argument("--skill", default=None, metavar="DIR",
                    help="только этот проектный скил (каталог или frontmatter name)")
    ps.add_argument("--include-archived", action="store_true",
                    help="включить проектные скилы status: archived")

    pst = sub.add_parser("status", help="состояние индекса (без эмбеддингов)")
    pst.add_argument("--root", default=None, help=argparse.SUPPRESS)
    pst.add_argument("--db", default=None, help=argparse.SUPPRESS)

    pr = sub.add_parser("reindex", help="полная пересборка (tmp + атомарная замена)")
    pr.add_argument("--root", default=None, help=argparse.SUPPRESS)
    pr.add_argument("--db", default=None, help=argparse.SUPPRESS)
    pr.add_argument("--max-new-chunks", type=int, default=1 << 30)

    pc = sub.add_parser("sync-copy",
                        help="обновить self-contained копию в "
                             "<root>/.agents/skills/project-search/")
    pc.add_argument("--root", default=None, help=argparse.SUPPRESS)
    pc.add_argument("--db", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "reindex":
        cmd_reindex(args)
    elif args.command == "sync-copy":
        cmd_sync_copy(args)


if __name__ == "__main__":
    main()
