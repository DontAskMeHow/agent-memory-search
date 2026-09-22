#!/usr/bin/env python3
"""MCP-сервер: поиск по сессиям AI-агентов (session-search).

4 инструмента:
  - session_index_status()          — когда последняя индексация, сколько источников
                                      новых/изменённых (без эмбеддингов, мгновенно)
  - session_reindex(full=False)     — явная (до)индексация с бюджетом новых чанков;
                                      при остатке — «вызови ещё раз»
  - search_sessions(query, ...)     — hybrid (BM25+vector, RRF-слияние) поиск +
                                      last_indexed_at в ответе
  - get_session_tail(session_id)    — хвост сессии из источника (всегда свежий)

Тонкая обёртка над scripts/session_search.py: тяжёлая работа — в базовом
Python 3.14 (sqlite-vec/httpx/snowballstemmer стоят без venv), сервер
запускается тем же интерпретатором (в нём есть fastmcp).
Никаких фоновых задач и демонов: принцип «агент сам решает», каждый вызов
детерминирован и ограничен бюджетом.

Подключение в mcp.json (kimi-code / kimi-cli):
    {
      "session-search": {
        "command": "python.exe",
        "args": ["session-search/mcp_server.py"],
        "env": { "PYTHONIOENCODING": "utf-8" }
      }
    }

Windows-паттерн (таймаут 150с, стриминг stdout до валидного JSON, НЕ ждать
exit — shutdown-hang, принудительный kill, `-u`, stdin=DEVNULL,
stderr=DEVNULL) живёт в общем ядре `common/mcp_common.py` (выделено в фазе 3
аудита памяти; таймауты обоих MCP-серверов синхронизированы на 150с).
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mcp_common import (  # noqa: E402
    detect_venv_python,
    run_cli,
)

SKILL_DIR = Path(__file__).resolve().parent            # .kimi-code/skills/session-search
CLI = SKILL_DIR / "scripts" / "session_search.py"
SKILLS_ROOT = SKILL_DIR.parent                         # .kimi-code/skills

# Python для CLI: venv рядом с корнем, если есть; без venv — системный
# интерпретатор (sqlite-vec, httpx, snowballstemmer стоят в базовом Python).
_PY = detect_venv_python(SKILLS_ROOT)

sys.path.insert(0, str(SKILLS_ROOT / "tools"))
from skills_lib import _MODES  # noqa: E402

mcp = FastMCP("session-search")


async def _run_cli(*args: str) -> str:
    return await run_cli(_PY, CLI, args, cwd=str(SKILLS_ROOT), proc_name="session-search")


@mcp.tool
async def session_index_status() -> str:
    """Состояние поискового индекса по сессиям: когда была последняя
    индексация, сколько источников новых/изменённых/пропавших, всего
    сессий и чанков. Вызов дешёвый (никаких эмбеддингов). Используй перед
    тем, как решить, нужен ли session_reindex."""
    return await _run_cli("status")


@mcp.tool
async def session_reindex(full: bool = False, max_chunks: int = 200) -> str:
    """Явно обновить индекс сессий (до-индексировать новые/изменённые).

    По умолчанию инкрементально (только новые/изменённые сессии) и с
    бюджетом эмбеддингов за вызов — если в ответе remaining_chunks > 0,
    вызови ещё раз, пока не станет 0. Полная пересборка (full=True) с
    недостаточным max_chunks НЕ тронет текущий индекс: вернёт ok=false с
    числом оставшихся чанков — подними max_chunks (полный корпус ~17k
    чанков) или запусти CLI напрямую. Таймаут MCP — 150 с: на полный
    корпус эмбеддинги могут не успеть, инкрементальный режим надёжнее.
    При недоступности embedding-модели вернёт ok=false — поиск при этом
    продолжит работать на старом индексе (hash mode вообще без модели).
    Перестроение сериализовано маркером .rebuild.lock: если перестроение
    уже идёт (из другого MCP-клиента), вернётся ok=false + locked=true —
    подожди и повтори. Протухший маркер (владелец мёртв >30 мин)
    перехватывается автоматически."""
    if full:
        return await _run_cli("reindex", "--max-new-chunks", str(max_chunks))
    return await _run_cli("index", "--max-new-chunks", str(max_chunks))


@mcp.tool
async def search_sessions(
    query: str,
    agent: str = "",
    workdir: str = "",
    top_k: int = 10,
    mode: str = "hybrid",
    rerank: bool = False,
) -> str:
    """Поиск по содержимому прошлых сессий AI-агентов (kimi-code, kimi-cli,
    claude code, omp). Отвечает на вопросы «над чем мы работали в такой-то
    сессии» и «нашёл ли кто-то уже решение». Возвращает JSON: results[]
    (score, agent, session_id, title, work_dir, role, is_subagent, ts, snippet,
    src_path, session_start, session_end — даты начала/конца сессии;
    is_subagent=true — сессия сабагента) + last_indexed_at.
    Если искомая сессия новее last_indexed_at — сначала вызови
    session_reindex, потом повтори поиск.

    Args:
        query: поисковый запрос (по-русски или по-английском, обычной фразой).
        agent: опционально ограничить: kimi-code | kimi-cli | claude | omp.
        workdir: опционально фильтр по рабочей директории (подстрока).
        top_k: сколько результатов (по умолчанию 10, максимум 30).
        mode: "hybrid" (BM25+vector, RRF-слияние; по умолчанию), "semantic",
              "keyword" (работает вообще без embedding-модели).
        rerank: применить BGE-reranker к топу hybrid (по умолчанию выкл —
              реранкер нестабилен на коротких диалоговых чанках; паритет
              с CLI-флагом --rerank, фаза 1 аудита р3)."""
    if mode not in _MODES:
        mode = "hybrid"
    try:
        top_k = max(1, min(int(top_k), 30))
    except (TypeError, ValueError):
        top_k = 10
    cmd = ["search", query, "--top-k", str(top_k), "--mode", mode]
    if agent:
        cmd += ["--agent", agent]
    if workdir:
        cmd += ["--workdir", workdir]
    if rerank:
        cmd += ["--rerank"]
    return await _run_cli(*cmd)


@mcp.tool
async def get_session_tail(session_id: str, agent: str = "", n: int = 20) -> str:
    """Хвост конкретной сессии — последние N сообщений (без tool-вызовов),
    читается напрямую из источника, всегда свежий. Большие wire.jsonl
    (>5 МБ) читаются с конца, окном — не целиком. Отвечает на вопрос
    «чем там всё закончилось», когда session_id уже найден через
    search_sessions или session_index_status.

    Args:
        session_id: идентификатор сессии (из результатов search_sessions;
            у сабагентов — с суффиксом /<subagent-name>).
        agent: опционально, если session_id не уникален между агентами.
        n: сколько последних сообщений (по умолчанию 20)."""
    try:
        n = max(1, min(int(n), 50))
    except (TypeError, ValueError):
        n = 20
    cmd = ["tail", session_id, "-n", str(n)]
    if agent:
        cmd += ["--agent", agent]
    return await _run_cli(*cmd)


if __name__ == "__main__":
    mcp.run(transport="stdio")
