#!/usr/bin/env python3
"""MCP-сервер: поиск по базе знаний скилов (skills-search) + поиск по проекту.

Опциональный stdio MCP-сервер для агентов, поддерживающих MCP. Даёт:
  - search_skills(query, ...)      — hybrid поиск по базе знаний
                                     (memory/records, env SKILLS_ROOT) + проектные
                                     скилы активного дерева (одним вызовом, слияние
                                     с лёгким приоритетом проектных; каждая строка
                                     с полем origin: global|project)
  - skills_search_status()         — состояние глобального индекса
  - skills_search_index()          — инкремент глобального индекса
  - search_project(query, ...)     — hybrid поиск по остальному содержимому
                                     проекта: документация (md вне скилов) +
                                     комментарии исходников (kind docs,code)
  - project_search_status()        — состояние проектного индекса
  - project_search_index()         — инкремент проектного индекса (с бюджетом)

Тонкая обёртка над `tools/skills-search.py` и `tools/project_search.py`:
тяжёлая логика выполняется в базовом Python 3.14 (sqlite-vec/httpx стоят
без venv). Корень базы знаний — env `SKILLS_ROOT`, fallback — корень
движка. Проектное дерево резолвится из аргумента
`root` или env-переменной `PROJECT_ROOT` в mcp.json; `project-search` также ищет корень «вверх от cwd», когда вызывается как CLI.

Подключение в mcp.json (kimi-code / kimi-cli):
    {
      "skills-search": {
        "command": "python.exe",
        "args": ["tools/skills_search_mcp.py"],
        "env": {
          "SKILLS_ROOT": "/path/to/knowledge",
          "PROJECT_ROOT": "/path/to/project",
          "PYTHONIOENCODING": "utf-8"
        }
      }
    }

Запускается тем же интерпретатором, что и остальные MCP-серверы (где fastmcp),
а тяжёлую логику поиска делегирует CLI через subprocess (с явными --path/--db
по SKILLS_ROOT).

ВАЖНО (Windows):
- Инструменты async и читают вывод CLI потоком, а по получении полного JSON
  принудительно завершают дочерний процесс. Дочерние CLI на этой машине после
  отработки main иногда «зависают на выходе» (shutdown-hang: единственный
  поток idle, exit не наступает), из-за чего `communicate()`/`subprocess.run()`
  висели до своего 120-секундного таймаута. Поэтому ждать exit дочернего
  процесса нельзя — только stdout.
- `-u` (unbuffered) нужен, чтобы JSON допечатывался в pipe сразу, а не
  удерживался буфером stdout до выхода процесса (который может не наступить).
- `stdin=DEVNULL` отсекает наследуемый от клиента pipe.
- `stderr=DEVNULL`: прогресс и предупреждения CLI пишутся в stderr, а читать
  его некому — заполненный pipe заблокировал бы дочерний процесс до таймаута.

Этот паттерн (таймаут 150с, стриминг stdout, принудительный kill) живёт в
общем ядре `common/mcp_common.py` (выделено в фазе 3 аудита памяти).
Паритет инструментов с session-search/mcp_server.py — фаза 6 аудита памяти.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mcp_common import (  # noqa: E402
    detect_venv_python,
    merge_origin_results,
    run_cli,
)

TOOLS_DIR = Path(__file__).resolve().parent            # .kimi-code/skills/tools
SKILLS_ROOT = Path(os.environ["SKILLS_ROOT"]) if os.environ.get("SKILLS_ROOT") \
    else TOOLS_DIR.parent                              # база знаний (fallback: корень движка)
_SEARCH = TOOLS_DIR / "skills-search.py"
_PROJECT_SEARCH = TOOLS_DIR / "project_search.py"

# Python для CLI (sqlite-vec, httpx, snowballstemmer): venv рядом с корнем
# базы, если он есть; без venv — фолбэк на интерпретатор сервера (у нас
# зависимости стоят в базовом Python 3.14).
_PY = detect_venv_python(SKILLS_ROOT)

sys.path.insert(0, str(TOOLS_DIR))
from skills_lib import (  # noqa: E402
    SEARCH_DB_PATH as _CLI_DB,
    SKILLS_PATH as _CLI_PATH,
    SOURCE_MODES as _SOURCES,
    VALID_CATEGORIES as _CATEGORIES,
    VALID_SCOPES as _SCOPES,
    _MODES,
)

mcp = FastMCP("skills-search")


async def _merged_skills_search(query: str, global_cmd: list, top_k: int,
                               scope: str, category: str, skill: str) -> str:
    """Глобальный поиск + проектный срез skill, слияние с приоритетом проектных."""
    text = await _run_cli(*global_cmd)
    try:
        global_rows = json.loads(text)
    except ValueError:
        return text
    if not isinstance(global_rows, list):  # объект ошибки глобального CLI
        return text
    root = _project_root("")
    if not root:
        for r in global_rows:
            r["origin"] = "global"
        return json.dumps(global_rows, ensure_ascii=False, indent=2)
    proj_cmd = ["search", query, "--top-k", str(max(10, int(top_k))),
                "--kind", "skill"]
    if scope:
        proj_cmd += ["--scope", scope]
    if category:
        proj_cmd += ["--category", category]
    if skill:
        proj_cmd += ["--skill", skill]
    try:
        proj_rows = json.loads(await _run_project_cli(*proj_cmd))
    except ValueError:
        proj_rows = []
    if not isinstance(proj_rows, list):
        proj_rows = []
    merged = merge_origin_results(global_rows, "global", proj_rows, "project",
                                  int(top_k))
    return json.dumps(merged, ensure_ascii=False, indent=2)


async def _run_cli(*args: str, marker: str = "[") -> str:
    """CLI skills-search с явными --path/--db (по SKILLS_ROOT из skills_lib).

    Порядок важен: глобальный --db идёт до подкоманды; --path — флаг
    index/reindex внутри неё.
    """
    argv = ["--db", str(_CLI_DB)]
    if args and args[0] in ("index", "reindex"):
        argv += [args[0], "--path", str(_CLI_PATH), *args[1:]]
    else:
        argv += list(args)
    return await run_cli(_PY, _SEARCH, argv, cwd=str(SKILLS_ROOT),
                         marker=marker, proc_name="skills-search")


async def _run_project_cli(*args: str, marker: str = "[") -> str:
    return await run_cli(_PY, _PROJECT_SEARCH, list(args), cwd=str(SKILLS_ROOT),
                         marker=marker, proc_name="project-search")


def _project_root(user_root: str) -> str:
    """Корень проекта для проектных инструментов: аргумент → env PROJECT_ROOT."""
    if user_root:
        return user_root
    return os.environ.get("PROJECT_ROOT", "")


@mcp.tool
async def search_skills(query: str, top_k: int = 10, mode: str = "hybrid",
                        scope: str = "", category: str = "", skill: str = "",
                        source: str = "knowledge", abstain: bool = True) -> str:
    """Поиск по содержимому всех записей базы знаний + проектные скилы.

    Один вызов ищет и в базе знаний (memory/records, env SKILLS_ROOT), и в
    проектных скилах активного дерева (.agents/skills; корень — env
    PROJECT_ROOT из mcp.json или аргумент root), сливает выдачу с лёгким
    приоритетом проектных канонов и метит каждую строку origin:
    global|project. Аналог grep, но по скилам/записям. Используй, когда не
    знаешь, где лежит нужное знание, вместо перечитывания SKILL.md вслепую
    или переспрашивания пользователя.

    Args:
        query: поисковый запрос (по-русски или по-английски, обычной фразой).
        top_k: сколько результатов вернуть (по умолчанию 10, максимум 50).
        mode: "hybrid" (по умолчанию: BM25 + вектор + reranker), "semantic"
              (только вектор), "keyword" (только BM25, работает без embedding).
        scope: опционально ограничить скилы их scope из frontmatter:
               personal | work | shared.
        category: опционально ограничить скилы категорией из frontmatter:
               project | infrastructure | methodology | entity | tool | reference.
        skill: опционально только один скил — имя его директории
               (или frontmatter name, например skills-tools для tools/).
        source: плоскость доступа (фаза 2 аудита р3): knowledge (дефолт;
               всё, кроме файлов задач — задачи не вытесняют знания из
               выдачи), skills (только скилы), streams, tasks (файлы задач),
               all (всё, прежнее поведение).
        abstain: абстеншен-порог (по умолчанию true). Если сигнал низкой
               уверенности сработал (v2: косинус запроса к ближайшему чанку
               индекса ниже порога калибровки golden), каждый результат несёт
               "low_confidence": true — надёжного совпадения в базе нет,
               выдачу использовать только как слабый ориентир. false —
               старая выдача без этого поля (CLI --no-abstain).

    Returns:
        JSON-массив объектов: {file, skill, heading, updated, score, snippet,
        low_confidence, origin} — самые релевантные сверху. skill:
        SKILL.md -> каноническое имя скила, файлы references/ и стримов ->
        <относительный-каталог>/<stem>. updated — дата свежести скила из
        frontmatter (null для чанков вне скилов). origin — global (запись
        базы знаний) или project (скил активного дерева) — проектные
        имеют лёгкий приоритет. low_confidence присутствует только при
        abstain=true. Скилы со status: archived в выдачу не попадают.
    """
    if mode not in _MODES:
        mode = "hybrid"
    try:
        top_k = max(1, min(int(top_k), 50))
    except (TypeError, ValueError):
        top_k = 10
    if scope and scope not in _SCOPES:
        return json.dumps(
            {"error": f"invalid scope {scope!r}, valid: {'|'.join(_SCOPES)}"},
            ensure_ascii=False)
    if category and category not in _CATEGORIES:
        return json.dumps(
            {"error": f"invalid category {category!r}, "
                      f"valid: {'|'.join(_CATEGORIES)}"},
            ensure_ascii=False)
    if source and source not in _SOURCES:
        return json.dumps(
            {"error": f"invalid source {source!r}, valid: {'|'.join(_SOURCES)}"},
            ensure_ascii=False)

    cmd = ["search", query, "--top-k", str(top_k), "--mode", mode]
    if scope:
        cmd += ["--scope", scope]
    if category:
        cmd += ["--category", category]
    if skill:
        cmd += ["--skill", skill]
    if source:
        cmd += ["--source", source]
    if abstain:
        cmd += ["--abstain"]
    else:
        cmd += ["--no-abstain"]
    return await _merged_skills_search(query, cmd, top_k, scope, category, skill)


@mcp.tool
async def skills_search_status() -> str:
    """Состояние поискового индекса базы знаний: сколько файлов/чанков/векторов,
    размер БД, топ скилов по чанкам. Вызов дешёвый (никаких эмбеддингов).
    Используй, чтобы понять, насколько свежий индекс, прежде чем звать
    skills_search_index."""
    return await _run_cli("status", "--json", marker="{")


@mcp.tool
async def skills_search_index() -> str:
    """Инкрементально доиндексировать базу знаний (новые/изменённые файлы;
    удалённые выкидываются). После индексации возвращает свежий статус
    индекса (файлы/чанки/векторы). Полная пересборка — только CLI
    `skills-search reindex`. Индексация требует доступности embedding-модели
    (BGE-M3): если она недоступна, вернёт error — поиск продолжит работать
    на старом индексе."""
    res = await _run_cli("index", "--json", marker="{")
    try:
        if "error" in json.loads(res):
            return res
    except (ValueError, TypeError):
        return json.dumps({"error": "unexpected index output"},
                          ensure_ascii=False)
    return await _run_cli("status", "--json", marker="{")


@mcp.tool
async def search_project(query: str, top_k: int = 10, mode: str = "hybrid",
                         kind: str = "docs,code", scope: str = "",
                         category: str = "", skill: str = "",
                         root: str = "", abstain: bool = True) -> str:
    """Поиск по ПРОЕКТУ: документация и комментарии исходников.

    Проектный индекс лежит в дереве проекта (<root>/.agents/.index/),
    наполняется инструментом project_search_index. Скилы проекта ищутся
    инструментом search_skills (он отдаёт глобальные и проектные скилы одним
    вызовом). Здесь — остальное содержимое: AGENTS.md/README/documentation
    (kind=docs) и комментарии из исходников (kind=code; тела кода не
    индексируются — по коду ищи Grep).

    Args:
        query: поисковый запрос (по-русски или по-английски, обычной фразой).
        top_k: сколько результатов вернуть (по умолчанию 10, максимум 50).
        mode: "hybrid" (по умолчанию), "semantic", "keyword" (работает без
              embedding-модели).
        kind: какие виды чанков искать: "docs,code" (по умолчанию), "docs",
              "code", "skill", "all" (список через запятую).
        scope / category / skill: фильтры по проектным скилам (нужны при kind,
              включающем skill).
        root: корень проекта (иначе env PROJECT_ROOT из mcp.json).
        abstain: порог уверенности (по умолчанию true) — при низком сигнале
              каждая строка несёт "low_confidence": true.

    Returns:
        JSON-массив {file, kind, skill, heading, line, score, snippet,
        low_confidence}. kind=code — стартовая строка комментария.
    """
    if mode not in _MODES:
        mode = "hybrid"
    try:
        top_k = max(1, min(int(top_k), 50))
    except (TypeError, ValueError):
        top_k = 10
    proj_root = _project_root(root)
    if not proj_root:
        return json.dumps(
            {"error": "проект не найден: передай root или задай PROJECT_ROOT "
                      "в mcp.json (см. скил agent-configs)"},
            ensure_ascii=False)
    cmd = ["search", query, "--top-k", str(top_k), "--mode", mode,
           "--kind", kind]
    if scope:
        cmd += ["--scope", scope]
    if category:
        cmd += ["--category", category]
    if skill:
        cmd += ["--skill", skill]
    cmd += ["--abstain"] if abstain else ["--no-abstain"]
    return await _run_project_cli(*cmd)


@mcp.tool
async def project_search_status(root: str = "") -> str:
    """Состояние проектного поискового индекса: файлы/чанки/векторы, разбивка
    по видам (skill/docs/code), размер БД, last_index_at. Дёшево (без
    эмбеддингов)."""
    proj_root = _project_root(root)
    if not proj_root:
        return json.dumps(
            {"error": "проект не найден: передай root или задай PROJECT_ROOT"},
            ensure_ascii=False)
    return await _run_project_cli("status", marker="{")


@mcp.tool
async def project_search_index(root: str = "", max_new_chunks: int = 300) -> str:
    """Инкрементально доиндексировать проект (новые/изменённые файлы; удалённые
    выкидываются). Полная пересборка — CLI `project-search reindex`. Требует
    доступную embedding-модель (BGE-M3): при недоступности вернёт error, а
    поиск продолжит работать на старом индексе (keyword-режим — и без модели)."""
    proj_root = _project_root(root)
    if not proj_root:
        return json.dumps(
            {"error": "проект не найден: передай root или задай PROJECT_ROOT"},
            ensure_ascii=False)
    res = await _run_project_cli("index", "--max-new-chunks",
                                 str(max(1, int(max_new_chunks))), "--json",
                                 marker="{")
    try:
        if "error" in json.loads(res):
            return res
    except (ValueError, TypeError):
        return json.dumps({"error": "unexpected index output"},
                          ensure_ascii=False)
    return await _run_project_cli("status", marker="{")


if __name__ == "__main__":
    mcp.run(transport="stdio")
