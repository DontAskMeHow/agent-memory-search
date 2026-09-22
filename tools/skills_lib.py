#!/usr/bin/env python3
"""skills_lib — общие примитивы для обслуживания базы знаний (stdlib-only).

Выделено в фазе 3 аудита памяти из дублировавшихся копий в скриптах tools/:
frontmatter-парсер (4 копии с разным поведением), сканер SKILL.md (2 копии),
DDL graph.db (2 копии), SKIP_DIRS (3 варианта), пути, entity-конвенции,
utf8_stdout. Скрипты остаются CLI-first и просто импортируют этот модуль
(лежит рядом, в tools/).

Канонический parse_frontmatter — супермножество всех четырёх бывших копий:
простые `key: value`; block scalars `>` (folded: строки склеиваются пробелами)
и `|` (literal: переносы сохраняются); inline arrays `[a, b, c]`; dash-списки
(`key:\n  - a\n  - b` → list); inline-комментарии (`value  # коммент` →
коммент отбрасывается, только при пробеле перед #); кириллица.
"""

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# --- Пути ---------------------------------------------------------------------

TOOLS_ROOT = Path(__file__).resolve().parent.parent  # корень движка (.kimi-code/skills)

# Корень базы знаний (каталог записей). Env SKILLS_ROOT — при адаптации под
# layout: база знаний — отдельный каталог записей.
# Без переменной — сам движок (старое поведение коллег: база = каталог скилов
# в корне системы, производные каталоги — внутри неё).
if os.environ.get("SKILLS_ROOT"):
    SKILLS_PATH = Path(os.environ["SKILLS_ROOT"])
    # База знаний лежит вне движка (memory/records): производные артефакты
    # кладём в каталог-родитель базы — memory/.search, memory/.graph.
    SEARCH_DB_PATH = SKILLS_PATH.parent / ".search" / "index.db"
    GRAPH_DIR = SKILLS_PATH.parent / ".graph"
else:
    SKILLS_PATH = TOOLS_ROOT
    SEARCH_DB_PATH = SKILLS_PATH / ".search" / "index.db"
    GRAPH_DIR = SKILLS_PATH / ".graph"
GRAPH_DB_PATH = GRAPH_DIR / "graph.db"
PROPOSALS_PATH = GRAPH_DIR / "proposals.json"
TASKS_PATH = SKILLS_PATH / "tasks"

# Каталоги, исключаемые из обхода базы знаний: VCS/venv-мусор, производные
# индексы (.search/.graph) и кэши инструментов (.pytest_cache — раньше его
# README.md попадал в поисковый индекс как шум; находка фазы 2 аудита).
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules",
             ".search", ".graph", ".pytest_cache"}

# --- Enum-константы (общие для CLI и MCP-обёрток) ----------------------------

VALID_SCOPES = ("personal", "work", "shared")
VALID_CATEGORIES = ("project", "infrastructure", "methodology",
                    "entity", "tool", "reference")
_MODES = ("hybrid", "semantic", "keyword")
# Плоскости доступа поиска (фаза 2 аудита р3): knowledge — дефолт (всё,
# кроме файлов задач), skills — каталоги скилов, streams, tasks — файлы
# задач (операционный слой), all — без фильтра.
SOURCE_MODES = ("knowledge", "skills", "streams", "tasks", "all")

# --- Frontmatter: парсинг -----------------------------------------------------

def _strip_inline_comment(val: str) -> str:
    """Отбросить inline-комментарий: `value  # коммент` → `value`.

    Комментарий признаётся только при пробеле перед `#` — URL с якорем
    (`http://x#frag`) не режется.
    """
    if " #" in val:
        val = val.split(" #", 1)[0]
    return val.rstrip()


def _is_dash_item(line: str) -> bool:
    """Строка — элемент dash-списка (`- пункт` / `-`)."""
    s = line.lstrip()
    return s.startswith("- ") or s == "-"


def _split_inline_array(val: str) -> list:
    """`a, b, 'c d'` → ['a', 'b', 'c d'] (кавычки снимаются)."""
    return [v.strip().strip("'\"") for v in val.split(",") if v.strip()]


def parse_frontmatter(text: str) -> tuple:
    """Разобрать YAML-frontmatter markdown-файла → (dict, body).

    Без зависимости от PyYAML. Поддерживает (канонический супермножество
    всех бывших копий в skills-search/skills-maintain/skills-graph/tasks):
    - простые `key: value`;
    - block scalar `key: >` — folded: непустые строки склеиваются пробелами;
    - block scalar `key: |` — literal: переносы строк сохраняются;
    - inline arrays `key: [a, b, c]`;
    - dash-списки `key:` + строки `  - a` → list;
    - inline-комментарии `value  # коммент` → коммент отбрасывается;
    - кириллицу и значения в кавычках.

    Файл без frontmatter (или без закрывающего `---`) → ({}, text).
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    fm_text = text[3:end].strip()
    body = text[end + 4:].strip()

    fm = {}
    lines = fm_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in stripped:
            i += 1
            continue
        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.strip()
        if not key:
            i += 1
            continue

        # Block scalar: "description: >" (folded) или "description: |" (literal)
        if val in (">", "|"):
            block_lines = []
            i += 1
            while i < len(lines):
                bl = lines[i]
                # continuation: индентированные строки (пустые допустимы)
                if bl.startswith("  ") or bl.startswith("\t") or bl.strip() == "":
                    block_lines.append(bl.strip())
                else:
                    break
                i += 1
            if val == ">":
                fm[key] = " ".join(ln for ln in block_lines if ln)
            else:
                fm[key] = "\n".join(block_lines).strip()
            continue

        val = _strip_inline_comment(val)

        # Dash-список: "related:" + следующие строки "  - a", "  - b"
        if val == "" and i + 1 < len(lines) and _is_dash_item(lines[i + 1]):
            items = []
            i += 1
            while i < len(lines) and _is_dash_item(lines[i]):
                item = lines[i].strip()[1:].strip()
                item = _strip_inline_comment(item).strip().strip("'\"")
                if item:
                    items.append(item)
                i += 1
            fm[key] = items
            continue

        # Inline array: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            fm[key] = _split_inline_array(val[1:-1])
        else:
            fm[key] = val.strip("'\"")
        i += 1
    return fm, body


# --- Frontmatter: обновление --------------------------------------------------

def _format_fm_field(key: str, value) -> str:
    """Сериализация поля frontmatter; списки — inline `[a, b]`."""
    if isinstance(value, list):
        return f"{key}: [{', '.join(str(v) for v in value)}]"
    return f"{key}: {value}"


def update_frontmatter(text: str, updates: dict, create: bool = True) -> str:
    """Обновить поля frontmatter, не трогая остальной файл.

    - Существующее значение ключа заменяется; списки пишутся inline `[a, b]`.
    - Continuation-строки заменяемого ключа (block scalar `>`/`|` или
      dash-список) поглощаются — orphan-строки не остаются.
    - Новые ключи дописываются в конец блока frontmatter.
    - Хвост файла (от закрывающего `---`) передаётся байт-в-байт.

    `create=True`: текст без frontmatter получает новый блок (семантика
    skills-maintain). `create=False`: такой текст возвращается без изменений
    (семантика tasks.set_fm_field).
    """
    if not text.startswith("---"):
        if not create:
            return text
        lines = ["---"]
        lines.extend(_format_fm_field(k, v) for k, v in updates.items())
        lines.append("---")
        return "\n".join(lines) + "\n" + text

    end = text.find("\n---", 3)
    if end == -1:
        return text

    fm_lines = text[3:end].split("\n")
    after = text[end:]  # хвост от "\n---" — без изменений

    new_lines = []
    updated_keys = set()
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        stripped = line.strip()

        # Ключ — только top-level строка (без отступа): индентированные
        # строки — continuation чужого block scalar/dash-списка, их не трогаем
        # (regex-версия tasks.py матчила так же; строчная версия
        # skills-maintain могла falsely заменить строку внутри `>`-блока).
        matched_key = None
        if not line.startswith((" ", "\t")):
            for key in updates:
                if stripped.startswith(key + ":"):
                    matched_key = key
                    break

        if matched_key is None:
            new_lines.append(line)
            i += 1
            continue

        val = stripped.partition(":")[2].strip()
        i += 1
        # поглотить continuation-строки старого значения
        if val in (">", "|"):
            while i < len(fm_lines) and (fm_lines[i].startswith("  ") or
                                         fm_lines[i].startswith("\t") or
                                         fm_lines[i].strip() == ""):
                i += 1
        elif val == "":
            # dash-список: indented "- item" строки принадлежат заменяемому ключу
            while i < len(fm_lines) and _is_dash_item(fm_lines[i]):
                i += 1
        new_lines.append(_format_fm_field(matched_key, updates[matched_key]))
        updated_keys.add(matched_key)

    for key, val in updates.items():
        if key not in updated_keys:
            new_lines.append(_format_fm_field(key, val))

    return "---" + "\n".join(new_lines) + after


def set_fm_field(text: str, key: str, value) -> str:
    """Заменить/добавить одно поле frontmatter, не трогая остальное.

    Текст без frontmatter возвращается без изменений (семантика tasks.py).
    """
    return update_frontmatter(text, {key: value}, create=False)


# --- Сканер скилов ------------------------------------------------------------

def find_all_skills(skills_path=None) -> list:
    """Найти все SKILL.md в базе знаний и распарсить frontmatter.

    Возвращает список dict: name (из frontmatter или имя каталога),
    file (rel-путь с /), full_path и path (абсолютный путь), frontmatter,
    body, full_text. Обход — с общим SKIP_DIRS.
    """
    root = Path(skills_path) if skills_path else SKILLS_PATH
    skills = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if "SKILL.md" not in files:
            continue
        full_path = Path(dirpath) / "SKILL.md"
        rel_path = str(full_path.relative_to(root)).replace("\\", "/")
        try:
            text = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        fm, body = parse_frontmatter(text)
        name = fm.get("name", Path(dirpath).name)
        skills.append({
            "name": name,
            "file": rel_path,
            "full_path": str(full_path),
            "path": str(full_path),
            "frontmatter": fm,
            "body": body,
            "full_text": text,
        })
    return skills


# --- Канонизация владельца чанка ----------------------------------------------

def skill_dir_of(file_path: str) -> str:
    """Каноническое имя скила-владельца чанка (rel-путь файла).

    Единая канонизация для поиска, eval (golden set) и dedup.
    В нашем layout (база = memory/records, пути относительны от неё):
    'dgx-spark-machines/SKILL.md' -> 'dgx-spark-machines',
    'skills-memory-adopt/README.md' -> 'skills-memory-adopt' (стрим),
    корневой файл ('README.md') -> '(root)'.
    Фаза 4 аудита р4: перенесена из skills-search.py — dedup'у
    нужен владелец по ПУТИ: у файлов references/ skill_name хранит stem
    файла, который коллизирует между скилами.
    """
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return "(root)"
    head = parts[0]
    if head == "tools":
        return "skills-tools"
    if head == "streams":
        return f"streams/{parts[1]}" if len(parts) > 2 else "streams"
    return head


# --- Graph DB -----------------------------------------------------------------

def init_graph_db(db_path=None) -> sqlite3.Connection:
    """Инициализировать graph.db (DDL: entities, edges, skill_entities).

    Граф — полностью производный артефакт: единственный источник данных —
    frontmatter SKILL.md (related + entities), наполняется `skills-graph
    build` без LLM. Схема упрощена в фазе 4 аудита: колонки
    entities.description и edges.confidence никогда не читались — убраны.
    entities.type заполняется только для скилов (`skill:<category>`);
    технологические сущности имеют NULL.

    Возвращает ОТКРЫТОЕ соединение — вызывающий закрывает его сам.
    """
    db_path = Path(db_path) if db_path else GRAPH_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            name TEXT PRIMARY KEY,
            type TEXT,
            first_seen TEXT,
            last_seen TEXT
        );
        CREATE TABLE IF NOT EXISTS edges (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT NOT NULL DEFAULT 'related_to',
            source_skill TEXT,
            created_at TEXT,
            PRIMARY KEY (source, target, relation)
        );
        CREATE TABLE IF NOT EXISTS skill_entities (
            skill_name TEXT NOT NULL,
            entity_name TEXT NOT NULL,
            PRIMARY KEY (skill_name, entity_name)
        );
    """)
    conn.commit()
    return conn


# --- Entity-конвенции ---------------------------------------------------------

ENTITY_PREFIX = "entity-"
ENTITY_SKILL_THRESHOLD = 3  # сколько скилов должны упоминать сущность

# Шум для entity-номенклатуры: общие слова, инфраструктурные ярлыки,
# внутренние аббревиатуры (не настоящие кросс-проектные сущности).
ENTITY_NOISE_STOPLIST = {
    "report", "дкур", "дмимс", "avroconfluent", "федоровское",
    "python", "docker", "kubernetes", "k8s",
}

# Понятия, покрытые существующим каноническим скилом: проверка
# missing_entity не считает их кандидатами на отдельный entity-скил.
def _load_entity_coverage() -> dict:
    """Покрытие понятий каноническими скилами — из tools/entity_coverage.json
    (фаза 4 аудита р3: вынесено из кода — знание в данных; правка маппинга
    больше не требует правки Python). Файл обязателен — отсутствие/битый JSON
    валит импорт сразу и с понятной ошибкой, а не молча меняет поведение lint.
    Ключ `_comment` игнорируется."""
    path = Path(__file__).resolve().parent / "entity_coverage.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"entity_coverage.json не читается ({path}): {e}. "
            f"Восстанови из git или создай заново (см. skills-management, "
            f"раздел «Сущности»).") from e
    return {k: v for k, v in data.items() if not k.startswith("_")}


ENTITY_COVERED_BY = _load_entity_coverage()


def entity_skill_name(entity: str) -> str:
    """Имя entity-скила для сущности: `ClickHouse` → `entity-clickhouse`."""
    return ENTITY_PREFIX + entity.lower().replace(" ", "-")


def is_entity_skill(skill_name: str) -> bool:
    """Скил — entity-скил по конвенции имён `entity-*`?"""
    return skill_name.startswith(ENTITY_PREFIX)


def is_noise_entity(name: str) -> bool:
    """Эвристика: извлечение шума, а не настоящая кросс-проектная сущность.

    Короткие (<3), слова из stoplist и кириллические аббревиатуры ВЕРХНИМ
    регистром (<=6 символов) — шум. Используется lint'ом и graph orphans;
    до фазы 3 был только в skills-maintain (orphans расходился с lint).
    """
    n = (name or "").strip()
    if len(n) < 3 or n.lower() in ENTITY_NOISE_STOPLIST:
        return True
    if n.isupper() and any("\u0400" <= c <= "\u04ff" for c in n) and len(n) <= 6:
        return True
    return False


# --- Прочее -------------------------------------------------------------------

def load_script(script_path, module_name=None):
    """Загрузить .py-скрипт с дефисом в имени как модуль (importlib).

    Общий хелпер фазы 4 аудита р4 для потребителей CLI-скриптов как модулей:
    tasks.py (find), skills-maintain.py (check-docs), conftest и eval_search —
    раньше паттерн spec_from_file_location копировался четырежды. Каждый вызов
    exec'ит модуль заново (кэширование — на вызывающем). Без подавления
    stdout: загружаемые скрипты на import ничего не печатают.
    """
    import importlib.util

    path = Path(script_path)
    spec = importlib.util.spec_from_file_location(
        module_name or f"_loaded_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"не удалось загрузить скрипт {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def drop_db_files(db_path) -> None:
    """Удалить файл БД вместе с WAL/SHM-сиблингами: осиротевшие -wal/-shm
    от старого индекса вытекали бы в пересобранный (общая копия из
    skills-search и session_search, фаза 1 аудита р3)."""
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()


def utf8_stdout() -> None:
    """Переключить stdout на UTF-8, если он не UTF-8 (Windows: cp1251 в pipe)."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
