#!/usr/bin/env python3
"""
skills-maintain: automated memory maintenance for the skills knowledge base.

Commands:
    lint [--fix] [--stale-days N] [--stream-stale-days N] [--reciprocal-full]  # integrity checks
    dedup                                                   # semantic duplicates
    check-docs                                              # сверка доков с кодом (без сети/LLM)

С фазы 4 аудита памяти lint и dedup работают без сети и LLM (enrich вынесен
в tools/skills-enrich.py). Lint читает frontmatter скилов напрямую и не
зависит от .graph/graph.db: источник данных для missing_entity — поле
`entities:` в frontmatter скилов (source of truth в git).

Проверки lint (полный список; таблица должна совпадать с tools/SKILL.md
и skills-management/SKILL.md):
    no_related          скил без поля related
    no_category         скил без поля category
    broken_ref          related указывает на несуществующий скил (чинит --fix)
    reciprocal_ref      односторонняя связь: A ссылается на B, B не ссылается на A (informational: не входит в Total, полный список --reciprocal-full)
    stale               updated старше STALE_DAYS дней
    missing_entity      сущность в ENTITY_THRESHOLD+ скилах без entity-скила (понятия из ENTITY_COVERED_BY пропускаются)
    orphan              нет related И никто не ссылается через свой related
    empty_description   description отсутствует или короче MIN_DESCRIPTION_LENGTH
    index_drift         INDEX.md расходится с записями на диске (у нас — таблица
                        memory/INDEX.md, первая колонка records/<name>/)
    stale_streams       стрим (category: stream) со status: working и updated
                        старше STREAM_STALE_DAYS дней
    stream_no_readme    запись с category: stream без README.md
    skill_too_long      SKILL.md длиннее SKILL_LINE_LIMIT строк
    uncommitted         незакоммиченные изменения в git (warning: чек-лист требует коммитить сразу)
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

from skills_lib import (
    ENTITY_COVERED_BY,
    ENTITY_SKILL_THRESHOLD,
    SEARCH_DB_PATH,
    SKILLS_PATH,
    SKIP_DIRS,
    TOOLS_ROOT,
    entity_skill_name,
    find_all_skills,
    is_noise_entity,
    load_script as _load_script,
    parse_frontmatter,
    skill_dir_of,
    update_frontmatter,
    utf8_stdout,
)
from search_core import init_vec_conn as _init_vec_conn

# Windows: cp1251-консоль не умеет «→» в выводах lint (broken refs)
utf8_stdout()

# --- Пороги (lint-пороги переопределяются CLI-флагами) ------------------------

STALE_DAYS = 90                              # stale: updated старше N дней
STREAM_STALE_DAYS = 30                       # stale_streams: стрим без апдейтов
ENTITY_THRESHOLD = ENTITY_SKILL_THRESHOLD    # missing_entity: сущность в N+ скилах
MIN_DESCRIPTION_LENGTH = 30                  # empty_description: мин. длина
SKILL_LINE_LIMIT = 1000                      # skill_too_long: SKILL.md длиннее N строк
DEDUP_SIM_THRESHOLD = 0.92                   # dedup: порог косинусной близости
DEDUP_K = 3                                  # dedup: соседей на чанк

if os.environ.get("SKILLS_ROOT"):
    INDEX_PATH = SKILLS_PATH.parent / "INDEX.md"
else:
    INDEX_PATH = SKILLS_PATH / "skills-management" / "references" / "index.md"


# --- check-docs: сверка доков с кодом (код — источник истины) -----------------

# Доки, сверяемые с константами кода: маркеры вида STALE_DAYS=90 в тексте.
# При SKILLS_ROOT каноны живут в references/ этого скилла; без — рядом с tools/ (старое поведение коллег).
if os.environ.get("SKILLS_ROOT"):
    DOC_DIR = TOOLS_ROOT / "references"
    DOC_TOOLS = DOC_DIR / "tools.md"
    DOC_MGMT = DOC_DIR / "skills-management" / "SKILL.md"
    MODELS_CONFIG_PATH = TOOLS_ROOT / "tools" / "models.json"
else:
    DOC_DIR = SKILLS_PATH
    DOC_TOOLS = DOC_DIR / "tools" / "SKILL.md"
    DOC_MGMT = DOC_DIR / "skills-management" / "SKILL.md"
    MODELS_CONFIG_PATH = SKILLS_PATH / "tools" / "models.json"
DOC_PATHS = {
    "tools/SKILL.md": DOC_TOOLS,
    "skills-management/SKILL.md": DOC_MGMT,
}

_NAMED_NUM_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*(\d+(?:\.\d+)?)")
# Константы-не-числа: проверяется, что значение встречается в доке подстрокой
_TEXT_CONSTS: set = set()
# Константы, чьё значение живёт в env/локальном конфиге машины и в доке не
# печатается: сверяется упоминание соответствующей env-переменной
_ENV_CONSTS = {"RERANKER_URL": "SKILLS_RERANKER_URL"}


def _const_spec() -> list:
    """[(module, name)] — именованные константы кода, сверяемые с доками."""
    here = Path(__file__).resolve()
    tools_dir = here.parent
    ss = _load_script(tools_dir / "skills-search.py", "_checkdocs_skills_search")
    tasks_mod = _load_script(tools_dir / "tasks.py", "_checkdocs_tasks")
    search_core = _load_script(tools_dir / "search_core.py", "_checkdocs_search_core")
    mcp_common = _load_script(TOOLS_ROOT / "common" / "mcp_common.py",
                              "_checkdocs_mcp_common")
    g = globals()
    return [
        (g, "STALE_DAYS"),                 # skills-maintain
        (g, "STREAM_STALE_DAYS"),
        (g, "MIN_DESCRIPTION_LENGTH"),
        (g, "SKILL_LINE_LIMIT"),
        (g, "DEDUP_SIM_THRESHOLD"),
        (g, "DEDUP_K"),
        (tasks_mod, "SCAN_DEADLINE_SOON_DAYS"),   # tasks.py: пороги scan
        (tasks_mod, "SCAN_INBOX_IDLE_DAYS"),
        (tasks_mod, "SCAN_OPEN_STALE_DAYS"),
        (ss, "CHUNK_SIZE"),                        # skills-search: чанкинг
        (ss, "DESC_PREFIX_MAX"),
        (search_core, "EMBED_BATCH"),              # search_core: модель/ранжирование
        (search_core, "EMBEDDING_DIM"),
        (search_core, "RRF_K"),
        (search_core, "RERANKER_TOP_N"),
        (search_core, "RERANKER_URL"),
        (search_core, "ABS_ABSENT_THRESHOLD"),     # search_core: абстеншен-порог
        (search_core, "MARGIN_MIN"),
        (search_core, "COS_ABSENT_THRESHOLD"),     # search_core: абстеншен v2 (косинус)
        (mcp_common, "MCP_TIMEOUT"),
    ]


def _doc_named_numbers(text: str) -> dict:
    """Именованные числа доков: {'STALE_DAYS': '90', ...}."""
    return {m.group(1): m.group(2) for m in _NAMED_NUM_RE.finditer(text)}


def _code_lint_checks() -> list:
    """Список проверок lint из docstring модуля (источник истины).

    Блок «Проверки lint ...» в docstring — единственное место, откуда
    берутся имена: две колонки `имя  описание` с отступом 4.
    """
    names = []
    in_checks = False
    for line in __doc__.splitlines():
        if line.startswith("Проверки lint"):
            in_checks = True
            continue
        if not in_checks:
            continue
        m = re.match(r"^\s{4}([a-z_]{2,})\s{2,}", line)
        if m:
            names.append(m.group(1))
        elif names:
            break
    return names


def _doc_table_cells(text: str) -> list:
    """Первые ячейки markdown-таблиц с нижнерегистровым идентификатором."""
    cells = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*`?([a-z_]{2,})`?\s*\|", line)
        if m:
            cells.append(m.group(1))
    return cells


def _doc_model_rows(text: str) -> list:
    """Строки таблицы моделей: [(номер, route, модель), ...]."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*`?([\w/.\-]+)`?\s*\|\s*([\w.\-]{2,})", line)
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3)))
    return rows


def _check_models(doc_text: str) -> list:
    """Таблица моделей tools/SKILL.md против tools/models.json (имя и порядок)."""
    issues = []
    try:
        cfg = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [f"tools/SKILL.md: models.json не читается ({e})"]
    models = cfg.get("models", [])
    rows = _doc_model_rows(doc_text)
    if not rows:
        return ["tools/SKILL.md: таблица моделей не найдена"]
    for i, m in enumerate(models):
        if i >= len(rows):
            issues.append(f"models: в доке нет строки #{i + 1} ({m.get('route')})")
            continue
        _, route, name = rows[i]
        if route != m.get("route"):
            issues.append(f"models: #{i + 1} роут в доке «{route}», код «{m.get('route')}»")
        if m.get("model"):
            if m["model"] != name:
                issues.append(f"models: #{i + 1} модель в доке «{name}», код «{m['model']}»")
        elif m.get("note") and name not in m["note"]:
            issues.append(f"models: #{i + 1} «{name}» не упомянута в models.json ({m.get('route')})")
    if len(rows) != len(models):
        issues.append(f"models: строк в доке {len(rows)} != {len(models)}")
    return issues


def run_check_docs() -> list:
    """Сверка доков с кодом: маркеры-числа, таблица lint, таблица моделей.

    Возвращает список проблем; пустой список = «docs in sync». Ошибок не
    бросает: любая находка — строка `док: причина` (формат lint).
    """
    issues = []
    spec = _const_spec()
    for label, path in DOC_PATHS.items():
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            issues.append(f"{label}: не читается ({e})")
            continue
        found = _doc_named_numbers(text)
        for mod, name in spec:
            val = mod.get(name) if isinstance(mod, dict) else getattr(mod, name)
            if name in _TEXT_CONSTS:
                if isinstance(val, str) and val not in text:
                    issues.append(f"{label}: нет «{val}» ({name})")
                continue
            if name in _ENV_CONSTS:
                if _ENV_CONSTS[name] not in text:
                    issues.append(f"{label}: нет env «{_ENV_CONSTS[name]}» ({name})")
                continue
            if name not in found:
                issues.append(f"{label}: нет маркера {name} (код: {val})")
            elif float(found[name]) != float(val):
                issues.append(f"{label}: {name}={found[name]} в доке, код: {val}")

    code_checks = _code_lint_checks()
    for label, path in DOC_PATHS.items():
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        cells = sorted({c for c in _doc_table_cells(text) if c in code_checks})
        if cells != sorted(code_checks):
            missing = sorted(set(code_checks) - set(cells))
            extra = sorted(set(cells) - set(code_checks))
            issues.append(f"{label}: таблица lint != коду (нет: {missing}, лишние: {extra})")

    try:
        issues += _check_models(DOC_PATHS["tools/SKILL.md"].read_text(encoding="utf-8"))
    except OSError as e:
        issues.append(f"tools/SKILL.md: не читается ({e})")
    return issues


def cmd_check_docs(args):
    """Сверка доков с кодом (без сети и LLM); пусто — «docs in sync»."""
    issues = run_check_docs()
    if not issues:
        print("docs in sync")
        return
    print(f"### Docs out of sync ({len(issues)})")
    for s in issues:
        print(f"  - {s}")


# --- Commands: lint ----------------------------------------------------------

def cmd_lint(args):
    """Check knowledge base integrity."""
    skills = find_all_skills()
    skill_names = {s["name"] for s in skills}
    # Стримы — каталоги с README.md; их имена валидны для related.
    if SKILLS_PATH.is_dir():
        for d in sorted(SKILLS_PATH.iterdir()):
            if not (d.is_dir() and not d.name.startswith(("_", "."))):
                continue
            if d.name in SKIP_DIRS:
                continue
            readme = d / "README.md"
            if readme.exists():
                try:
                    fm, _ = parse_frontmatter(readme.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, OSError):
                    continue
                if fm.get("name"):
                    skill_names.add(str(fm["name"]))
    issues = {"orphans": [], "broken_refs": [], "stale": [],
              "missing_entities": [], "no_category": [], "no_related": [],
              "empty_description": [], "long_skills": [], "stream_no_readme": [],
              "uncommitted": [], "reciprocal_refs": []}

    # Входящие ссылки: кто ссылается на скил через свой related (self-links не считаются)
    incoming_refs: set[str] = set()

    # related каждого скила (для reciprocity-проверки ниже)
    related_map: dict[str, set] = {}

    # Entity frequency counter — из frontmatter скилов, не из graph.db
    entity_counts: dict[str, int] = {}

    for s in skills:
        fm = s["frontmatter"]
        name = s["name"]

        # Check related
        related = fm.get("related", [])
        if isinstance(related, str):
            related = [related]
        related = [r for r in related if r]
        if not related:
            issues["no_related"].append(name)
        related_map[name] = set(related)

        # Check broken refs + собрать входящие ссылки
        for ref in related:
            if ref != name:
                incoming_refs.add(ref)
            if ref not in skill_names:
                issues["broken_refs"].append(f"{name} → {ref}")

        # Check category
        if not fm.get("category"):
            issues["no_category"].append(name)

        # Check description (отсутствует или короче MIN_DESCRIPTION_LENGTH)
        desc = fm.get("description", "")
        if isinstance(desc, list):
            desc = " ".join(desc)
        desc = (desc or "").strip()
        if len(desc) < MIN_DESCRIPTION_LENGTH:
            why = "no description" if not desc else f"{len(desc)} chars"
            issues["empty_description"].append(f"{name} ({why})")

        # Check SKILL.md size (число строк == wc -l)
        lines = s["full_text"].count("\n")
        if not s["full_text"].endswith("\n"):
            lines += 1
        if lines > SKILL_LINE_LIMIT:
            issues["long_skills"].append(f"{name} ({lines} lines)")

        # Check stale
        updated = fm.get("updated", "")
        if updated:
            try:
                d = datetime.fromisoformat(updated)
                age_days = (datetime.now() - d).days
                if age_days > args.stale_days:
                    issues["stale"].append(f"{name} (updated {updated}, {age_days}d ago)")
            except (ValueError, TypeError):
                pass

        # Count entities from frontmatter
        entities = fm.get("entities", [])
        if isinstance(entities, str):
            entities = [entities]
        for ent in entities:
            if ent:
                entity_counts[ent] = entity_counts.get(ent, 0) + 1

    # Orphans: нет related И нет входящих ссылок (настоящий orphan-детектор)
    for s in skills:
        if not s["frontmatter"].get("related") and s["name"] not in incoming_refs:
            issues["orphans"].append(s["name"])

    # Reciprocal refs (фаза 6 аудита р3): односторонние related-ссылки —
    # A ссылается на B, B не ссылается обратно. Informational, не входит
    # в Total issues: асимметрия — не ошибка целостности, а конвенционный
    # запах (двусторонние связи улучшают навигацию; LLM-Wiki, +2–8 F1 —
    # веб-исследование аудита р3). На старте проверки в базе 241 асимметрия —
    # массовый бэкфилл раздул бы related хаб-скилов (alpine-vm-infra, 23
    # входящих), потому чинится точечно. Self-links и битые refs (broken_ref)
    # здесь не рассматриваются.
    for s in skills:
        name = s["name"]
        for ref in sorted(related_map.get(name, ())):
            if ref != name and ref in skill_names \
                    and name not in related_map.get(ref, set()):
                issues["reciprocal_refs"].append(f"{name} → {ref} (нет обратной)")

    # Missing entity-skills (entity in N+ skills but no entity-skill;
    # понятия из ENTITY_COVERED_BY покрыты каноническим скилом и пропускаются)
    for ent, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        if is_noise_entity(ent):
            continue
        if ent in ENTITY_COVERED_BY:
            continue
        if count >= ENTITY_THRESHOLD:
            entity_skill = entity_skill_name(ent)
            if entity_skill not in skill_names:
                issues["missing_entities"].append(f"{ent} ({count} skills)")

    # Check streams. У нас стримы — каталоги записей <name>/README.md с
    # frontmatter category: stream (база знаний в SKILLS_PATH). Протухший
    # стрим — status: working и updated старше порога; архив (archived) не
    # протухает никогда. stream_no_readme — запись, помеченная stream, но
    # без README.md (у нас не возникает, у коллег — каталоги streams/).
    stale_streams = []
    if SKILLS_PATH.is_dir():
        for d in sorted(SKILLS_PATH.iterdir()):
            if not (d.is_dir() and not d.name.startswith(("_", "."))):
                continue
            if d.name in SKIP_DIRS:
                continue
            readme = d / "README.md"
            if readme.exists():
                try:
                    fm, _ = parse_frontmatter(readme.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, OSError):
                    fm = {}
                if fm.get("category") != "stream":
                    continue
                status = (fm.get("status") or "").strip().lower()
                if status == "archived":
                    continue
                if status == "working":
                    updated = fm.get("updated", "")
                    try:
                        d_upd = datetime.fromisoformat(updated)
                        age_days = (datetime.now() - d_upd).days
                    except (ValueError, TypeError):
                        continue
                    if age_days > args.stream_stale_days:
                        stale_streams.append(
                            f"{d.name} ({age_days}d, updated {updated})")
            elif (d / "SKILL.md").exists():
                try:
                    fm, _ = parse_frontmatter(
                        (d / "SKILL.md").read_text(encoding="utf-8"))
                except (UnicodeDecodeError, OSError):
                    fm = {}
                if fm.get("category") == "stream":
                    issues["stream_no_readme"].append(d.name)

    # Index drift: сверяем каталог записей с INDEX_PATH. Наш формат —
    # markdown-таблица memory/INDEX.md, первая колонка 'records/<name>/';
    # у коллег — список '- **name** — описание' в
    # skills-management/references/index.md. «Записи на диске» у нас —
    # все каталоги базы (знания с SKILL.md + стримы с README.md).
    index_missing, index_phantom = [], []
    if INDEX_PATH.exists():
        index_skills = set()
        for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*\|\s*`?records/([A-Za-z0-9_.-]+)/`?\s*\|", line)
            if m:
                index_skills.add(m.group(1))
                continue
            m = re.match(r"^\s*-\s+\*\*([A-Za-z0-9_-]+)\*\*\s+—", line)
            if m:
                index_skills.add(m.group(1))
        if os.environ.get("SKILLS_ROOT"):
            disk_names = {d.name for d in SKILLS_PATH.iterdir()
                          if d.is_dir() and not d.name.startswith((".", "_"))
                          and d.name not in SKIP_DIRS}
        else:
            disk_names = skill_names
        index_missing = sorted(disk_names - index_skills)
        index_phantom = sorted(index_skills - disk_names)

    # Uncommitted changes (фаза 4 аудита р3): чек-лист skills-management
    # требует коммитить сразу после изменения скилов. Warning, не error:
    # во время активной сессии дерево грязное по определению — сигнал
    # «не забудь закоммитить», а не «база сломана». При SKILLS_ROOT база
    # лежит в отдельном репозитории — git-корень ищем вверх от неё
    # (records → репозиторий). Репозиторий без .git
    # (контентная копия) — проверка молча пропускается.
    def _git_root(path: Path):
        p = Path(path)
        for _ in range(6):
            if (p / ".git").exists():
                return p
            if p.parent == p:
                return None
            p = p.parent
        return None

    repo_root = _git_root(SKILLS_PATH)
    if repo_root is not None:
        try:
            st = subprocess.run(["git", "-C", str(repo_root), "status", "--short"],
                                capture_output=True, text=True, timeout=15)
            dirty = [l.strip() for l in (st.stdout or "").splitlines() if l.strip()]
        except (OSError, subprocess.SubprocessError):
            dirty = []
        if dirty:
            sample = ", ".join(d.strip() for d in dirty[:3])
            more = f" …(+{len(dirty) - 3})" if len(dirty) > 3 else ""
            issues["uncommitted"].append(
                f"{len(dirty)} файлов: {sample}{more} — чек-лист: "
                f"commit сразу после изменения записей памяти")

    # Print report
    print(f"## Lint report [{date.today().isoformat()}]\n")

    total = 0
    if issues["no_related"]:
        print(f"### No related field ({len(issues['no_related'])})")
        for s in issues["no_related"]:
            print(f"  - {s}")
        total += len(issues["no_related"])
        print()

    if issues["no_category"]:
        print(f"### No category field ({len(issues['no_category'])})")
        for s in issues["no_category"]:
            print(f"  - {s}")
        total += len(issues["no_category"])
        print()

    if issues["uncommitted"]:
        print(f"### Uncommitted changes ({len(issues['uncommitted'])})")
        for s in issues["uncommitted"]:
            print(f"  - {s}")
        total += len(issues["uncommitted"])
        print()

    if issues["broken_refs"]:
        print(f"### Broken refs ({len(issues['broken_refs'])})")
        for s in issues["broken_refs"]:
            print(f"  - {s}")
        total += len(issues["broken_refs"])
        print()

        if args.fix:
            print("  → Fixing broken refs...")
            _fix_broken_refs(skills, skill_names)

    if issues["reciprocal_refs"]:
        n = len(issues["reciprocal_refs"])
        print(f"### Non-reciprocal refs ({n}) — informational, не входит в Total")
        print("  Односторонние related-ссылки. Асимметрия — не ошибка "
              "целостности; массовый бэкфиллом не чинить (хабы раздуются),")
        print("  добавлять обратную ссылку точечно при работе со скилом. "
              f"Полный список: lint --reciprocal-full")
        shown = issues["reciprocal_refs"] if getattr(args, "reciprocal_full", False) \
            else issues["reciprocal_refs"][:15]
        for s in shown:
            print(f"  - {s}")
        if not getattr(args, "reciprocal_full", False) and n > len(shown):
            print(f"  … и ещё {n - len(shown)}")
        print()

    if issues["stale"]:
        print(f"### Stale ({len(issues['stale'])})")
        for s in issues["stale"]:
            print(f"  - {s}")
        total += len(issues["stale"])
        print()

    if issues["missing_entities"]:
        print(f"### Missing entity-skills ({len(issues['missing_entities'])})")
        for s in issues["missing_entities"]:
            print(f"  - {s}")
        total += len(issues["missing_entities"])
        print()

    if issues["orphans"]:
        print(f"### Orphans ({len(issues['orphans'])})")
        for s in issues["orphans"]:
            print(f"  - {s} (нет related, никто не ссылается)")
        total += len(issues["orphans"])
        print()

    if issues["empty_description"]:
        print(f"### Empty or short description ({len(issues['empty_description'])})")
        for s in issues["empty_description"]:
            print(f"  - {s}")
        total += len(issues["empty_description"])
        print()

    if issues["long_skills"]:
        print(f"### Skills exceeding line limit ({len(issues['long_skills'])})")
        for s in issues["long_skills"]:
            print(f"  - {s}")
        total += len(issues["long_skills"])
        print()

    if index_missing or index_phantom:
        print("### Index drift")
        for s in index_missing:
            print(f"  - missing in index.md: {s}")
            total += 1
        for s in index_phantom:
            print(f"  - in index.md but not on disk: {s}")
            total += 1
        print()

    if stale_streams:
        print(f"### Stale streams ({len(stale_streams)})")
        for s in stale_streams:
            print(f"  - {s}")
        total += len(stale_streams)
        print()

    if issues["stream_no_readme"]:
        print(f"### Streams without README ({len(issues['stream_no_readme'])})")
        for s in issues["stream_no_readme"]:
            print(f"  - {s}")
        total += len(issues["stream_no_readme"])
        print()

    # Секция docs: сверка доков с кодом (check-docs) — выводится при находках
    doc_issues = run_check_docs()
    if doc_issues:
        print(f"### Docs out of sync ({len(doc_issues)})")
        for s in doc_issues:
            print(f"  - {s}")
            total += 1
        print()

    if total == 0:
        print("All checks passed! ✓")
    else:
        print(f"Total issues: {total}")


def _fix_broken_refs(skills: list[dict], valid_names: set[str]):
    """Remove broken refs from frontmatter.

    Правка контентная (меняется related) — поле `updated` поднимается.
    """
    today = date.today().isoformat()
    for s in skills:
        related = s["frontmatter"].get("related", [])
        if isinstance(related, str):
            related = [related]
        related = [r for r in related if r]
        broken = [r for r in related if r not in valid_names]
        if broken:
            new_related = [r for r in related if r in valid_names]
            text = Path(s["full_path"]).read_text(encoding="utf-8")
            new_text = update_frontmatter(text, {"related": new_related, "updated": today})
            Path(s["full_path"]).write_text(new_text, encoding="utf-8")
            print(f"    ✓ {s['name']}: removed {broken}")


# --- Commands: dedup ---------------------------------------------------------

def _owner_dir(file_path: str):
    """Первый компонент rel-пути чанка — для SQL-фильтра «тот же владелец».

    Аналог skills-search._file_root_dir: 'skills/refs/x.md' -> 'skills',
    'streams/y/z.md' -> 'streams', корневой файл -> None (фильтр не нужен).
    Намеренно шире канонического skill_dir_of (стримы всех тем — один
    'streams'): дубли «стрим ↔ стрим» интереса не представляют, журналы.
    """
    parts = file_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else None


def cmd_dedup(args):
    """Find semantic duplicates using search index."""
    if not SEARCH_DB_PATH.exists():
        print("Search index not found. Run 'skills-search index' first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(SEARCH_DB_PATH))
    try:
        _init_vec_conn(conn, strict=True)
    except ImportError:
        print("sqlite-vec not installed. Run: pip install sqlite-vec", file=sys.stderr)
        sys.exit(1)

    # Все эмбеддинги — одним JOIN, а не SELECT на каждый чанк (N+1 из ~4.5 тыс.
    # запросов; находка аудита р4). Владелец чанка — skill_dir_of(file_path):
    # раньше «другой скил» определялся по колонке skill_name, а у файлов
    # references/ там stem файла, коллизирующий между скилами.
    rows = conn.execute(
        "SELECT c.file_path, c.heading, v.embedding "
        "FROM chunks c JOIN chunks_vec v ON v.id = c.id"
    ).fetchall()

    print(f"Checking {len(rows)} chunks for duplicates...\n")
    duplicates = []
    seen_pairs: set = set()  # O(1)-проверка «пара уже найдена» вместо O(N²)-скана

    # For each chunk, find closest from different skills
    for file_path, heading, embedding in rows:
        owner_dir = _owner_dir(file_path)
        if owner_dir is None:
            # корневой файл (owners нет — README в корне): исключить только
            # сам файл; чанки одного файла сравнивать нечего
            neighbors = conn.execute(
                """SELECT cv.distance, c.file_path, c.heading
                   FROM chunks_vec cv
                   JOIN chunks c ON c.id = cv.id
                   WHERE cv.embedding MATCH ? AND k = ?
                   AND c.file_path != ?
                   ORDER BY cv.distance""",
                (embedding, DEDUP_K, file_path),
            ).fetchall()
        else:
            neighbors = conn.execute(
                """SELECT cv.distance, c.file_path, c.heading
                   FROM chunks_vec cv
                   JOIN chunks c ON c.id = cv.id
                   WHERE cv.embedding MATCH ? AND k = ?
                   AND c.file_path NOT GLOB ?
                   ORDER BY cv.distance""",
                (embedding, DEDUP_K, f"{owner_dir}/*"),
            ).fetchall()

        owner = skill_dir_of(file_path)
        for dist, nfile_path, nheading in neighbors:
            similarity = 1.0 - (dist / 2.0)
            if similarity > DEDUP_SIM_THRESHOLD:
                nowner = skill_dir_of(nfile_path)
                pair = tuple(sorted([f"{owner}/{heading}", f"{nowner}/{nheading}"]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    duplicates.append({
                        "a": f"{owner}/{heading}",
                        "b": f"{nowner}/{nheading}",
                        "similarity": round(similarity, 4),
                    })

    conn.close()

    if duplicates:
        print(f"Found {len(duplicates)} potential duplicates:\n")
        for d in sorted(duplicates, key=lambda x: -x["similarity"]):
            print(f"  {d['similarity']:.4f}  {d['a']}")
            print(f"           ↔ {d['b']}")
            print()
    else:
        print(f"No semantic duplicates found (threshold: {DEDUP_SIM_THRESHOLD}).")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="skills-maintain",
        description="Automated memory maintenance for skills knowledge base",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # lint
    lint_p = sub.add_parser("lint", help="Check knowledge base integrity")
    lint_p.add_argument("--fix", action="store_true", help="Auto-fix deterministic issues")
    lint_p.add_argument("--stale-days", type=int, default=STALE_DAYS,
                        help=f"Stale threshold in days (default {STALE_DAYS})")
    lint_p.add_argument("--stream-stale-days", type=int, default=STREAM_STALE_DAYS,
                        help=f"Stale stream threshold in days (default {STREAM_STALE_DAYS})")
    lint_p.add_argument("--reciprocal-full", action="store_true",
                        help="Print the full non-reciprocal refs list "
                             "(default: top-15, informational)")

    # dedup
    sub.add_parser("dedup", help="Find semantic duplicates")

    # check-docs
    sub.add_parser("check-docs", help="Сверка доков с кодом (без сети/LLM)")

    args = parser.parse_args()

    if args.command == "lint":
        cmd_lint(args)
    elif args.command == "dedup":
        cmd_dedup(args)
    elif args.command == "check-docs":
        cmd_check_docs(args)


if __name__ == "__main__":
    main()
