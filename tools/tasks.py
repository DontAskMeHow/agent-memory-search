#!/usr/bin/env python3
"""tasks.py — мини-CLI слоя текущих задач (каталог tasks/ базы знаний).

Задача = markdown-файл NNN-slug.md с frontmatter (плоские ключи,
списки в [a, b]). Скрипт выполняет детерминированные операции;
интеллектуальную работу (обсуждение, ревью, пересечения) делает агент.

Команды:
    next-id                  следующий свободный id и префикс файла
    list [фильтры]           список задач (markdown-таблица)
    find "запрос"            семантический поиск задач (keyword, плоскость tasks)
    dashboard [--write]      сводка-агрегат; --write -> tasks/index.md
    done T-NNN [--note ...]  закрыть задачу и перенести в archive/
    scan                     проблемы: дедлайны, зависший inbox, stale
    check                    валидация файлов (id, frontmatter, enum, related)
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from skills_lib import (
    SEARCH_DB_PATH,
    SKILLS_PATH,
    TASKS_PATH,
    load_script as _load_script,
    parse_frontmatter,
    set_fm_field,
    utf8_stdout,
)

TASKS_DIR = TASKS_PATH
ARCHIVE_DIR = TASKS_DIR / "archive"
INDEX_PATH = TASKS_DIR / "index.md"
REVIEWS_PATH = TASKS_DIR / "reviews.md"

ACTIVE_STATUSES = ("inbox", "open", "waiting", "someday")
PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3}

utf8_stdout()


# --- Пороги scan (именованные — маркеры доков сверяет check-docs) ---------------

SCAN_DEADLINE_SOON_DAYS = 7   # scan: дедлайн ≤ N дней — «скоро дедлайн»
SCAN_INBOX_IDLE_DAYS = 7      # scan: inbox без уточнения > N дней
SCAN_OPEN_STALE_DAYS = 21     # scan: open без апдейта > N дней

# --- загрузка ----------------------------------------------------------------

def load_tasks(include_archive: bool = False) -> list[dict]:
    """Все задачи из tasks/ (и archive/ при include_archive)."""
    tasks = []
    paths = [p for p in sorted(TASKS_DIR.glob("*.md")) if p.name != "index.md"]
    if include_archive and ARCHIVE_DIR.exists():
        paths += sorted(ARCHIVE_DIR.glob("*.md"))
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            print(f"warning: не читается {p}", file=sys.stderr)
            continue
        fm, _body = parse_frontmatter(text)
        if not fm.get("id"):
            continue
        fm["_path"] = str(p)
        fm["_file"] = p.name
        fm["_archived"] = p.parent == ARCHIVE_DIR
        tasks.append(fm)
    return tasks


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def esc(s) -> str:
    return str(s).replace("|", "\\|")


def today() -> date:
    return date.today()


# --- next-id -----------------------------------------------------------------

def cmd_next_id(_args):
    nums = []
    for d in (TASKS_DIR, ARCHIVE_DIR):
        if d.exists():
            for p in d.glob("*.md"):
                m = re.match(r"^(\d{3})-", p.name)
                if m:
                    nums.append(int(m.group(1)))
    n = max(nums) + 1 if nums else 1
    print(f"T-{n:03d}")
    print(f"файл: {n:03d}-<slug>.md")


# --- list --------------------------------------------------------------------

def task_matches(t: dict, args) -> bool:
    if args.sphere and t.get("sphere") != args.sphere:
        return False
    if args.status and t.get("status") != args.status:
        return False
    if args.priority and t.get("priority") != args.priority:
        return False
    if args.tag:
        tags = t.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if args.tag not in tags:
            return False
    if args.id and t.get("id") != args.id:
        return False
    return True


def sort_tasks(tasks: list[dict], mode: str) -> list[dict]:
    def key(t):
        pr = PRIORITY_ORDER.get(t.get("priority"), 9)
        dl = parse_date(t.get("deadline")) or date(9999, 12, 31)
        up = parse_date(t.get("updated")) or date(1970, 1, 1)
        st = ACTIVE_STATUSES.index(t.get("status")) if t.get("status") in ACTIVE_STATUSES else 9
        if mode == "deadline":
            return (dl, pr)
        if mode == "updated":
            return (-up.toordinal(), pr)
        return (pr, dl, st)
    return sorted(tasks, key=key)


def cmd_list(args):
    tasks = [t for t in load_tasks(include_archive=args.archive) if task_matches(t, args)]
    tasks = sort_tasks(tasks, args.sort)
    if not tasks:
        print("Задач не найдено.")
        return
    print("| id | title | sphere | type | status | P | elaboration | deadline | updated |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in tasks:
        print(f"| {t.get('id', '')} | {esc(t.get('title', ''))} | {t.get('sphere', '')} | "
              f"{t.get('type', '')} | {t.get('status', '')} | {t.get('priority', '')} | "
              f"{t.get('elaboration', '')} | {t.get('deadline', '')} | {t.get('updated', '')} |")
    print(f"\nВсего: {len(tasks)}")


# --- find (фаза 2 аудита р3) ---------------------------------------------------

_SEARCH_MOD = None


def _search_module():
    """skills-search.py как модуль (дефис в имени — общий хелпер skills_lib
    load_script; лениво, с кэшем)."""
    global _SEARCH_MOD
    if _SEARCH_MOD is None:
        _SEARCH_MOD = _load_script(
            Path(__file__).resolve().parent / "skills-search.py",
            "_skills_search_for_find")
    return _SEARCH_MOD


def cmd_find(args):
    """Семантический поиск задач: плоскость tasks общего поискового индекса.

    Дефолт keyword (BM25): реранкер нестабилен на коротких текстах задач
    (см. tools/SKILL.md), а стемминг покрывает морфологию. Кросс-поиск
    «задачи + знания» — skills-search search "X" --source all.
    """
    if not Path(SEARCH_DB_PATH).exists():
        print("Поисковый индекс не построен: сначала "
              "skills-search.py index (или reindex).")
        sys.exit(1)
    mod = _search_module()
    ns = argparse.Namespace(db=str(SEARCH_DB_PATH), query=args.query,
                            top_k=args.top_k, mode=args.mode, source="tasks")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        mod.cmd_search(ns)
    try:
        rows = json.loads(out.getvalue())
    except json.JSONDecodeError:
        print("Поиск вернул не-JSON (индекс повреждён?): skills-search.py reindex")
        sys.exit(1)

    # файлы в порядке релевантности, дедуп; frontmatter — для статусной строки
    by_file: dict = {}
    for r in rows:
        by_file.setdefault(r["file"], r)
    tasks_by_file = {t["_file"]: t for t in load_tasks(include_archive=True)}
    hits = [(f, r) for f, r in by_file.items() if f.split("/")[-1] in tasks_by_file]
    if not hits:
        print("Ничего не найдено (плоскость tasks; полный поиск — "
              "skills-search search \"...\" --source all).")
        return
    print("| id | title | status | P | deadline | updated |")
    print("|---|---|---|---|---|---|")
    for f, _r in hits:
        t = tasks_by_file[f.split("/")[-1]]
        print(f"| {t.get('id', '')} | {esc(t.get('title', ''))} | "
              f"{t.get('status', '')}{' (архив)' if t['_archived'] else ''} | "
              f"{t.get('priority', '')} | {t.get('deadline', '')} | "
              f"{t.get('updated', '')} |")
    print(f"\nНайдено задач: {len(hits)}")


# --- review (фаза 5 аудита р3) --------------------------------------------------

def cmd_review(args):
    """Подготовка к ревью: stale-кандидаты с возрастом и контекстом.

    Доказательный разбор (решена / не актуальна / жива) делает агент по
    шаблону из tasks/SKILL.md — session-search по T-id/заголовку, git log,
    skills-search; закрытие только с подтверждения пользователя.
    """
    if not getattr(args, "prepare", False):
        print("Использование: tasks.py review --prepare", file=sys.stderr)
        sys.exit(2)
    tasks = [t for t in load_tasks(include_archive=True)
             if t.get("status") in ACTIVE_STATUSES]
    t7 = today() + timedelta(days=SCAN_DEADLINE_SOON_DAYS)
    rows = []
    for t in tasks:
        upd = parse_date(t.get("updated"))
        age = (today() - upd).days if upd else None
        dl = parse_date(t.get("deadline"))
        stale_open = (t.get("status") == "open" and age is not None
                      and age > SCAN_OPEN_STALE_DAYS)
        stale_inbox = (t.get("status") == "inbox" and age is not None
                       and age > SCAN_INBOX_IDLE_DAYS)
        overdue = dl is not None and dl < today()
        if stale_open or stale_inbox or overdue:
            rows.append((t, age, overdue))
    rows.sort(key=lambda x: -(x[1] or 0))
    if not rows:
        print("Кандидатов на ревью нет (нет stale/просроченных).")
        return
    print("| id | title | status | P | возраст | дедлайн | origin | related |")
    print("|---|---|---|---|---|---|---|---|")
    for t, age, overdue in rows:
        dl = t.get("deadline", "") + (" ⚠ просрочен" if overdue else "")
        rel = ", ".join(r for r in (t.get("related") or [])[:4])
        print(f"| {t['id']} | {esc(t.get('title', ''))} | {t.get('status', '')} | "
              f"{t.get('priority', '')} | {age if age is not None else '?'} дн. | "
              f"{dl} | {t.get('origin', '?')} | {esc(rel)} |")
    print(f"\nКандидатов: {len(rows)}. Дальше — саб-агент «Ревью задач с "
          f"доказательствами» (шаблон в tasks/SKILL.md): session-search по "
          f"заголовку/T-id, git log, skills-search → вердикт решена / "
          f"не актуальна / жива. Закрытие — только с подтверждения "
          f"пользователя (done --resolution solved|obsolete).")


# --- ics (фаза 5 аудита р3) -----------------------------------------------------

def _ics_escape(s: str) -> str:
    return (str(s).replace("\\", "\\\\").replace(";", "\\;")
            .replace(",", "\\,").replace("\n", "\\n"))


def cmd_ics(args):
    """iCal (.ics) из задач с deadline: VEVENT all-day + VALARM за сутки.

    Для трей-календаря на рабочем столе (Rainlendar Lite читает локальный
    .ics; встроенный календарь Windows 11 без облачного аккаунта не
    работает — см. tasks/SKILL.md «Календарь на рабочем столе»)."""
    sphere = getattr(args, "sphere", None) or "personal"
    tasks = [t for t in load_tasks(include_archive=False)
             if t.get("status") in ACTIVE_STATUSES and t.get("deadline")
             and (sphere == "all" or t.get("sphere") == sphere)]
    if not tasks:
        print(f"Задач с deadline (sphere={sphere}) не найдено.")
        return
    now = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//skills-tasks//RU",
        "CALSCALE:GREGORIAN",
    ]
    for t in sorted(tasks, key=lambda x: str(x.get("deadline"))):
        d = parse_date(t.get("deadline"))
        if not d:
            continue
        summary = _ics_escape(f"{t.get('id')} {t.get('title', '')}")
        descr = _ics_escape(
            f"Задача из tasks каталога базы знаний; статус: {t.get('status', '')}; "
            f"P: {t.get('priority', '')}")
        alarm = _ics_escape(f"Дедлайн задачи {t.get('id')} завтра")
        # all-day VEVENT: дата без времени; VALARM за сутки до дедлайна
        lines += [
            "BEGIN:VEVENT",
            f"UID:{t.get('id')}@skills-tasks",
            f"DTSTAMP:{now}",
            f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime('%Y%m%d')}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{descr}",
            "BEGIN:VALARM",
            "TRIGGER:-P1D",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{alarm}",
            "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    out = "\r\n".join(lines) + "\r\n"
    out_path = Path(getattr(args, "out", None) or (TASKS_DIR / "deadlines.ics"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8", newline="")
    print(f"Календарь: {out_path} ({len(tasks)} задач с deadline, sphere={sphere})")
    print("Rainlendar Lite: положи файл в папку данных (или подпишись в "
          "трее) — см. tasks/SKILL.md «Календарь на рабочем столе».")


# --- dashboard ---------------------------------------------------------------

def last_review_date():
    if not REVIEWS_PATH.exists():
        return None
    dates = re.findall(r"(?m)^## \[(\d{4}-\d{2}-\d{2})\]", REVIEWS_PATH.read_text(encoding="utf-8"))
    return max(dates) if dates else None


def cmd_dashboard(args):
    all_tasks = load_tasks(include_archive=True)
    archived = [t for t in all_tasks if t["_archived"]]
    active = [t for t in all_tasks if not t["_archived"] and t.get("status") != "done"]
    lines = []
    lines.append("# Задачи — дашборд")
    lines.append("")
    lines.append("> Генерируется `tools/tasks.py dashboard --write`. Руками не править.")
    lines.append("")
    lr = last_review_date()
    lines.append(f"_Сгенерировано: {today().isoformat()}. Последний ревью: {lr or '—'}_")
    lines.append("")

    by_status, by_sphere = {}, {}
    for t in active:
        by_status[t.get("status", "?")] = by_status.get(t.get("status", "?"), 0) + 1
        by_sphere[t.get("sphere", "?")] = by_sphere.get(t.get("sphere", "?"), 0) + 1
    st = ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())) or "—"
    sp = ", ".join(f"{k}: {v}" for k, v in sorted(by_sphere.items())) or "—"
    lines.append("## Сводка")
    lines.append("")
    lines.append(f"- Активных: **{len(active)}** (статусы — {st}; сферы — {sp})")
    lines.append(f"- В архиве: {len(archived)}")
    lines.append("")

    t7 = today() + timedelta(days=SCAN_DEADLINE_SOON_DAYS)
    overdue = [t for t in active if (d := parse_date(t.get("deadline"))) and d < today()]
    soon = [t for t in active if (d := parse_date(t.get("deadline"))) and today() <= d <= t7]
    lines.append("## Срочное")
    lines.append("")
    if overdue or soon:
        for label, group in (("Просрочено", overdue), ("Дедлайн ≤ 7 дней", soon)):
            if not group:
                continue
            lines.append(f"**{label}:**")
            lines.append("")
            lines.append("| id | title | deadline | P |")
            lines.append("|---|---|---|---|")
            for t in sorted(group, key=lambda x: parse_date(x.get("deadline"))):
                lines.append(f"| {t['id']} | {esc(t.get('title', ''))} | "
                             f"{t.get('deadline')} | {t.get('priority', '')} |")
            lines.append("")
    else:
        lines.append("Ничего срочного.")
        lines.append("")

    inbox = [t for t in active if t.get("status") == "inbox"]
    if inbox:
        lines.append("## Inbox (не разобрано)")
        lines.append("")
        for t in sort_tasks(inbox, "priority"):
            lines.append(f"- **{t['id']}** {t.get('title', '')} ({t.get('updated', '')})")
        lines.append("")

    for sphere in sorted({t.get("sphere", "?") for t in active}):
        group = [t for t in active if t.get("sphere", "?") == sphere and t.get("status") != "inbox"]
        if not group:
            continue
        lines.append(f"## {sphere}")
        lines.append("")
        lines.append("| id | title | status | P | elaboration | deadline | updated |")
        lines.append("|---|---|---|---|---|---|---|")
        for t in sort_tasks(group, "priority"):
            lines.append(f"| {t['id']} | {esc(t.get('title', ''))} | {t.get('status', '')} | "
                         f"{t.get('priority', '')} | {t.get('elaboration', '')} | "
                         f"{t.get('deadline', '')} | {t.get('updated', '')} |")
        lines.append("")

    # Происхождение (фаза 5 аудита р3): user — по явному запросу/согласию,
    # agent — синтезирована агентом; resolution — чем закончились закрытые
    by_origin = {"user": 0, "agent": 0}
    for t in active:
        by_origin[t.get("origin", "?")] = by_origin.get(t.get("origin", "?"), 0) + 1
    by_res = {}
    for t in all_tasks:
        if t.get("status") == "done":
            by_res[t.get("resolution", "?")] = by_res.get(t.get("resolution", "?"), 0) + 1
    lines.append("## Происхождение и финалы")
    lines.append("")
    lines.append(f"- Активные по происхождению: user — {by_origin.get('user', 0)}, "
                 f"agent — {by_origin.get('agent', 0)}"
                 + (f", без поля — {by_origin.get('?', 0)}" if by_origin.get("?") else ""))
    if by_res:
        lines.append(f"- Закрытые по финалу: "
                     f"{', '.join(f'{k} — {v}' for k, v in sorted(by_res.items()))}")
    lines.append("")

    out = "\n".join(lines)
    if args.write:
        INDEX_PATH.write_text(out + "\n", encoding="utf-8")
        print(f"Дашборд записан: {INDEX_PATH}")
    else:
        print(out)


# --- done --------------------------------------------------------------------

def cmd_done(args):
    for t in load_tasks():
        if t.get("id") == args.id:
            p = Path(t["_path"])
            text = p.read_text(encoding="utf-8")
            text = set_fm_field(text, "status", "done")
            # resolution (фаза 5 аудита р3): чем закончилась — решена или
            # перестала быть актуальной; видно в архиве, дашборде и reviews
            resolution = getattr(args, "resolution", None) or "solved"
            text = set_fm_field(text, "resolution", resolution)
            text = set_fm_field(text, "updated", today().isoformat())
            if args.note:
                text = text.rstrip("\n") + f"\n\n## Результат\n\n{args.note}\n"
            ARCHIVE_DIR.mkdir(exist_ok=True)
            # атомарный перенос: обновлённый текст пишется во временный файл
            # архива, финальное состояние меняется одним os.replace — без
            # промежуточного «done в tasks/» (раньше файл переписывался на
            # месте и только потом переносился). .tmp-хвосты glob("*.md") не
            # видит; исходник удаляется последним шагом.
            tmp = ARCHIVE_DIR / f".{p.name}.tmp"
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, ARCHIVE_DIR / p.name)
            p.unlink()
            print(f"Закрыта {args.id} -> {ARCHIVE_DIR / p.name}")
            return
    print(f"Задача {args.id} не найдена среди активных.", file=sys.stderr)
    sys.exit(1)


# --- scan --------------------------------------------------------------------

def cmd_scan(_args):
    tasks = [t for t in load_tasks() if t.get("status") != "done"]
    problems = []
    t7 = today() + timedelta(days=SCAN_DEADLINE_SOON_DAYS)
    for t in tasks:
        dl = parse_date(t.get("deadline"))
        upd = parse_date(t.get("updated"))
        if dl and dl < today():
            problems.append(("просрочено", t, f"дедлайн {t.get('deadline')}"))
        elif dl and dl <= t7:
            problems.append(("скоро дедлайн", t, f"дедлайн {t.get('deadline')}"))
        if t.get("status") == "inbox" and upd and (today() - upd).days > SCAN_INBOX_IDLE_DAYS:
            problems.append(("inbox завис", t, f"без уточнения {(today() - upd).days} дн."))
        if t.get("status") == "open" and upd and (today() - upd).days > SCAN_OPEN_STALE_DAYS:
            problems.append(("stale", t, f"без апдейта {(today() - upd).days} дн."))
    if not problems:
        print("Проблем не найдено.")
        return
    order = {"просрочено": 0, "скоро дедлайн": 1, "inbox завис": 2, "stale": 3}
    problems.sort(key=lambda x: (order.get(x[0], 9), PRIORITY_ORDER.get(x[1].get("priority"), 9)))
    print("| что | id | title | P | деталь |")
    print("|---|---|---|---|---|")
    for kind, t, detail in problems:
        print(f"| {kind} | {t['id']} | {esc(t.get('title', ''))} | {t.get('priority', '')} | {detail} |")
    print(f"\nВсего проблем: {len(problems)}")


# --- check: валидация файлов задач --------------------------------------------

VALID_STATUSES = ACTIVE_STATUSES + ("done",)
VALID_PRIORITIES = ("P1", "P2", "P3")
VALID_SPHERES = ("personal", "work")
VALID_TYPES = ("one-off", "recurring", "stream")
VALID_ELABORATIONS = ("idea", "sketch", "detailed", "ready")
# Происхождение задачи (фаза 5 аудита р3): user — по явному запросу или
# согласию пользователя; agent — синтезирована агентом (аудит, ревью,
# миграция backlog). resolution — чем закончилась (фаза 5): solved —
# решена; obsolete — перестала быть актуальной.
VALID_ORIGINS = ("user", "agent")
VALID_RESOLUTIONS = ("solved", "obsolete")
REQUIRED_FIELDS = ("id", "title", "sphere", "type", "status", "priority",
                   "elaboration", "created", "updated", "origin")
_DATE_FIELDS = ("created", "updated", "deadline")


def load_task_files() -> list[dict]:
    """Сырые файлы задач `NNN-*.md` — включая «битые» (без id/frontmatter).

    В отличие от load_tasks, который молча выкидывает файлы без id, здесь
    каждый файл возвращается как {_path, _file, _fm, _body} с _fm=None для
    нечитаемых — check должен видеть все проблемы. Архив не входит.
    """
    out = []
    for p in sorted(TASKS_DIR.glob("[0-9][0-9][0-9]-*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            out.append({"_path": str(p), "_file": p.name, "_fm": None, "_body": ""})
            continue
        fm, body = parse_frontmatter(text)
        out.append({"_path": str(p), "_file": p.name, "_fm": fm, "_body": body})
    return out


def _next_step_empty(body: str) -> bool:
    """Секция «## Следующий шаг» отсутствует или пустая (до следующего ##)."""
    m = re.search(r"(?m)^##\s+Следующий шаг\s*$(.*?)(?=^##\s|\Z)", body, re.DOTALL)
    if not m:
        return True
    return not m.group(1).strip()


def cmd_check(_args):
    """Валидация файлов задач (только чтение — ничего не правит).

    Проверки на файл: id-формат T-NNN и совпадение с числом в имени файла;
    дубли id; обязательные frontmatter-поля; enum-ы; related (T-ids архив +
    каталоги скилов); даты YYYY-MM-DD; пустой «Следующий шаг».
    """
    files = load_task_files()
    known_ids = {t.get("id") for t in load_tasks(include_archive=True) if t.get("id")}
    skill_dirs = {d.name for d in SKILLS_PATH.iterdir() if d.is_dir()}

    problems = []
    seen_ids: dict[str, str] = {}

    for rec in files:
        fname = rec["_file"]
        fm = rec["_fm"]

        if fm is None:
            problems.append((fname, "не читается/не парсится"))
            continue

        mnum = re.match(r"^(\d{3})-", fname)
        expected = f"T-{int(mnum.group(1)):03d}" if mnum else None
        if mnum is None:
            problems.append((fname, "имя файла не NNN-slug.md"))

        tid = fm.get("id")
        if not tid:
            problems.append((fname, "нет обязательного поля id"))
        else:
            if not re.fullmatch(r"T-\d{3}", str(tid)):
                problems.append((fname, f"id {tid} не в формате T-NNN"))
            elif mnum is not None and tid != expected:
                problems.append((fname, f"id {tid} != числу в имени файла ({expected})"))
            if str(tid) in seen_ids:
                problems.append((fname, f"дубликат id {tid} (уже в {seen_ids[str(tid)]})"))
            else:
                seen_ids[str(tid)] = fname

        missing = [k for k in REQUIRED_FIELDS if k not in fm]
        if missing:
            problems.append((fname, f"нет обязательных полей: {', '.join(missing)}"))

        enum_map = {
            "status": VALID_STATUSES, "priority": VALID_PRIORITIES,
            "sphere": VALID_SPHERES, "type": VALID_TYPES,
            "elaboration": VALID_ELABORATIONS,
            "origin": VALID_ORIGINS, "resolution": VALID_RESOLUTIONS,
        }
        for field, valid in enum_map.items():
            val = fm.get(field)
            if val is not None and val not in valid:
                problems.append((fname, f"{field}: {val} не из {'|'.join(valid)}"))

        for field in _DATE_FIELDS:
            val = fm.get(field)
            if val is not None and parse_date(val) is None:
                problems.append((fname, f"{field}: {val} не YYYY-MM-DD"))

        related = fm.get("related") or []
        if isinstance(related, str):
            related = [related]
        for r in related:
            if r and r not in known_ids and r not in skill_dirs:
                problems.append((fname, f"related: {r} — нет такой задачи/скила"))

        if _next_step_empty(rec["_body"]):
            problems.append((fname, "пустой «Следующий шаг»"))

    if not problems:
        print("Проблем не найдено.")
        return
    print("| файл | проблема |")
    print("|---|---|")
    for fname, why in problems:
        print(f"| {fname} | {esc(why)} |")
    print(f"\nВсего проблем: {len(problems)}")


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Слой текущих задач (tasks/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("next-id", help="следующий свободный id")

    p_list = sub.add_parser("list", help="список задач")
    p_list.add_argument("--sphere")
    p_list.add_argument("--status")
    p_list.add_argument("--priority")
    p_list.add_argument("--tag")
    p_list.add_argument("--id")
    p_list.add_argument("--sort", choices=["priority", "deadline", "updated"], default="priority")
    p_list.add_argument("--archive", action="store_true", help="включая архив")
    p_list.set_defaults(func=cmd_list)

    p_find = sub.add_parser("find", help="семантический поиск задач (keyword, плоскость tasks)")
    p_find.add_argument("query", help="поисковый запрос (обычной фразой)")
    p_find.add_argument("--top-k", type=int, default=10, help="сколько чанков брать из выдачи")
    p_find.add_argument("--mode", choices=["hybrid", "semantic", "keyword"],
                        default="keyword", help="режим поиска (дефолт keyword — без реранкера)")
    p_find.set_defaults(func=cmd_find)

    p_dash = sub.add_parser("dashboard", help="сводка-агрегат")
    p_dash.add_argument("--write", action="store_true", help="записать в tasks/index.md")
    p_dash.set_defaults(func=cmd_dashboard)

    p_done = sub.add_parser("done", help="закрыть задачу")
    p_done.add_argument("id")
    p_done.add_argument("--note")
    p_done.add_argument("--resolution", choices=list(VALID_RESOLUTIONS), default="solved",
                        help="чем закончилась: solved (решена) | obsolete (не актуальна)")
    p_done.set_defaults(func=cmd_done)

    p_review = sub.add_parser("review", help="подготовка к ревью задач")
    p_review.add_argument("--prepare", action="store_true",
                          help="таблица stale-кандидатов с возрастом и контекстом")
    p_review.set_defaults(func=cmd_review)

    p_ics = sub.add_parser("ics", help="iCal (.ics) из задач с deadline")
    p_ics.add_argument("--sphere", choices=["personal", "work", "all"], default="personal",
                       help="какие сферы включать (дефолт personal — приватность)")
    p_ics.add_argument("--out", default=None, help="путь файла (дефолт tasks/deadlines.ics)")
    p_ics.set_defaults(func=cmd_ics)

    p_scan = sub.add_parser("scan", help="проблемы: дедлайны, inbox, stale")
    p_scan.set_defaults(func=cmd_scan)

    sub.add_parser("check", help="валидация файлов задач (id, frontmatter, related)") \
        .set_defaults(func=cmd_check)

    args = ap.parse_args()
    if args.cmd == "next-id":
        cmd_next_id(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
