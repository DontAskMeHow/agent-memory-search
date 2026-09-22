#!/usr/bin/env python3
"""
skills-search: hybrid BM25 + vector search over markdown knowledge base.

Uses SQLite FTS5 for keyword search, sqlite-vec for vector similarity,
and BGE-M3 via OpenAI-compatible HTTP endpoint for embeddings.
Results fused with Reciprocal Rank Fusion (RRF).

Usage:
    skills-search index   [--path DIR] [--db FILE] [--json]  # index/reindex markdown files
    skills-search search  QUERY [--top-k N] [--mode M]
                          [--scope S] [--category C] [--skill DIR]
                          [--include-archived] [--no-rerank]
                          [--abstain|--no-abstain]
                          [--source {knowledge|skills|streams|tasks|all}]
    skills-search status  [--db FILE] [--json]              # show index stats
    skills-search reindex [--db FILE] [--path DIR]          # full reindex (drop + rebuild)

Metadata-фильтры search строятся из frontmatter SKILL.md на лету
(find_all_skills): scope/category/skill и исключение status: archived
по умолчанию. Поле updated в выдаче — тоже из frontmatter (колонки БД
под это не заводится).
"""

import argparse
import fnmatch
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import httpx

from skills_lib import (
    SEARCH_DB_PATH,
    SKILLS_PATH,
    SKIP_DIRS,
    SOURCE_MODES,
    VALID_CATEGORIES,
    VALID_SCOPES,
    _MODES,
    drop_db_files as _drop_db_files,
    find_all_skills,
    parse_frontmatter,
    skill_dir_of,
    utf8_stdout,
)

# Windows: cp1251-консоль под GIT-пайпом коверкает кириллицу в JSON выдачи
utf8_stdout()
from search_core import (
    EMBEDDING_DIM,
    QUERY_PREFIX,
    RERANKER_TOP_N,
    RRF_K,
    _is_cyrillic,
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
# Общие функции чанкера markdown вынесены в chunker_md.py (фаза проектного
# поиска). Имена импортируются в namespace модуля — тесты и check-docs
# обращаются к ним как к атрибутам skills-search.
from chunker_md import (  # noqa: E402
    CHUNK_SIZE,
    DESC_PREFIX_MAX,
    _LIST_RE,
    _find_fence_ranges,
    _group_blocks,
    _is_inside_fence,
    _parse_atomic_blocks,
    _split_by_headings_fence_aware,
    chunk_markdown,
)

# --- Configuration -----------------------------------------------------------

DEFAULT_SKILLS_PATH = SKILLS_PATH  # база знаний: env SKILLS_ROOT или корень движка
DEFAULT_DB_PATH = SEARCH_DB_PATH

# Directories/patterns to skip during indexing
SKIP_FILES = {".env", "package-lock.json", "yarn.lock"}
# templates/ — скелеты записей с плейсхолдерами, шум в выдаче
SKIP_DIR_NAMES = SKIP_DIRS | {"templates"}
# Ротация журнала (фаза 4 аудита р3): архивы log-archive-*.md — история
# «когда мы делали X», доступная чтением, но не засоряющая выдачу поиска
SKIP_REL_GLOBS = ("skills-management/references/log-archive*.md",)
# Производные агрегаты: дашборд задач дублирует заголовки задач и
# ранжируется выше самих файлов задач
SKIP_REL_PATHS = {"tasks/index.md"}

# File patterns to index
INDEX_EXTENSIONS = {".md"}


# --- Database ----------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with FTS5 and sqlite-vec."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    init_vec_conn(conn, strict=True)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            indexed_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            heading TEXT NOT NULL DEFAULT '',
            part INTEGER NOT NULL DEFAULT 0,
            text TEXT NOT NULL,
            stemmed_text TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL,
            FOREIGN KEY (file_path) REFERENCES files(path) ON DELETE CASCADE
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            stemmed_text, skill_name, heading,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61'
        );

        -- Triggers to keep FTS in sync (index stemmed_text for BM25)
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, stemmed_text, skill_name, heading)
            VALUES (new.id, new.stemmed_text, new.skill_name, new.heading);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, stemmed_text, skill_name, heading)
            VALUES ('delete', old.id, old.stemmed_text, old.skill_name, old.heading);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, stemmed_text, skill_name, heading)
            VALUES ('delete', old.id, old.stemmed_text, old.skill_name, old.heading);
            INSERT INTO chunks_fts(rowid, stemmed_text, skill_name, heading)
            VALUES (new.id, new.stemmed_text, new.skill_name, new.heading);
        END;
    """)

    # Create vector table (must be done separately, can't be in executescript)
    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)
    except sqlite3.OperationalError:
        # недоступен vec0 — init_vec_conn(strict=True) выше уже бы упал
        pass

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    return conn


def file_content_hash(content: str) -> str:
    """SHA-256 hash of file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# --- Metadata-фильтры (scope/category/skill, archived) ------------------------

def _file_root_dir(file_path: str):
    """Первый компонент rel-пути (для GLOB-фильтров 'dir/*' по file_path).

    Каноническое ИМЯ — skill_dir_of; корневой файл → None.
    """
    parts = file_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else None


def build_skill_meta(skills: list) -> dict:
    """find_all_skills() -> {каноническое имя скила: мета из frontmatter}.

    Ключ — каноническое имя (skill_dir_of: tools -> skills-tools), в значении
    сохраняется и 'dir' (каталог для GLOB-фильтров file_path). Читается за
    миллисекунды (54 SKILL.md), поэтому пересобирается на каждый поиск,
    а не хранится в БД (схему индекса не трогаем).
    """
    meta = {}
    for s in skills:
        canon = skill_dir_of(s["file"])
        if canon == "(root)":
            continue
        fm = s["frontmatter"]
        meta[canon] = {
            "name": s["name"],
            "dir": _file_root_dir(s["file"]),
            "scope": fm.get("scope"),
            "category": fm.get("category"),
            "status": fm.get("status", "active"),
            "updated": fm.get("updated"),
        }
    return meta


def filter_skill_dirs(meta: dict, scope=None, category=None, skill=None,
                      include_archived=False) -> set:
    """Канонические имена скилов, прошедших фильтры scope/category/skill.

    --skill принимается и как имя каталога, и как frontmatter name
    (у tools/SKILL.md name='skills-tools', каталог — 'tools').
    Скилы со status: archived выкидываются всегда, кроме include_archived.
    Пустое множество = несовместимые фильтры (нет ни одного скила).
    """
    kept = set()
    for d, m in meta.items():
        if m["status"] == "archived" and not include_archived:
            continue
        if skill is not None and skill not in (m["dir"], m["name"]):
            continue
        if scope is not None and m["scope"] != scope:
            continue
        if category is not None and m["category"] != category:
            continue
        kept.add(d)
    return kept


def _glob_predicate(dirs: set, col: str = "file_path", negate: bool = False):
    """SQL-предикат принадлежности чанка каталогам скилов.

    Фильтруем по file_path (GLOB 'dir/*'), а не по skill_name: у файлов
    references/ в skill_name лежит stem файла ('servers', 'README'),
    который коллизирует между скилами. Возвращает (sql, params);
    пустое dirs -> ('0=1', []) либо ('1=1', []) при negate.
    """
    dirs = sorted(dirs)
    if not dirs:
        return ("1=1" if negate else "0=1"), []
    op = "NOT GLOB" if negate else "GLOB"
    joiner = " AND " if negate else " OR "
    sql = "(" + joiner.join(f"{col} {op} ?" for _ in dirs) + ")"
    return sql, [f"{d}/*" for d in dirs]


# --- Плоскости доступа (фаза 2 аудита р3) --------------------------------------

# Файлы задач — операционный слой (уровень E): NNN-slug.md, archive/,
# reviews.md. tasks/SKILL.md и tasks/references/ — знания (конвенции),
# остаются в плоскости knowledge.
_TASK_FILE_GLOBS = ("tasks/[0-9]*.md", "tasks/archive/*", "tasks/reviews.md")


def _source_predicate(source: str, col: str = "file_path"):
    """SQL-предикат плоскости доступа по file_path.

    knowledge (дефолт) — всё, кроме файлов задач (задачи не должны
    вытеснять знания из выдачи); skills — каталоги скилов (без streams/
    и файлов задач); streams — только стримы; tasks — файлы задач
    (операционный слой); all — без фильтра (поведение до фазы 2).
    """
    if source == "all":
        return ("1=1", [])
    if source == "streams":
        return (f"({col} GLOB ?)", ["streams/*"])
    if source == "tasks":
        ors = " OR ".join(f"{col} GLOB ?" for _ in _TASK_FILE_GLOBS)
        return (f"({ors})", list(_TASK_FILE_GLOBS))
    excl = (["streams/*"] if source == "skills" else []) + list(_TASK_FILE_GLOBS)
    ands = " AND ".join(f"{col} NOT GLOB ?" for _ in excl)
    return (f"({ands})", excl)


def _display_skill(file_path: str, skill_name: str) -> str:
    """Поле skill в выдаче: SKILL.md -> каноническое имя; остальные файлы ->
    <относительный-каталог>/<stem> (фаза 2 аудита р3: голой stem у файлов
    references/ коллизирует между скилами — optimization-report-2026-09-06
    в entity-deepseek и entity-glm, README ×6, links ×4)."""
    p = file_path.replace("\\", "/")
    if p.endswith("/SKILL.md") or p == "SKILL.md":
        return skill_name
    head, _, tail = p.rpartition("/")
    stem = tail[:-3] if tail.endswith(".md") else tail
    if tail == "README.md" and head == skill_name:
        return skill_name  # README стрима — каноническое имя каталога
    return f"{head}/{stem}" if head else stem


# --- Commands ----------------------------------------------------------------

def cmd_index(args):
    """Index markdown files (incremental by content hash)."""
    skills_path = Path(args.path).resolve()
    db_path = Path(args.db).resolve()
    as_json = getattr(args, "json", False)

    if not skills_path.is_dir():
        print(f"Error: {skills_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    conn = init_db(db_path)

    md_files = _collect_md_files(skills_path)

    # Check which files need (re)indexing
    existing = dict(conn.execute("SELECT path, content_hash FROM files").fetchall())
    to_index = _plan_indexing(md_files, existing)
    current_paths = {rel for _, rel in md_files}

    # Remove files that no longer exist
    removed = set(existing.keys()) - current_paths

    stats = {"indexed_files": 0, "chunks_indexed": 0, "removed_files": len(removed),
             "total_files": len(current_paths), "up_to_date": False}

    if not to_index:
        stats["up_to_date"] = True
        if removed:
            for path in removed:
                _remove_file_from_index(conn, path)
            conn.commit()
            if not as_json:
                print(f"Removed {len(removed)} deleted files from index")
        if as_json:
            print(json.dumps(stats, ensure_ascii=False))
        else:
            print(f"Index up to date ({len(current_paths)} files)")
        conn.close()
        return

    if not as_json:
        print(f"Indexing {len(to_index)} files (of {len(md_files)} total)...")

    # Parse and chunk files (без DML — БД получит изменения только после
    # успешного embedding, единой транзакцией ниже)
    all_chunks = _chunks_of(to_index)

    stats["indexed_files"] = len(to_index)

    if not all_chunks:
        for path in removed:
            _remove_file_from_index(conn, path)
        conn.commit()
        if as_json:
            print(json.dumps(stats, ensure_ascii=False))
        else:
            print("No chunks to index")
        conn.close()
        return

    # Get embeddings in batch
    if not as_json:
        print(f"Embedding {len(all_chunks)} chunks...")
    t0 = time.time()
    texts_for_embedding = [c["text_for_embedding"] for c in all_chunks]
    try:
        embeddings = embed_texts(texts_for_embedding)
    except BaseException:
        conn.close()  # откат + отпустить файл БД (Windows lock, tmp-reindex)
        raise
    t_embed = time.time() - t0
    if not as_json:
        print(f"Embedded in {t_embed:.1f}s ({len(all_chunks)/t_embed:.0f} chunks/s)")

    # Guard целостности: векторов обязано быть ровно столько же, сколько
    # чанков — иначе zip ниже молча обрежет вставку. Мисматч = сбой модели,
    # неатомарная вставка запрещена, летим с понятной ошибкой.
    if len(embeddings) != len(all_chunks):
        conn.close()
        raise RuntimeError(
            f"embedding count mismatch: {len(embeddings)} vectors "
            f"for {len(all_chunks)} chunks")

    stats["chunks_indexed"] = len(all_chunks)
    stats["embed_seconds"] = round(t_embed, 1)

    _store_chunks(conn, removed, to_index, all_chunks, embeddings)

    if as_json:
        print(json.dumps(stats, ensure_ascii=False))
    else:
        print(f"Done: indexed {len(to_index)} files, {len(all_chunks)} chunks. "
              f"Total: {len(current_paths)} files in index.")


# --- Helpers cmd_index --------------------------------------------------------

def _collect_md_files(skills_path: Path) -> list[tuple[Path, str]]:
    """Markdown-файлы базы знаний: os.walk с SKIP_DIR_NAMES (вкл. templates/),
    SKIP_FILES и SKIP_REL_PATHS. Возвращает (full_path, rel_path)."""
    md_files = []
    for root, dirs, files in os.walk(skills_path):
        # Skip excluded directories (incl. templates/)
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for f in files:
            if Path(f).suffix in INDEX_EXTENSIONS and f not in SKIP_FILES:
                full_path = Path(root) / f
                rel_path = str(full_path.relative_to(skills_path)).replace("\\", "/")
                if rel_path in SKIP_REL_PATHS:
                    continue
                if any(fnmatch.fnmatch(rel_path, g) for g in SKIP_REL_GLOBS):
                    continue
                md_files.append((full_path, rel_path))
    return md_files


def _plan_indexing(md_files: list, existing: dict) -> list:
    """Файлы к (пере)индексации: content hash отсутствует в индексе или
    отличается (инкрементальность по hash). Возвращает
    [(full_path, rel_path, content, hash)]."""
    to_index = []
    for full_path, rel_path in md_files:
        try:
            content = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        h = file_content_hash(content)
        if rel_path not in existing or existing[rel_path] != h:
            to_index.append((full_path, rel_path, content, h))
    return to_index


def _chunks_of(to_index: list) -> list[dict]:
    """Чанки файлов к индексации (парсинг без DML; file_hash проставлен)."""
    all_chunks = []
    for full_path, rel_path, content, h in to_index:
        chunks = chunk_markdown(content, rel_path)
        for chunk in chunks:
            chunk["file_hash"] = h
        all_chunks.extend(chunks)
    return all_chunks


def _store_chunks(conn: sqlite3.Connection, removed: set, to_index: list,
                  all_chunks: list, embeddings: list) -> None:
    """Единая транзакция на ВСЕ DML (удаления старых данных + вставка новых).

    Любой сбой mid-insert откатывается целиком — файл не остаётся
    «наполовину», удалённые файлы не пропадают при неудаче embedding.
    conn закрывается в finally (и на исключении — отпуск файла БД под
    Windows: временный файл атомарного reindex).
    """
    try:
        with conn:
            for path in removed:
                _remove_file_from_index(conn, path)
            for _, rel_path, _, _ in to_index:
                _remove_file_from_index(conn, rel_path)

            # Insert file records first (FK constraint)
            for _, rel_path, _, h in to_index:
                conn.execute(
                    "INSERT OR REPLACE INTO files (path, content_hash, indexed_at) VALUES (?, ?, ?)",
                    (rel_path, h, time.time()),
                )

            # Insert chunks + vectors
            for chunk, embedding in zip(all_chunks, embeddings):
                stemmed = stem_text(chunk["text"])
                cursor = conn.execute(
                    "INSERT INTO chunks (file_path, skill_name, heading, part, text, stemmed_text, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk["file"], chunk["skill"], chunk["heading"],
                     chunk["part"], chunk["text"], stemmed, chunk["file_hash"]),
                )
                chunk_id = cursor.lastrowid
                conn.execute(
                    "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_f32(embedding)),
                )
    finally:
        conn.close()


def _remove_file_from_index(conn: sqlite3.Connection, rel_path: str):
    """Remove a file and its chunks from the index.

    Подзапрос по file_path вместо плейсхолдер-IN на каждый id чанка:
    старая форма падала на лимите хост-параметров SQLite (999) у файлов
    с тысячей+ чанков.
    """
    conn.execute("DELETE FROM chunks_vec WHERE id IN "
                 "(SELECT id FROM chunks WHERE file_path = ?)", (rel_path,))
    conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
    conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))


def cmd_search(args):
    """Search the index."""
    db_path = Path(args.db).resolve()
    mode = args.mode
    # argparse ограничивает choices только в CLI; программные вызовы
    # (eval_search, тесты, импорт) собирают Namespace без проверки —
    # неизвестный mode раньше молча давал пустой результат
    if mode not in _MODES:
        raise ValueError(f"unknown search mode: {mode!r} (expected {'|'.join(_MODES)})")
    if not db_path.exists():
        print("Error: index not found. Run 'skills-search index' first.", file=sys.stderr)
        # JSON в stdout — MCP-обёртка читает stdout до валидного JSON и не
        # ждёт зависший на выходе процесс до таймаута (фаза 1 аудита р3)
        print(json.dumps({"error": "index not found; run 'skills-search index' first",
                          "exists": False}, ensure_ascii=False))
        sys.exit(1)

    conn = init_db(db_path)
    query = args.query
    # Кламп top_k (находка аудита р4): отрицательный/нулевой top-k через
    # программный вызов или CLI давал LIMIT < 0 = «все строки» в SQLite и
    # мусорный срез results[:top_k]
    top_k = max(1, int(args.top_k))
    mode = args.mode
    # Новые флаги — через getattr: программные вызовы (eval_search, тесты)
    # собирают Namespace без них; CLI-совместимость сохраняется
    scope = getattr(args, "scope", None)
    category = getattr(args, "category", None)
    skill = getattr(args, "skill", None)
    include_archived = getattr(args, "include_archived", False)
    abstain = getattr(args, "abstain", True)
    rerank_enabled = getattr(args, "rerank", True)
    source = getattr(args, "source", "knowledge")
    if source not in SOURCE_MODES:
        raise ValueError(f"unknown source: {source!r} "
                         f"(expected {'|'.join(SOURCE_MODES)})")

    meta = _fetch_skill_meta()

    preds = _plan_search_filters(meta, scope, category, skill, include_archived,
                                 source)
    if preds is None:  # пустое разрешённое множество — несовместимые фильтры
        print(f"Warning: фильтры (scope={scope}, category={category}, "
              f"skill={skill}) не соответствуют ни "
              f"одному скилу/стриму — пустой результат", file=sys.stderr)
        print("[]")
        conn.close()
        return
    pred_bm25, pred_vec = preds

    results = _collect_search_results(conn, query, top_k, mode, pred_bm25, pred_vec,
                                      rerank_enabled)
    output = _render_results(conn, results[:top_k], meta)

    # Абстеншен v2 (фаза 6 аудита р3): косинус top-1 НЕфильтрованной векторной
    # ветки — основной сигнал (калибровка на golden v2, search_core.absent_by_cosine);
    # probe не зависит от активных фильтров выдачи. Keyword-режим и fallback BGE —
    # composite low_confidence по top-скорам (фолбэк, ловит лишь явные выбросы).
    # С abstain=False вывод без изменений.
    low_conf = False
    if abstain:
        cos_top1 = None
        if mode in ("hybrid", "semantic"):
            try:
                vec = _search_vector(conn, query, 1)
                if vec:
                    cos_top1 = vec[0]["score"]
            except (httpx.HTTPError, OSError, RuntimeError):
                pass  # BGE недоступна -> composite-фолбэк ниже
        cos_sig = absent_by_cosine(cos_top1)
        low_conf = (cos_sig if cos_sig is not None
                    else low_confidence([r["score"] for r in output]))
        if low_conf:
            print("УВЕРЕННОГО СОВПАДЕНИЯ В БАЗЕ НЕТ (порог абстеншена). "
                  "Слабые кандидаты ниже — только как ориентир.", file=sys.stderr)
        for r in output:
            r["low_confidence"] = low_conf

    print(json.dumps(output, ensure_ascii=False, indent=2))
    conn.close()


# --- Helpers cmd_search -------------------------------------------------------

def _fetch_skill_meta() -> dict:
    """Мета скилов (scope/category/status/updated) — для metadata-фильтров
    и поля updated в выдаче; сбой сканера frontmatter не роняет поиск."""
    try:
        return build_skill_meta(find_all_skills())
    except Exception:
        return {}


def _plan_search_filters(meta: dict, scope, category, skill, include_archived,
                         source: str = "knowledge"):
    """SQL-предикаты metadata-фильтров и плоскости доступа (обе ветки).

    Явные фильтры — только скилы разрешённого множества (canonical →
    каталог для GLOB). Плоскость доступа source комбинируется AND.
    Пустое разрешённое множество -> None (несовместимые фильтры).
    """
    explicit = scope is not None or category is not None or skill is not None
    bm25_parts: list = []
    vec_parts: list = []

    if explicit:
        kept = filter_skill_dirs(meta, scope=scope, category=category, skill=skill,
                                 include_archived=include_archived)
        dirs = {meta[d]["dir"] for d in kept}
        if not dirs:
            return None
        bm25_parts.append(_glob_predicate(dirs, "file_path"))
        vec_parts.append(_glob_predicate(dirs, "c.file_path"))
    else:
        archived = (set() if include_archived else
                    {d for d, m in meta.items() if m["status"] == "archived"})
        if archived:
            dirs = {meta[d]["dir"] for d in archived}
            bm25_parts.append(_glob_predicate(dirs, "file_path", negate=True))
            vec_parts.append(_glob_predicate(dirs, "c.file_path", negate=True))

    bm25_parts.append(_source_predicate(source, "file_path"))
    vec_parts.append(_source_predicate(source, "c.file_path"))

    def _and(parts):
        parts = [p for p in parts if p[0] != "1=1"]
        if not parts:
            return None
        return (" AND ".join(p[0] for p in parts),
                [param for p in parts for param in p[1]])

    return _and(bm25_parts), _and(vec_parts)


def _collect_search_results(conn, query: str, top_k: int, mode: str,
                            pred_bm25, pred_vec, rerank_enabled: bool = True) -> list[dict]:
    """Кандидаты обеих веток + RRF/реранк (hybrid) или дедуп по score.

    semantic при недоступности BGE-M3 — понятная ошибка с exit 1;
    hybrid — деградация на BM25 со stderr-предупреждением.
    rerank_enabled=False (--no-rerank) — hybrid без реранкера, порядок RRF.
    """
    results = []

    if mode in ("hybrid", "keyword"):
        results.extend(_search_bm25(conn, query, top_k * 3, pred_bm25))

    if mode in ("hybrid", "semantic"):
        try:
            # KNN у sqlite-vec применяется ДО join-фильтра, поэтому при
            # активном фильтре замахиваемся шире (до всего индекса —
            # брутфорс-скан всё равно полный, цена только в выборке строк)
            knn_k = None
            if pred_vec is not None:
                n_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                knn_k = min(n_total, max(top_k * 3 * 10, 500))
            results.extend(_search_vector(conn, query, top_k * 3, pred_vec, knn_k))
        except (httpx.HTTPError, OSError, RuntimeError) as e:
            if mode == "semantic":
                print(f"Error: BGE-M3 недоступна ({e}) — используйте --mode keyword",
                      file=sys.stderr)
                conn.close()
                print(json.dumps({"error": f"BGE-M3 недоступна ({e}); "
                                           f"используйте --mode keyword"},
                                 ensure_ascii=False))
                sys.exit(1)
            # hybrid: деградируем на BM25 (зеркально session_search cmd_search)
            print(f"Warning: BGE-M3 недоступна ({e}) — fallback на BM25", file=sys.stderr)

    if mode == "hybrid":
        results = _rrf_fuse(results, k=RRF_K)
        if rerank_enabled:
            # пул реранкера = max(20, top_k), иначе --top-k 50 молча обрезался до 20;
            # кандидаты уже отфильтрованы обеими ветками — пул чист по построению
            results = _rerank(query, results, conn, top_n=max(RERANKER_TOP_N, top_k))
    else:
        # Deduplicate by chunk_id, keep best score
        seen = {}
        for r in results:
            cid = r["chunk_id"]
            if cid not in seen or r["score"] > seen[cid]["score"]:
                seen[cid] = r
        results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    return results


def _render_results(conn, results: list[dict], meta: dict) -> list[dict]:
    """Чанки выдачи -> строки JSON (file/skill/heading/updated/score/snippet)."""
    output = []
    for r in results:
        row = conn.execute(
            "SELECT file_path, skill_name, heading, text FROM chunks WHERE id = ?",
            (r["chunk_id"],),
        ).fetchone()
        if row:
            snippet = row[3][:300].replace("\n", " ")
            output.append({
                "file": row[0],
                "skill": _display_skill(row[0], row[1]),
                "heading": row[2],
                # свежесть скила-владельца чанка из frontmatter (не из БД);
                # чанки вне скилов (streams/, корень) -> null
                "updated": meta.get(skill_dir_of(row[0]), {}).get("updated"),
                "score": round(r["score"], 4),
                "snippet": snippet,
            })
    return output


def _search_bm25(conn: sqlite3.Connection, query: str, limit: int,
                 path_pred=None) -> list[dict]:
    """BM25 keyword search via FTS5 with Snowball stemming.

    path_pred — опциональный (sql, params) предикат на chunks.file_path
    (metadata-фильтры): применяется через rowid-подзапрос до LIMIT.
    """
    # Tokenize, stem each word, then join with OR for FTS5 (column-filter
    # stemmed_text: match только по тексту чанка, не по skill_name/heading)
    stemmed_words = stem_query(query)
    if not stemmed_words:
        return []
    # column-filter против phantom-хитов: без него запрос «grafana» матчит
    # имена скилов entity-grafana, а не только текст чанков (см. search_core)
    fts_query = fts_column_query(stemmed_words, "stemmed_text")

    sql = "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ?"
    params = [fts_query]
    if path_pred is not None:
        pred_sql, pred_params = path_pred
        sql += f" AND rowid IN (SELECT id FROM chunks WHERE {pred_sql})"
        params += pred_params
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        # битый/отсутствующий chunks_fts или синтаксис MATCH — не молчим
        print(f"Warning: BM25-запрос не выполнен ({e}) — keyword-ветка пуста",
              file=sys.stderr)
        return []

    results = []
    for rowid, rank in rows:
        # bm25() в FTS5: меньше (отрицательнее) = лучше; score = -rank сохраняет
        # «лучше — выше». 1/(1+|rank|) инвертировал порядок — keyword-ветка
        # отдавала хвост вместо головы (гибрид маскировал реранкером).
        results.append({"chunk_id": rowid, "score": -rank, "source": "bm25"})
    return results


def _search_vector(conn: sqlite3.Connection, query: str, limit: int,
                   path_pred=None, knn_k: int = None) -> list[dict]:
    """Vector similarity search via sqlite-vec.

    path_pred — опциональный (sql, params) предикат на chunks (алиас c.):
    JOIN-фильтр sqlite-vec применяет ПОСЛЕ KNN, поэтому при активном
    фильтре вызывающий передаёт расширенный knn_k; здесь результат
    обрезается до limit (топ отфильтрованного множества).
    """
    query_embedding = embed_texts([QUERY_PREFIX + query])[0]
    k = knn_k if knn_k is not None else limit
    if path_pred is not None:
        pred_sql, pred_params = path_pred
        sql = ("SELECT v.id, v.distance FROM chunks_vec v "
               "JOIN chunks c ON c.id = v.id "
               f"WHERE v.embedding MATCH ? AND v.k = ? AND {pred_sql}")
        params = [serialize_f32(query_embedding), k] + list(pred_params)
    else:
        sql = ("SELECT id, distance FROM chunks_vec "
               "WHERE embedding MATCH ? AND k = ?")
        params = [serialize_f32(query_embedding), k]
    rows = conn.execute(sql + " ORDER BY distance", params).fetchall()

    results = []
    for chunk_id, distance in rows[:limit]:
        # sqlite-vec cosine distance: 0 = identical, 2 = opposite
        score = 1.0 - (distance / 2.0)
        results.append({"chunk_id": chunk_id, "score": score, "source": "vector"})
    return results


def cmd_status(args):
    """Show index statistics."""
    db_path = Path(args.db).resolve()
    as_json = getattr(args, "json", False)
    if not db_path.exists():
        if as_json:
            print(json.dumps({"error": "index not found", "exists": False},
                             ensure_ascii=False))
        else:
            print("No index found.")
        return

    conn = init_db(db_path)
    n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_vec = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]

    db_size = db_path.stat().st_size / 1024 / 1024

    # Show top skills by chunk count
    rows = conn.execute(
        "SELECT skill_name, COUNT(*) as cnt FROM chunks GROUP BY skill_name ORDER BY cnt DESC LIMIT 10"
    ).fetchall()

    if as_json:
        # машиночитаемый режим для MCP (skills_search_status): run_cli
        # ждёт JSON-объект, человеческий вывод остаётся дефолтом CLI
        print(json.dumps({
            "exists": True,
            "db_path": str(db_path),
            "files": n_files,
            "chunks": n_chunks,
            "vectors": n_vec,
            "db_size_mb": round(db_size, 1),
            "top_skills": [{"skill": s, "chunks": c} for s, c in rows],
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Index: {db_path}")
        print(f"Files:  {n_files}")
        print(f"Chunks: {n_chunks}")
        print(f"Vectors: {n_vec}")
        print(f"DB size: {db_size:.1f} MB")

        if rows:
            print("\nTop skills by chunks:")
            for skill, cnt in rows:
                print(f"  {skill}: {cnt}")

    conn.close()


def cmd_reindex(args):
    """Full reindex: сборка нового индекса во временный файл + атомарная замена.

    Старый индекс остаётся нетронутым до полного успеха пересборки: падение
    embedding-модели на полпути больше не оставляет базу знаний без индекса
    (раньше _drop_db_files бежал до первого байта нового индекса).
    """
    db_path = Path(args.db).resolve()
    tmp_db = Path(str(db_path) + ".tmp")
    _drop_db_files(tmp_db)  # мусор от прошлого оборванного reindex
    tmp_args = argparse.Namespace(path=args.path, db=str(tmp_db),
                                  json=getattr(args, "json", False))
    try:
        cmd_index(tmp_args)
    except BaseException:
        _drop_db_files(tmp_db)
        print(f"reindex failed — старый индекс не тронут: {db_path}",
              file=sys.stderr)
        raise
    # Соединение с tmp закрыто (cmd_index) — вычистить хвосты WAL временной БД
    for suffix in ("-wal", "-shm"):
        p = Path(str(tmp_db) + suffix)
        if p.exists():
            p.unlink()
    _drop_db_files(db_path)
    os.replace(tmp_db, db_path)
    print(f"Reindexed: {db_path}")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="skills-search",
        description="Hybrid BM25 + vector search over markdown knowledge base",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help=f"Path to SQLite index (default: {DEFAULT_DB_PATH})")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index/reindex markdown files")
    p_index.add_argument("--path", default=str(DEFAULT_SKILLS_PATH),
                         help="Path to skills directory")
    p_index.add_argument("--json", action="store_true",
                         help="machine-readable output (JSON summary)")

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=10, help="Number of results")
    p_search.add_argument("--mode", choices=["hybrid", "semantic", "keyword"],
                          default="hybrid", help="Search mode")
    p_search.add_argument("--scope", choices=list(VALID_SCOPES), default=None,
                          help="фильтр по scope из frontmatter скила")
    p_search.add_argument("--category", choices=list(VALID_CATEGORIES), default=None,
                          help="фильтр по category из frontmatter скила")
    p_search.add_argument("--skill", default=None, metavar="DIR",
                          help="только этот скил (имя директории или frontmatter name)")
    p_search.add_argument("--include-archived", action="store_true",
                          help="включить скилы со status: archived (по умолчанию исключены)")
    p_search.add_argument("--abstain", dest="abstain", action="store_true",
                          default=True,
                          help="абстеншен-порог: при низкой уверенности "
                               "предупредить, что уверенного совпадения нет "
                               "(по умолчанию включён; --no-abstain отключает)")
    p_search.add_argument("--no-abstain", dest="abstain", action="store_false",
                          help="отключить абстеншен-предупреждение и поле "
                               "low_confidence в выдаче")
    p_search.add_argument("--no-rerank", dest="rerank", action="store_false",
                          default=True,
                          help="отключить BGE-reranker в hybrid (порядок RRF); "
                               "по умолчанию реранк включён (в session-search "
                               "наоборот: --rerank opt-in — реранкер нестабилен "
                               "на коротких диалоговых чанках)")
    p_search.add_argument("--source", choices=list(SOURCE_MODES), default="knowledge",
                          help="плоскость доступа: knowledge (дефолт; всё, кроме "
                               "файлов задач — задачи не вытесняют знания), skills "
                               "(только скилы), streams, tasks (файлы задач), all")

    p_status = sub.add_parser("status", help="Show index statistics")
    p_status.add_argument("--json", action="store_true",
                          help="machine-readable output (JSON)")

    p_reindex = sub.add_parser("reindex", help="Full reindex (drop + rebuild)")
    p_reindex.add_argument("--path", default=str(DEFAULT_SKILLS_PATH),
                           help="Path to skills directory")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "reindex":
        cmd_reindex(args)


if __name__ == "__main__":
    main()
