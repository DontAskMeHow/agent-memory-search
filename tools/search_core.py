#!/usr/bin/env python3
"""search_core — поисковые примитивы hybrid-поиска (BM25 + vector).

Канонические копии embed/serialize/stem/RRF/rerank, выделенные в фазе 3
аудита памяти из tools/skills-search.py и
session-search/scripts/session_search.py (почти идентичные дубли).

Потребители импортируют нужные имена в свой namespace
(`from search_core import ...`) — тесты и внешние вызовы продолжают
обращаться к атрибутам модулей-скриптов.

Зависимости: httpx, snowballstemmer (в нашем воркспейсе — базовый Python 3.14
без venv; у коллег — в .venv базы знаний);
sqlite-vec — опционально (guarded import, см. init_vec_conn).
"""

import os
import re
import sqlite3
import struct
import time

import httpx
import json
import snowballstemmer

try:
    import sqlite_vec  # noqa: F401
except Exception:  # pragma: no cover
    sqlite_vec = None

# --- Конфигурация моделей (BGE-M3 / реранкер через LLM Gateway) ---------------
# Адреса: env SKILLS_EMBEDDING_URL / SKILLS_RERANKER_URL, затем локальный
# конфиг машины ~/.config/skills-gateway.json (embedding_url / reranker_url).
# Персональный адрес шлюза сознательно НЕ зашит в код: у каждого разработчика
# он свой и в репозиториях не хранится. Без адреса semantic-моды дают понятную
# ошибку, реранкер деградирует на RRF-порядок.

_GATEWAY_CONFIG = os.path.join(
    os.path.expanduser("~"), ".config", "skills-gateway.json")


def _local_gateway() -> dict:
    try:
        with open(_GATEWAY_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_LOCAL_GW = _local_gateway()

EMBEDDING_URL = (os.environ.get("SKILLS_EMBEDDING_URL")
                 or _LOCAL_GW.get("embedding_url"))
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024
EMBED_BATCH = 32
EMBED_TIMEOUT = 60.0
QUERY_PREFIX = ""  # BGE-M3 не требует query-инструкции; убран после A/B (2026-08-29)

RERANKER_URL = (os.environ.get("SKILLS_RERANKER_URL")
                or _LOCAL_GW.get("reranker_url"))
RERANKER_MODEL = "bge-reranker-v2-m3"
RERANKER_TOP_N = 20  # минимальный пул RRF-кандидатов для реранкера (растёт до top_k)

RRF_K = 60  # константа Reciprocal Rank Fusion: 1/(k + rank + 1)

# --- Стемминг (Snowball ru/en) -------------------------------------------------

_stemmer_ru = snowballstemmer.stemmer("russian")
_stemmer_en = snowballstemmer.stemmer("english")


def _is_cyrillic(word: str) -> bool:
    """Слово кириллическое, если большинство alpha-символов — кириллица."""
    cyr = sum(1 for c in word if '\u0400' <= c <= '\u04FF')
    lat = sum(1 for c in word if c.isascii() and c.isalpha())
    return cyr > lat


def stem_text(text: str) -> str:
    """Стемминг текста: Snowball Russian для кириллицы, English для латиницы.

    Не-словные токены проходят как есть; результат склеивается в исходную
    форму (пробелы/пунктуация сохраняются).
    """
    tokens = re.findall(r'\w+|\W+', text, re.UNICODE)
    result = []
    for token in tokens:
        if not token or not token[0].isalnum():
            result.append(token)
        elif _is_cyrillic(token):
            result.append(_stemmer_ru.stemWord(token.lower()))
        elif token.isascii():
            result.append(_stemmer_en.stemWord(token.lower()))
        else:
            result.append(token.lower())
    return "".join(result)


def stem_query(query: str) -> list:
    """Слова запроса (len >= 2), каждое простеммлено — для FTS5 MATCH.

    Потребитель склеивает через " OR " (BM25-поиск обеих CLI).
    """
    words = re.findall(r'\w{2,}', query, re.UNICODE)
    stemmed = []
    for w in words:
        if _is_cyrillic(w):
            stemmed.append(_stemmer_ru.stemWord(w.lower()))
        elif w.isascii():
            stemmed.append(_stemmer_en.stemWord(w.lower()))
        else:
            stemmed.append(w.lower())
    return stemmed


def fts_column_query(terms: list, column: str = "stemmed_text") -> str:
    """FTS5 MATCH-выражение с column-filter: токены ищутся ТОЛЬКО в `column`.

    Без column-filter FTS5 матчит по всем колонкам таблицы — имена скилов
    (skill_name) и заголовки (heading) дают phantom-хиты запросов.
    Термы — простеммленные слова; кавычки внутри экранируются удвоением.
    """
    quoted = ['"' + str(t).replace('"', '""') + '"' for t in terms]
    return f"{column} : (" + " OR ".join(quoted) + ")"


# --- Эмбеддинги (BGE-M3) ------------------------------------------------------

def _embed_request(batch: list, url: str = EMBEDDING_URL,
                   model: str = EMBEDDING_MODEL,
                   timeout: float = EMBED_TIMEOUT) -> list:
    """Один batch-POST к BGE-M3 с одной повторной попыткой при ошибке
    транспорта (httpx.TransportError/OSError) и на HTTP 5xx (gateway иногда
    моргает); две попытки максимум, зацикливания нет. 4xx и malformed
    payload не ретраятся — разбор тела обёрнут в понятную ошибку.

    Фолбэк requests (2026-09-22): встречен маршрут сети, который стабильно
    отдаёт пустой 503 именно httpx-клиенту (`proxy-connection: close`), а
    urllib3/requests при тех же заголовках проходят — третья попытка идёт
    через requests."""
    if not url:
        raise RuntimeError(
            "BGE-M3 endpoint не задан: установите SKILLS_EMBEDDING_URL "
            "или пропишите ~/.config/skills-gateway.json")
    last_exc = None
    for attempt in range(2):
        if attempt:
            time.sleep(1.0)
        try:
            resp = httpx.post(
                url,
                json={"model": model, "input": batch},
                timeout=timeout,
            )
            resp.raise_for_status()
            try:
                return resp.json()["data"]
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(
                    f"BGE-M3 вернула malformed ответ (нет .data): {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise
            last_exc = e
        except (httpx.TransportError, OSError) as e:
            last_exc = e
    try:
        import requests
        r = requests.post(
            url, json={"model": model, "input": batch},
            timeout=(5.0, timeout))
        if r.status_code >= 500:
            raise last_exc
        r.raise_for_status()
        return r.json()["data"]
    except requests.RequestException:
        raise last_exc
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"BGE-M3 вернула malformed ответ (нет .data, fallback): {e}") from e


def embed_texts(texts: list, batch_size: int = EMBED_BATCH,
                url: str = EMBEDDING_URL, model: str = EMBEDDING_MODEL,
                timeout: float = EMBED_TIMEOUT) -> list:
    """Эмбеддинги текстов через OpenAI-совместимый endpoint (батчами)."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        data = _embed_request(batch, url=url, model=model, timeout=timeout)
        # Sort by index to preserve order
        try:
            data.sort(key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in data])
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"BGE-M3 вернула malformed ответ (нет index/embedding): {e}"
            ) from e
    return all_embeddings


def serialize_f32(vec: list) -> bytes:
    """Упаковать float-вектор для sqlite-vec (native float32)."""
    return struct.pack(f"{len(vec)}f", *vec)


# --- RRF-слияние --------------------------------------------------------------

def rrf_fuse(results: list, k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion: слияние ранжирований BM25 и vector.

    score(id) = Σ по источникам 1/(k + rank + 1), rank с нуля внутри
    каждого source (сортировка по score по убыванию).
    """
    by_source: dict = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r)

    for source in by_source:
        by_source[source].sort(key=lambda x: x["score"], reverse=True)

    rrf_scores: dict = {}
    for source, items in by_source.items():
        for rank, item in enumerate(items):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)

    fused = [{"chunk_id": cid, "score": score, "source": "rrf"}
             for cid, score in rrf_scores.items()]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


# --- Реранкер (BGE-Reranker-v2-M3) --------------------------------------------

def rerank(query: str, candidates: list, get_texts, top_n: int = RERANKER_TOP_N,
           url: str = RERANKER_URL, model: str = RERANKER_MODEL,
           timeout: float = 30.0) -> list:
    """Переранжировать топ-N кандидатов через BGE-Reranker-v2-M3.

    candidates — список dict с "chunk_id" (пул RRF-результатов);
    get_texts(ids) → {chunk_id: text} — колбэк получения текста кандидата
    (у скриптов разные схемы БД). При недоступности реранкера — fallback
    на исходный порядок (полным пулом, не обрезанным до top_n по умолчанию).
    """
    if not candidates:
        return candidates
    pool = candidates[:top_n]
    chunk_ids = [c["chunk_id"] for c in pool]
    text_map = get_texts(chunk_ids)

    documents = [text_map.get(c["chunk_id"], "") for c in pool]

    try:
        resp = httpx.post(
            url,
            json={"model": model, "query": query, "documents": documents},
            timeout=timeout,
        )
        resp.raise_for_status()
        reranked = resp.json()["results"]
        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        result = []
        for r in reranked:
            idx = r["index"]
            result.append({
                "chunk_id": pool[idx]["chunk_id"],
                "score": r["relevance_score"],
                "source": "reranked",
            })
        return result
    except Exception:
        # Fallback: return original RRF order if reranker unavailable
        return pool


def rerank_from_db(query: str, candidates: list, conn,
                   top_n: int = RERANKER_TOP_N) -> list:
    """Переранжировать кандидатов через BGE-Reranker; тексты — из таблицы
    `chunks` (id, text) — общая схема skills-search и session-search.

    Обёртка фазы 4 аудита р4: прежде идентичные копии жили в обоих
    CLI-скриптах (skills_search._rerank / session_search._rerank).
    """
    def _texts(chunk_ids: list) -> dict:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    return rerank(query, candidates, _texts, top_n=top_n)


# --- Абстеншен (low confidence) ------------------------------------------------

# Калибровано на golden v2 (tools/tests/golden_queries.json, 60 запросов,
# 2026-09-07, калибровочный прогон 3×60 запросов через eval_search.py):
# полное разделение abstention/normal невозможно — скоры реранкера у обеих
# групп кластеризуются у ~0.99 (min gap top1−top2 среди normal = 0.0,
# нормальные top1-скоры 0.9831..0.9988 перекрываются abstention 0.8956..0.9980).
# Приоритет калибровки — ни одного ложного срабатывания на normal-запросах:
# при ABS=0.90, MARGIN=0.0 порог ловит 1 из 12 golden-abstention
# (q039 «рецепт борща», top1=0.8956 — единственный выброс ниже 0.98),
# false-positive на normal — 0 (recall@10 не проседает). Остальные 11
# abstention-запросов сознательно не отсекаются (допустимо по ТЗ: часть
# мусора не отсечь, чем ронять normal).
ABS_ABSENT_THRESHOLD = 0.90
MARGIN_MIN = 0.0

# Абстеншен v2 (фаза 6 аудита р3, 2026-09-08): косинус top-1 сырой векторной
# ветки (BGE-M3, без фильтров) — единственный сигнал, разделяющий мусорные и
# нормальные запросы на golden v2 (63 запроса): abstention 0.4399..0.5159,
# normal 0.5202..0.6797, коридор разделения ~0.004. Порог — середина коридора.
# Скоры реранкера для этой задачи непригодны (на мусоре 0.98+, перекрытие с
# normal почти полное — см. комментарий к ABS_ABSENT_THRESHOLD выше).
# Хрупкость: коридор узкий, при существенном росте корпуса переканить
# калибровку (скрипт фазы 6 сохранён в стриме session10-audit-r3) и
# пересмотреть порог; дрейф ловит метрика «правильный отказ» в eval_search.
COS_ABSENT_THRESHOLD = 0.518


def absent_by_cosine(cos_top1: float | None,
                     threshold: float = COS_ABSENT_THRESHOLD) -> bool | None:
    """Сигнал «в базе нет ничего близкого к запросу» по косинусу top-1.

    cos_top1 — косинус запроса к ближайшему чанку НЕфильтрованного индекса
    (сырая векторная ветка, score = 1 - distance/2 из _search_vector).
    None — векторная ветка не работала (keyword-режим, fallback BGE):
    сигнал недоступен, вызывающий использует composite low_confidence().
    """
    if cos_top1 is None:
        return None
    return cos_top1 < threshold


def low_confidence(top_scores: list, absent_threshold: float = ABS_ABSENT_THRESHOLD,
                   margin_min: float = MARGIN_MIN) -> bool:
    """Фолбэк-сигнал «уверенного совпадения нет» для top-скоров выдачи.

    Основной сигнал — absent_by_cosine (косинус векторной ветки, калибровка
    фазы 6 аудита р3); этот composite применяется, когда векторная ветка не
    работала (keyword-режим, fallback BGE), и по калибровке golden v2 ловит
    лишь явные выбросы (см. комментарий к ABS_ABSENT_THRESHOLD).

    Композитный порог (по убыванию уверенности): срабатывает, если либо
    скор лучшего результата ниже ABS_ABSENT_THRESHOLD (планка «нерелевантно»),
    либо разрыв top1−top2 меньше MARGIN_MIN (нет явного победителя).

    top_scores — скоры первых результатов выдачи по убыванию релевантности
    (обычно первые top_k строк CLI/MCP; минимум 2 значения для margin-ветки).
    Пустой список — признак «совпадений нет» (True).
    """
    if not top_scores:
        return True
    if top_scores[0] < absent_threshold:
        return True
    if len(top_scores) > 1 and (top_scores[0] - top_scores[1]) < margin_min:
        return True
    return False


# --- sqlite-vec ---------------------------------------------------------------

def init_vec_conn(conn: sqlite3.Connection, strict: bool = False) -> None:
    """Загрузить расширение sqlite-vec в соединение SQLite.

    strict=True — упасть, если sqlite-vec недоступен (требование
    skills-search: векторный поиск обязателен); strict=False — молча
    пропустить (толерантность session-search: keyword-поиск работает
    и без sqlite-vec).
    """
    conn.enable_load_extension(True)
    try:
        if sqlite_vec is None:
            raise ImportError("sqlite-vec not installed")
        sqlite_vec.load(conn)
    except Exception:
        if strict:
            raise
    finally:
        conn.enable_load_extension(False)
