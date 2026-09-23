# agent-memory-search

Гибридный поисковый движок для баз знаний ИИ-агентов и журналов сессий:
ключевые слова BM25 (SQLite FTS5 + стеммеры Snowball для русского и английского),
векторный поиск (sqlite-vec + эмбеддинги BGE-M3 через любой OpenAI-совместимый
эндпоинт), слияние по reciprocal rank fusion, опциональный реранкинг
крос-энкодером и отказ от ответа по уверенности, откалиброванный на
«золотых» запросах.

Движок обслуживает долговременную память ИИ-агентов, хранимую как локальные
Markdown-записи (`<тема>/SKILL.md` с YAML-frontmatter), но к предметной
области не привязан: проиндексировать можно любой каталог Markdown-файлов.

## Компоненты

| Часть | Путь | Назначение |
| --- | --- | --- |
| Поисковый CLI | `tools/skills-search.py` | `index` / `search` / `status` / `reindex` по базе знаний |
| Ядро поиска | `tools/search_core.py` | BM25, векторный индекс, слияние RRF, реранкинг, отказ по уверенности |
| Чанкер Markdown | `tools/chunker_md.py` | разбиение Markdown по заголовкам и блокам кода |
| Поиск по проекту | `tools/project_search.py` | гибридный поиск по документам проекта и комментариям исходников |
| Поиск по сессиям | `session-search/` | инкрементальная индексация журналов сессий агентов (JSONL), свой CLI + MCP-сервер |
| MCP-серверы | `tools/skills_search_mcp.py`, `session-search/mcp_server.py` | публикация поиска для MCP-клиентов |
| Обслуживание | `skills-maintain.py`, `skills-enrich.py`, `skills-graph.py`, `tasks.py` | линт, LLM-обогащение, граф сущностей, слой задач |

## Быстрый старт

Требуется Python 3.11+.

```bash
pip install -r requirements.txt
```

Укажите движку базу знаний (каталог папок-записей):

```bash
export SKILLS_ROOT=/path/to/knowledge

python tools/skills-search.py index
python tools/skills-search.py search "vllm offload" --json
```

Для векторного поиска и реранкинга нужны адреса эмбеддера и реранкера —
скопируйте `config/skills-gateway.example.json` в `~/.config/skills-gateway.json`
или задайте переменные `SKILLS_EMBEDDING_URL` / `SKILLS_RERANKER_URL`
(OpenAI-совместимые эндпоинты). Без эндпоинтов движок работает в режиме
ключевых слов:

```bash
python tools/skills-search.py search "запрос" --mode keyword
```

Поиск по сессиям индексирует JSONL-журналы сессий агентов (раскладка
настраивается):

```bash
python session-search/scripts/session_search.py index
python session-search/scripts/session_search.py search "past work" --top-k 5
```

## Тесты

```bash
python -m unittest discover -s tests -v
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
