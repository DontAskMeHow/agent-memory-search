# agent-memory-search

Hybrid search engine for AI-agent knowledge bases and session logs:
keyword BM25 (SQLite FTS5 + Snowball stemmers for Russian and English),
vector search (sqlite-vec + BGE-M3 embeddings via any OpenAI-compatible
endpoint), reciprocal rank fusion, optional cross-encoder reranking, and
confidence-based abstention calibrated on golden queries.

The engine manages long-term memory of AI agents stored as local Markdown
records (`<topic>/SKILL.md` with YAML frontmatter), but it is domain-agnostic:
any directory of Markdown files can be indexed.

## Components

| Part | Path | Purpose |
| --- | --- | --- |
| Search CLI | `tools/skills-search.py` | `index` / `search` / `status` / `reindex` over a knowledge base |
| Search core | `tools/search_core.py` | BM25, vector index, RRF fusion, rerank, abstention |
| Markdown chunker | `tools/chunker_md.py` | heading- and fence-aware chunking of Markdown |
| Project search | `tools/project_search.py` | hybrid search over project docs and source comments |
| Session search | `session-search/` | incremental indexing of AI-agent session logs (JSONL), own CLI + MCP server |
| MCP servers | `tools/skills_search_mcp.py`, `session-search/mcp_server.py` | expose search to MCP clients |
| Maintenance | `skills-maintain.py`, `skills-enrich.py`, `skills-graph.py`, `tasks.py` | lint, LLM enrichment, entity graph, task layer |

## Quick start

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

Point the engine at a knowledge base (directory of record folders):

```bash
export SKILLS_ROOT=/path/to/knowledge

python tools/skills-search.py index
python tools/skills-search.py search "vllm offload" --json
```

For vector search and reranking, provide embedding/reranker endpoints — copy
`config/skills-gateway.example.json` to `~/.config/skills-gateway.json`, or set
`SKILLS_EMBEDDING_URL` / `SKILLS_RERANKER_URL` (OpenAI-compatible endpoints).
Without endpoints the engine still works in keyword mode:

```bash
python tools/skills-search.py search "запрос" --mode keyword
```

Session search indexes JSONL session logs of AI-agents (layout is
configurable):

```bash
python session-search/scripts/session_search.py index
python session-search/scripts/session_search.py search "past work" --top-k 5
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
