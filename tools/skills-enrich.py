#!/usr/bin/env python3
"""
skills-enrich: LLM-обогащение метаданных скилов (legacy/fallback).

Commands:
    extract [--all | --missing | --file F] [--model R]  # LLM-извлечение proposals
    apply [--dry-run] [--skill S]                        # применить proposals к frontmatter
    migrate-entities                                     # graph.db → entities в frontmatter

Вынесен из skills-maintain.py в фазе 4 аудита памяти: lint/dedup работают
без сети и LLM, enrich — единственный сетевой потребитель.

Enrich — legacy: основное интеллектуальное обслуживание выполняет саб-агент
(см. skills-management, раздел «Обслуживание памяти»); enrich остаётся
fallback'ом для случаев без агента.

Двухшаговый proposal→apply flow:
  1. extract: LLM читает скилы → .graph/proposals.json (gitignored)
  2. apply: proposals применяются к frontmatter SKILL.md (related, category,
     status, entities) — детерминированно и ревьюимо.

С фазы 4 enrich НЕ пишет в graph.db: source of truth для entities — поле
`entities:` в frontmatter (git), граф наполняется только `skills-graph
build` из frontmatter.
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import httpx

from skills_lib import (
    GRAPH_DB_PATH,
    GRAPH_DIR,
    PROPOSALS_PATH,
    SKILLS_PATH,
    VALID_CATEGORIES,
    find_all_skills,
    update_frontmatter,
)

# --- Configuration -----------------------------------------------------------

MODELS_CONFIG_PATH = Path(__file__).resolve().parent / "models.json"

# Defaults if models.json is missing or unreadable
_DEFAULT_GATEWAY = "http://127.0.0.1:8040"
_DEFAULT_MODELS = [
    {"route": "llmops", "model": "claude-4-6-opus"},
    {"route": "local/kimi"},
    {"route": "local/deepseek"},
    {"route": "local/qwen27b"},
]


def _load_models_config() -> tuple[str, list[dict]]:
    """Load gateway and model priority from tools/models.json.

    Falls back to built-in defaults if file is missing or broken.
    """
    if MODELS_CONFIG_PATH.exists():
        try:
            cfg = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
            gateway = cfg.get("gateway_base", _DEFAULT_GATEWAY)
            models = cfg.get("models", _DEFAULT_MODELS)
            return gateway, models
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: failed to parse {MODELS_CONFIG_PATH}: {e}", file=sys.stderr)
    return _DEFAULT_GATEWAY, _DEFAULT_MODELS


GATEWAY_BASE, MODEL_PRIORITY = _load_models_config()

# VALID_CATEGORIES — общая enum-константа из skills_lib (см. выше)


# --- LLM interaction --------------------------------------------------------

def _check_model_available(route: str, model_hint: str | None = None) -> str | None:
    """Check if a model endpoint is available. Returns model name or None.

    If model_hint is given, verify that exact model is in the list
    (for multi-model gateways like llmops). Otherwise take the first model.
    """
    try:
        resp = httpx.get(f"{GATEWAY_BASE}/{route}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if not models:
                return None
            if model_hint:
                for m in models:
                    if m["id"] == model_hint:
                        return model_hint
                return None
            return models[0]["id"]
    except Exception:
        pass
    return None


def _select_model(override: str | None = None) -> tuple[str, str] | None:
    """Select LLM model. Returns (route, model_name) or None if all unavailable.

    If override is given, use that route. Otherwise try priority list
    from tools/models.json (Opus → Kimi → DeepSeek → Qwen3.8).

    Models are checked in priority order. The first available one is used.
    If none is available, returns None instead of crashing — the caller
    decides whether to abort or skip.
    """
    if override:
        model_name = _check_model_available(override)
        if model_name:
            return override, model_name
        print(f"Warning: {override} unavailable", file=sys.stderr)

    tried = []
    for m in MODEL_PRIORITY:
        route = m["route"]
        model_hint = m.get("model")
        model_name = _check_model_available(route, model_hint)
        if model_name:
            print(f"Using model: {model_name} ({route})")
            return route, model_name
        tried.append(f"{route}" + (f"/{model_hint}" if model_hint else ""))

    print(f"Error: no LLM model available. Tried: {', '.join(tried)}", file=sys.stderr)
    print(f"Check models config: {MODELS_CONFIG_PATH}", file=sys.stderr)
    print(f"Gateway: {GATEWAY_BASE}", file=sys.stderr)
    return None


def _llm_extract(route: str, model_name: str, skill_text: str,
                 skill_name: str, all_skill_names: list[str]) -> dict:
    """Call LLM to extract metadata from a skill."""
    names_list = ", ".join(all_skill_names)

    prompt = f"""Проанализируй содержимое скила и верни JSON с метаданными.

ВАЖНО: верни ТОЛЬКО валидный JSON, без markdown-обёртки, без пояснений.

Поля JSON:
- "category": один из [{", ".join(VALID_CATEGORIES)}]
- "related": список имён связанных скилов из доступных (выбери подходящие)
- "status": "active" или "archived" (archived если информация устарела)
- "entities": 2-6 важнейших сущностей - отдельные технологии/продукты/серверы/модели с собственным именем (например ClickHouse, vLLM, Kafka, Grafana, BGE-M3). НЕ включать: общие слова и абстракции (report, Python, Docker), внутренние сокращения/аббревиатуры, названия полей/таблиц, дата-центры, проекты-зонтики.

Доступные скилы для related: [{names_list}]

Текущий скил "{skill_name}":
---
{skill_text[:6000]}
---

Верни JSON:"""

    resp = httpx.post(
        f"{GATEWAY_BASE}/{route}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code block if present
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        m = re.search(r'\{[\s\S]*\}', content)
        if m:
            result = json.loads(m.group())
        else:
            raise ValueError(f"Could not parse LLM response as JSON: {content[:200]}")

    # Validate and sanitize
    valid_names = set(all_skill_names)
    result["category"] = result.get("category", "reference")
    if result["category"] not in VALID_CATEGORIES:
        result["category"] = "reference"

    result["related"] = [r for r in result.get("related", [])
                         if r in valid_names and r != skill_name]
    result["status"] = result.get("status", "active")
    if result["status"] not in ("active", "archived", "draft"):
        result["status"] = "active"

    result["entities"] = result.get("entities", [])
    if not isinstance(result["entities"], list):
        result["entities"] = []

    return result


# --- Commands ----------------------------------------------------------------

def cmd_extract(args):
    """Extract metadata proposals using LLM."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    result = _select_model(args.model)
    if result is None:
        print("Skipping enrich: no LLM available. Fix models in tools/models.json",
              file=sys.stderr)
        sys.exit(1)
    route, model_name = result

    skills = find_all_skills()
    all_names = [s["name"] for s in skills]

    # Filter skills to process
    if args.file:
        skills = [s for s in skills if args.file in s["file"]]
        if not skills:
            print(f"Error: no skill found matching '{args.file}'", file=sys.stderr)
            sys.exit(1)
    elif args.missing:
        skills = [s for s in skills
                  if not s["frontmatter"].get("related") or not s["frontmatter"].get("category")]
    # else: --all

    if not skills:
        print("All skills already have metadata.")
        return

    # Load existing proposals (to append/update)
    proposals = []
    if PROPOSALS_PATH.exists():
        proposals = json.loads(PROPOSALS_PATH.read_text(encoding="utf-8"))
    existing_skills = {p["skill"] for p in proposals}

    print(f"Extracting metadata for {len(skills)} skills using {model_name}...")
    t0 = time.time()

    for i, skill in enumerate(skills):
        name = skill["name"]
        print(f"  [{i+1}/{len(skills)}] {name}...", end=" ", flush=True)

        try:
            result = _llm_extract(route, model_name, skill["full_text"],
                                  name, all_names)
            proposal = {
                "skill": name,
                "file": skill["file"],
                "proposed": {
                    "category": result["category"],
                    "related": result["related"],
                    "status": result["status"],
                    "entities": result["entities"],
                },
                "model": model_name,
                "extracted_at": date.today().isoformat(),
            }

            # Update or append
            if name in existing_skills:
                proposals = [p if p["skill"] != name else proposal for p in proposals]
            else:
                proposals.append(proposal)
                existing_skills.add(name)

            print(f"category={result['category']}, "
                  f"related={len(result['related'])}, "
                  f"entities={len(result['entities'])}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Save after each skill (incremental, survives interruption)
        PROPOSALS_PATH.write_text(
            json.dumps(proposals, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Proposals saved to {PROPOSALS_PATH}")
    print(f"Review with: cat {PROPOSALS_PATH}")
    print(f"Apply with:  skills-enrich.py apply [--dry-run]")


def cmd_apply(args):
    """Apply proposals to skill frontmatter.

    Все поля (related/category/status/entities) пишутся в frontmatter
    через skills_lib.update_frontmatter. graph.db не трогается — граф
    пересобирается отдельно: `skills-graph build`.
    """
    if not PROPOSALS_PATH.exists():
        print("No proposals found. Run 'extract' first.", file=sys.stderr)
        sys.exit(1)

    proposals = json.loads(PROPOSALS_PATH.read_text(encoding="utf-8"))
    if not proposals:
        print("No proposals to apply.")
        return

    # Filter by skill name if specified
    if args.skill:
        proposals = [p for p in proposals if p["skill"] == args.skill]
        if not proposals:
            print(f"No proposal for skill '{args.skill}'", file=sys.stderr)
            sys.exit(1)

    today = date.today().isoformat()

    applied = 0
    for p in proposals:
        skill_path = SKILLS_PATH / p["file"]
        if not skill_path.exists():
            print(f"  SKIP {p['skill']}: file not found ({p['file']})")
            continue

        proposed = p["proposed"]
        updates = {}

        if proposed.get("category"):
            updates["category"] = proposed["category"]
        if proposed.get("related"):
            updates["related"] = proposed["related"]
        if proposed.get("status"):
            updates["status"] = proposed["status"]
        if proposed.get("entities"):
            updates["entities"] = sorted(proposed["entities"])
        updates["updated"] = today

        if args.dry_run:
            print(f"  {p['skill']}:")
            for k, v in updates.items():
                print(f"    {k}: {v}")
            continue

        # Update frontmatter
        text = skill_path.read_text(encoding="utf-8")
        new_text = update_frontmatter(text, updates)
        skill_path.write_text(new_text, encoding="utf-8")

        applied += 1
        print(f"  ✓ {p['skill']}: category={updates.get('category')}, "
              f"related={len(proposed.get('related', []))}, "
              f"entities={len(proposed.get('entities', []))}")

    if args.dry_run:
        print(f"\nDry run: {len(proposals)} skills would be updated.")
    else:
        print(f"\nApplied: {applied}/{len(proposals)} skills updated.")
        print("Rebuild the graph to pick up new entities: skills-graph.py build")


def cmd_migrate_entities(args):
    """Одноразовая миграция (фаза 4 аудита): entities из graph.db → frontmatter.

    Для каждого скила берёт список entity-имён из таблицы skill_entities
    и записывает в frontmatter его SKILL.md поле `entities: [a, b, c]`
    (алфавитная сортировка). Поле `updated` НЕ поднимается — это машинные
    метаданные, а не контентная правка.

    Идемпотентно: скилы с уже совпадающим полем пропускаются, повторный
    запуск не меняет файлы. Скилы без entities в graph.db не трогаются
    (пустое поле не пишется, ручные entities не затираются).
    """
    if not GRAPH_DB_PATH.exists():
        print("No graph database (.graph/graph.db). Nothing to migrate:")
        print("frontmatter is the source of truth, rebuild via 'skills-graph build'.")
        return

    conn = sqlite3.connect(str(GRAPH_DB_PATH))
    try:
        rows = conn.execute(
            "SELECT skill_name, entity_name FROM skill_entities"
        ).fetchall()
    finally:
        conn.close()

    by_skill: dict[str, list[str]] = {}
    for skill_name, entity in rows:
        by_skill.setdefault(skill_name, []).append(entity)

    skills = find_all_skills()
    skill_names = {s["name"] for s in skills}

    migrated = unchanged = 0
    for skill in skills:
        entities = sorted(by_skill.get(skill["name"], []))
        if not entities:
            continue
        current = skill["frontmatter"].get("entities", [])
        if isinstance(current, str):
            current = [current]
        if sorted(current) == entities:
            unchanged += 1
            continue
        path = Path(skill["full_path"])
        text = path.read_text(encoding="utf-8")
        new_text = update_frontmatter(text, {"entities": entities})
        path.write_text(new_text, encoding="utf-8")
        migrated += 1
        print(f"  ✓ {skill['name']}: {len(entities)} entities")

    # Мёртвые строки graph.db: ссылки на скилы, которых больше нет на диске
    for name in sorted(set(by_skill) - skill_names):
        print(f"  SKIP {name}: skill not on disk "
              f"({len(by_skill[name])} entity links dropped)")

    print(f"\nMigrated: {migrated} skills, unchanged: {unchanged}.")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="skills-enrich",
        description="LLM-based metadata enrichment for the skills knowledge base (legacy)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    extract_p = sub.add_parser("extract", help="Extract metadata proposals via LLM")
    extract_p.add_argument("--all", action="store_true", help="Process all skills")
    extract_p.add_argument("--missing", action="store_true",
                           help="Only skills without related/category (default)")
    extract_p.add_argument("--file", type=str, help="Process specific file")
    extract_p.add_argument("--model", type=str,
                           help="LLM route override (e.g. local/qwen27b)")

    # apply
    apply_p = sub.add_parser("apply", help="Apply proposals to frontmatter")
    apply_p.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    apply_p.add_argument("--skill", type=str, help="Apply only for specific skill")

    # migrate-entities
    sub.add_parser("migrate-entities",
                   help="One-time: copy entities from graph.db to frontmatter")

    args = parser.parse_args()

    if args.command == "extract":
        if not args.all and not args.file:
            args.missing = True
        cmd_extract(args)
    elif args.command == "apply":
        cmd_apply(args)
    elif args.command == "migrate-entities":
        cmd_migrate_entities(args)


if __name__ == "__main__":
    main()
