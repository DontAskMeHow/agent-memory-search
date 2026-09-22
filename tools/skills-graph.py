#!/usr/bin/env python3
"""
skills-graph: knowledge graph over the skills knowledge base.

Граф — полностью производный артефакт: `build` пересобирает его из frontmatter
SKILL.md (related + entities) без LLM. Source of truth — git; потерянный
.graph/graph.db восстанавливается одним `build`.

Commands:
    build                           # rebuild graph from frontmatter (related + entities)
    query ENTITY [--hops N]         # find connections to an entity
    orphans                         # entities in 3+ skills without entity-skill
    status                          # graph statistics
"""

import argparse
import sqlite3
import sys
from datetime import date

from skills_lib import (
    ENTITY_SKILL_THRESHOLD,
    GRAPH_DB_PATH,
    entity_skill_name,
    find_all_skills as _find_all_skills,
    init_graph_db as _init_db,
    is_noise_entity,
)


# --- Commands ----------------------------------------------------------------

def cmd_build(args):
    """Build graph entirely from frontmatter (related + entities).

    Граф — полностью производный артефакт (фаза 4 аудита): source of truth —
    frontmatter SKILL.md в git, пересборка детерминирована и не требует LLM.
    Каждый запуск пересоздаёт все таблицы с нуля, поэтому мёртвые строки
    удалённых скилов и entities не накапливаются.
    """
    conn = _init_db()
    today = date.today().isoformat()

    # Полная пересборка: граф производный, чистим все таблицы
    conn.executescript("""
        DELETE FROM edges;
        DELETE FROM skill_entities;
        DELETE FROM entities;
    """)

    skills = _find_all_skills()
    skill_names = {s["name"] for s in skills}

    # Add skills as entities
    for s in skills:
        fm = s["frontmatter"]
        skill_type = fm.get("category", "unknown")
        conn.execute(
            "INSERT INTO entities (name, type, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?)",
            (s["name"], f"skill:{skill_type}", today, today),
        )

    # Add tech entities + skill-entity links from frontmatter `entities`
    for s in skills:
        entities = s["frontmatter"].get("entities", [])
        if isinstance(entities, str):
            entities = [entities]
        for entity in entities:
            if not entity:
                continue
            # Имя сущности может совпадать с именем скила (напр.
            # ip-sockets-cpp-lite — и скил, и библиотека): строка уже
            # вставлена как skill-entity, tech-вставка её не перезаписывает
            conn.execute(
                "INSERT INTO entities (name, type, first_seen, last_seen) "
                "VALUES (?, NULL, ?, ?) ON CONFLICT(name) DO NOTHING",
                (entity, today, today),
            )
            conn.execute(
                "INSERT OR IGNORE INTO skill_entities (skill_name, entity_name) "
                "VALUES (?, ?)",
                (s["name"], entity),
            )

    # Add edges from frontmatter related
    for s in skills:
        related = s["frontmatter"].get("related", [])
        if isinstance(related, str):
            related = [related]
        related = [r for r in related if r]
        for target in related:
            if target in skill_names and target != s["name"]:
                conn.execute(
                    "INSERT OR IGNORE INTO edges (source, target, relation, "
                    "source_skill, created_at) VALUES (?, ?, 'related_to', ?, ?)",
                    (s["name"], target, s["name"], today),
                )

    conn.commit()

    # Stats
    n_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    n_skill_ent = conn.execute("SELECT COUNT(*) FROM skill_entities").fetchone()[0]

    print(f"Graph built: {n_entities} entities, {n_edges} edges, "
          f"{n_skill_ent} skill-entity links")
    conn.close()


def cmd_query(args):
    """Query the graph for connections to an entity."""
    conn = _init_db()
    entity = args.entity
    hops = args.hops

    # Find entity (case-insensitive partial match)
    rows = conn.execute(
        "SELECT name, type FROM entities WHERE name LIKE ? OR name LIKE ?",
        (f"%{entity}%", f"%{entity.lower()}%"),
    ).fetchall()

    if not rows:
        print(f"No entity found matching '{entity}'")
        conn.close()
        return

    print(f"Matches for '{entity}':\n")
    for name, etype in rows:
        print(f"  [{etype or '?'}] {name}")

        # Direct edges (hop 1)
        outgoing = conn.execute(
            "SELECT target, relation FROM edges WHERE source = ?", (name,)
        ).fetchall()
        incoming = conn.execute(
            "SELECT source, relation FROM edges WHERE target = ?", (name,)
        ).fetchall()

        if outgoing:
            print(f"    → outgoing ({len(outgoing)}):")
            for target, rel in outgoing[:15]:
                print(f"      → {target} ({rel})")
            if len(outgoing) > 15:
                print(f"      ... and {len(outgoing) - 15} more")

        if incoming:
            print(f"    ← incoming ({len(incoming)}):")
            for source, rel in incoming[:15]:
                print(f"      ← {source} ({rel})")
            if len(incoming) > 15:
                print(f"      ... and {len(incoming) - 15} more")

        # Skills that mention this entity
        mentions = conn.execute(
            "SELECT skill_name FROM skill_entities WHERE entity_name = ?", (name,)
        ).fetchall()
        if mentions:
            print(f"    📄 mentioned in ({len(mentions)}):")
            for (sk,) in mentions:
                print(f"      {sk}")

        # Hop 2 (if requested)
        if hops >= 2:
            hop2_targets = set()
            for target, _ in outgoing:
                h2 = conn.execute(
                    "SELECT target, relation FROM edges WHERE source = ? AND target != ?",
                    (target, name),
                ).fetchall()
                for t, r in h2:
                    hop2_targets.add((target, t, r))
            for source, _ in incoming:
                h2 = conn.execute(
                    "SELECT source, relation FROM edges WHERE target = ? AND source != ?",
                    (source, name),
                ).fetchall()
                for s, r in h2:
                    hop2_targets.add((source, s, r))

            if hop2_targets:
                print(f"    ↔ 2-hop connections ({len(hop2_targets)}):")
                for via, target, rel in sorted(hop2_targets)[:20]:
                    print(f"      {name} → {via} → {target} ({rel})")

        print()

    conn.close()


def cmd_orphans(args):
    """Find entities mentioned in 3+ skills but without entity-skill."""
    conn = _init_db()
    skills = _find_all_skills()
    skill_names = {s["name"] for s in skills}

    rows = conn.execute(
        """SELECT entity_name, COUNT(*) as cnt
           FROM skill_entities
           GROUP BY entity_name
           HAVING cnt >= ?
           ORDER BY cnt DESC""",
        (ENTITY_SKILL_THRESHOLD,),
    ).fetchall()

    if not rows:
        print("No orphan entities found.")
        conn.close()
        return

    print("Entities in 3+ skills without entity-skill:\n")
    count = 0
    for name, cnt in rows:
        if is_noise_entity(name):
            continue
        entity_skill = entity_skill_name(name)
        if entity_skill not in skill_names:
            skills_list = conn.execute(
                "SELECT skill_name FROM skill_entities WHERE entity_name = ?", (name,)
            ).fetchall()
            print(f"  {name} ({cnt} skills): {', '.join(s[0] for s in skills_list)}")
            count += 1

    if count == 0:
        print("All frequent entities have entity-skills. ✓")

    conn.close()


def cmd_status(args):
    """Show graph statistics."""
    if not GRAPH_DB_PATH.exists():
        print("No graph database. Run 'skills-graph build' first.")
        return

    conn = _init_db()
    n_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    n_skill_ent = conn.execute("SELECT COUNT(*) FROM skill_entities").fetchone()[0]

    db_size = GRAPH_DB_PATH.stat().st_size / 1024

    print(f"Graph: {GRAPH_DB_PATH}")
    print(f"Entities:          {n_entities}")
    print(f"Edges:             {n_edges}")
    print(f"Skill-entity links: {n_skill_ent}")
    print(f"DB size:           {db_size:.0f} KB")

    # Top entities by connection count
    print("\nTop entities by connections:")
    # JOIN+GROUP BY вместо коррелированного подзапроса на каждую строку
    # (фаза 1 аудита р3); ребро учитывается для каждой из своих вершин
    rows = conn.execute(
        """SELECT e.name, e.type, COALESCE(c.conns, 0) AS conns
           FROM entities e
           LEFT JOIN (
               SELECT name, COUNT(*) AS conns FROM (
                   SELECT source AS name FROM edges
                   UNION ALL
                   SELECT target AS name FROM edges
               ) GROUP BY name
           ) c ON c.name = e.name
           ORDER BY conns DESC LIMIT 15"""
    ).fetchall()
    for name, etype, conns in rows:
        if conns > 0:
            print(f"  {conns:3d} connections: {name} [{etype}]")

    # Entity type distribution
    print("\nEntity types:")
    rows = conn.execute(
        "SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC"
    ).fetchall()
    for etype, cnt in rows:
        print(f"  {cnt:3d} {etype or 'unknown'}")

    conn.close()


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="skills-graph",
        description="Knowledge graph over the skills knowledge base",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="Rebuild graph from frontmatter (related + entities)")

    query_p = sub.add_parser("query", help="Query entity connections")
    query_p.add_argument("entity", help="Entity name (partial match)")
    query_p.add_argument("--hops", type=int, choices=[1, 2], default=1,
                         help="Number of hops (1 or 2)")

    sub.add_parser("orphans", help="Find entities needing entity-skills")
    sub.add_parser("status", help="Show graph statistics")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "orphans":
        cmd_orphans(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
