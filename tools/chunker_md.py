#!/usr/bin/env python3
"""chunker_md — нарезка markdown-документов на чанки для embedding.

Общие функции вынесены из tools/skills-search.py (фаза проектного поиска,
2026-09): трёхуровневая нарезка — заголовки (fence-aware), атомарные блоки
(code/table/list/paragraph), группировка с перекрытием — и контекстный
префикс для эмбеддинга. Поведение идентично прежней встроенной версии.

Зависимости: stdlib + опциональный `skills_lib` (канонический frontmatter-
парсер в базе знаний). В self-contained копиях в дереве проекта (без
skills_lib) используется встроенный мини-парсер frontmatter — подмножество
канонического (block scalars `>`/`|`, dash-списки, inline-комментарии).
"""

import re
from pathlib import Path

try:  # в персональной базе знаний (вне репозиториев) канонический парсер — skills_lib
    from skills_lib import parse_frontmatter as _parse_frontmatter
except ImportError:  # self-contained копия в дереве проекта — мини-парсер ниже
    _parse_frontmatter = None

CHUNK_SIZE = 1500          # ~375 tokens
DESC_PREFIX_MAX = 200      # description-префикс эмбеддинга (description[:200])


def _mini_parse_frontmatter(text: str) -> tuple:
    """Мини-парсер YAML-frontmatter для self-contained копий (без skills_lib).

    Поддерживает подмножество канонического парсера: простые `key: value`,
    block scalars `description: >` (folded) и `|` (literal), inline arrays
    `[a, b]`, dash-списки, inline-комментарии после ` #`. Полный парсер —
    `tools/skills_lib.py::parse_frontmatter` в базе знаний.
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

        if val in (">", "|"):
            block_lines = []
            i += 1
            while i < len(lines):
                bl = lines[i]
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

        if " #" in val:  # inline-комментарий (только при пробеле перед #)
            val = val.split(" #", 1)[0].rstrip()

        if val == "" and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("- "):
            items = []
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                item = lines[i].lstrip()[2:].strip().strip("'\"")
                if item:
                    items.append(item)
                i += 1
            fm[key] = items
            continue

        if val.startswith("[") and val.endswith("]"):
            fm[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        else:
            fm[key] = val.strip("'\"")
        i += 1
    return fm, body


def parse_frontmatter(text: str) -> tuple:
    """Frontmatter-парсер: канонический из skills_lib либо мини-парсер."""
    if _parse_frontmatter is not None:
        return _parse_frontmatter(text)
    return _mini_parse_frontmatter(text)


# --- Markdown chunking -------------------------------------------------------

# Block types for atomic markdown parsing
_BLOCK_CODE = "code_fence"
_BLOCK_TABLE = "table"
_BLOCK_LIST = "list"
_BLOCK_PARA = "paragraph"

# Regex for list item start (unordered or ordered)
_LIST_RE = re.compile(r'^(\s*)([-*]|\d+\.)\s')


def _parse_atomic_blocks(text: str) -> list[tuple[str, str]]:
    """Parse text into atomic blocks that should not be split.

    Returns list of (block_type, block_text) tuples.
    Block types: code_fence, table, list, paragraph.

    Code blocks (``` or ~~~) are never split.
    Tables (consecutive |... lines) are never split.
    Lists (consecutive - / * / 1. lines with continuations) are never split.
    Everything else is a paragraph (split at blank lines).
    """
    lines = text.split("\n")
    blocks = []
    i = 0
    n = len(lines)

    def _flush_paragraph(para_lines):
        t = "\n".join(para_lines).strip()
        if t:
            blocks.append((_BLOCK_PARA, t))

    para_buf = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # --- Code fence: ``` or ~~~ ---
        if stripped.startswith("```") or stripped.startswith("~~~"):
            _flush_paragraph(para_buf)
            para_buf = []
            fence_marker = stripped[:3]
            fence_lines = [line]
            i += 1
            # Collect until closing fence
            while i < n:
                fence_lines.append(lines[i])
                if lines[i].strip().startswith(fence_marker) and i > 0:
                    i += 1
                    break
                i += 1
            blocks.append((_BLOCK_CODE, "\n".join(fence_lines)))
            continue

        # --- Table: consecutive lines starting with | ---
        if stripped.startswith("|"):
            _flush_paragraph(para_buf)
            para_buf = []
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append((_BLOCK_TABLE, "\n".join(table_lines)))
            continue

        # --- List: consecutive list items with continuations ---
        if _LIST_RE.match(line):
            _flush_paragraph(para_buf)
            para_buf = []
            list_lines = []
            # Get the base indent of the first item
            while i < n:
                l = lines[i]
                ls = l.strip()
                if not ls:
                    # Blank line inside list: keep if next line is still a list item or indented
                    if (i + 1 < n and
                            (_LIST_RE.match(lines[i + 1]) or lines[i + 1].startswith("  "))):
                        list_lines.append(l)
                        i += 1
                        continue
                    else:
                        break
                if _LIST_RE.match(l) or l.startswith("  ") or l.startswith("\t"):
                    list_lines.append(l)
                    i += 1
                else:
                    break
            blocks.append((_BLOCK_LIST, "\n".join(list_lines)))
            continue

        # --- Blank line: flush paragraph ---
        if not stripped:
            _flush_paragraph(para_buf)
            para_buf = []
            i += 1
            continue

        # --- Regular text: accumulate into paragraph ---
        para_buf.append(line)
        i += 1

    _flush_paragraph(para_buf)
    return blocks


def _group_blocks(blocks: list[tuple[str, str]], max_chars: int,
                  overlap_blocks: int = 1) -> list[str]:
    """Group atomic blocks into chunks respecting max_chars.

    - Accumulates blocks until adding the next would exceed max_chars.
    - A single block > max_chars is emitted as-is (never split).
    - Overlap: last `overlap_blocks` block(s) from the previous chunk are
      prepended to the next chunk, but only if they fit: an oversized block
      is never carried over, and the carry must leave room for the incoming
      block. Invariant: every chunk is <= max_chars, except a chunk holding
      a single oversized block.
    """
    if not blocks:
        return []

    chunks = []
    current_blocks = []
    current_len = 0

    for btype, btext in blocks:
        blen = len(btext)

        # If adding this block would exceed limit and we have content, flush
        if current_len + blen + 2 > max_chars and current_blocks:
            chunks.append("\n\n".join(t for _, t in current_blocks))
            # Overlap: carry last N blocks, innermost first, while they fit
            # together with the incoming block (oversized never carries).
            carry: list[tuple[str, str]] = []
            carry_len = 0
            for ob in reversed(current_blocks[-overlap_blocks:] if overlap_blocks > 0 else []):
                if len(ob[1]) > max_chars:
                    break
                if carry_len + len(ob[1]) + 2 + blen + 2 > max_chars:
                    break
                carry.insert(0, ob)
                carry_len += len(ob[1]) + 2
            current_blocks = carry
            current_len = carry_len

        current_blocks.append((btype, btext))
        current_len += blen + 2  # +2 for \n\n separator

    if current_blocks:
        chunks.append("\n\n".join(t for _, t in current_blocks))

    return chunks


def _find_fence_ranges(text: str) -> list[tuple[int, int]]:
    """Find all fenced code block ranges (start, end) in text.

    Pairs ``` or ~~~ markers. If a fence is unclosed, it extends to EOF.
    Used to exclude heading splits inside code blocks (qmd/fidx approach).
    """
    ranges = []
    i = 0
    lines = text.split("\n")
    pos = 0
    in_fence = False
    fence_start = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if not in_fence:
                in_fence = True
                fence_start = pos
            else:
                ranges.append((fence_start, pos + len(line)))
                in_fence = False
        pos += len(line) + 1  # +1 for \n

    # Unclosed fence extends to EOF
    if in_fence:
        ranges.append((fence_start, len(text)))

    return ranges


def _is_inside_fence(pos: int, fence_ranges: list[tuple[int, int]]) -> bool:
    """Check if a character position falls inside a fenced code block."""
    for start, end in fence_ranges:
        if start <= pos <= end:
            return True
        if start > pos:
            break
    return False


def _split_by_headings_fence_aware(body: str) -> list[str]:
    """Split body by headings (#{1,3}), but skip headings inside code fences.

    Returns list of sections (each starts with a heading line, except possibly the first).
    """
    fence_ranges = _find_fence_ranges(body)
    sections = []
    last_split = 0

    for m in re.finditer(r'\n(?=#{1,3}\s)', body):
        pos = m.start()
        if not _is_inside_fence(pos, fence_ranges):
            sections.append(body[last_split:pos])
            last_split = pos + 1  # skip the \n

    sections.append(body[last_split:])
    return sections


def chunk_markdown(text: str, file_path: str, owner: str = None) -> list[dict]:
    """Split markdown into chunks by headings, then sub-split long sections
    using atomic block parsing (code fences, tables, lists preserved).

    owner — необязательный владелец документа для префикса эмбеддинга:
    имя скила (проектные скилы) или rel-путь документа (docs). По умолчанию
    (skills-search) — frontmatter name или stem файла.
    """
    frontmatter, body = parse_frontmatter(text)
    skill_name = frontmatter.get("name", Path(file_path).stem)
    if owner is not None:
        skill_name = str(owner)
    description = frontmatter.get("description", "")
    if isinstance(description, list):
        description = " ".join(description)

    # Split by headings — fence-aware (won't split inside code blocks)
    sections = _split_by_headings_fence_aware(body)
    chunks = []

    # Стек заголовков (уровень, текст): путь от верхнего раздела к текущему.
    # Каждый split даёт секцию, начинающуюся с заголовка уровня L — со стека
    # сбрасываются заголовки уровня >= L, текущий кладётся сверху. Итоговая
    # цепочка (title-chain, arXiv 2608.00824: +23.8% MRR на markdown KB)
    # идёт в префикс эмбеддинга: [skill] description > H2 > H3.
    heading_stack: list[tuple[int, str]] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading
        heading_match = re.match(r'^(#{1,3})\s+(.+)', section)
        heading = ""
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
        chain = [title for _, title in heading_stack]

        # If section is too long, split using atomic blocks
        if len(section) > CHUNK_SIZE:
            atomic_blocks = _parse_atomic_blocks(section)
            sub_chunks = _group_blocks(atomic_blocks, CHUNK_SIZE)
            for j, sub in enumerate(sub_chunks):
                chunks.append({
                    "text": sub,
                    "file": file_path,
                    "skill": skill_name,
                    "heading": heading,
                    "heading_chain": list(chain),
                    "part": j + 1 if len(sub_chunks) > 1 else 0,
                })
        else:
            chunks.append({
                "text": section,
                "file": file_path,
                "skill": skill_name,
                "heading": heading,
                "heading_chain": list(chain),
                "part": 0,
            })

    # If no chunks were created (e.g. file has no headings), use whole body
    if not chunks and body.strip():
        atomic_blocks = _parse_atomic_blocks(body.strip())
        sub_chunks = _group_blocks(atomic_blocks, CHUNK_SIZE)
        for j, sub in enumerate(sub_chunks):
            chunks.append({
                "text": sub,
                "file": file_path,
                "skill": skill_name,
                "heading": "",
                "heading_chain": [],
                "part": j + 1 if len(sub_chunks) > 1 else 0,
            })

    # Prepend context to each chunk for better embedding: skill name,
    # description and the FULL heading chain of the section (H2 > H3).
    # Префикс участвует только в эмбеддинге: stemmed_text для BM25
    # индексирует чистый текст чанка (см. cmd_index), FTS префикс не видит.
    for chunk in chunks:
        prefix = f"[{skill_name}]"
        if isinstance(description, str) and description:
            prefix += f" {description[:DESC_PREFIX_MAX]}"
        if chunk["heading_chain"]:
            prefix += f" > {' > '.join(chunk['heading_chain'])}"
        chunk["text_for_embedding"] = f"{prefix}\n\n{chunk['text']}"

    return chunks
