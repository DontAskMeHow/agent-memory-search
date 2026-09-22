import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault("SKILLS_ROOT", str(REPO))

from chunker_md import _find_fence_ranges, _is_inside_fence, chunk_markdown  # noqa: E402
from search_core import _is_cyrillic, fts_column_query, rrf_fuse, stem_query, stem_text  # noqa: E402
from skills_lib import parse_frontmatter  # noqa: E402


class TestStemming(unittest.TestCase):
    def test_english_stem(self):
        self.assertEqual(stem_text("running"), "run")

    def test_russian_stem_returns_text(self):
        out = stem_text("память агента")
        self.assertIsInstance(out, str)
        self.assertTrue(out)

    def test_stem_query(self):
        terms = stem_query("память агентов")
        self.assertIsInstance(terms, list)
        self.assertTrue(terms)

    def test_cyrillic_detector(self):
        self.assertTrue(_is_cyrillic("память"))
        self.assertFalse(_is_cyrillic("memory"))


class TestFtsQuery(unittest.TestCase):
    def test_query_builds_on_default_column(self):
        q = fts_column_query(["память", "агент"])
        self.assertIn("stemmed_text", q)
        self.assertIn("память", q)


class TestChunker(unittest.TestCase):
    def test_fence_ranges(self):
        text = "Заголовок\n\n```python\nx = 1\n```\n\nТекст."
        ranges = _find_fence_ranges(text)
        self.assertEqual(len(ranges), 1)
        self.assertTrue(_is_inside_fence(text.index("x = 1"), ranges))

    def test_chunk_markdown(self):
        chunks = chunk_markdown("## Intro\nКороткий текст", "demo.md")
        self.assertIsInstance(chunks, list)
        self.assertTrue(chunks)
        self.assertIsInstance(chunks[0], dict)


class TestFrontmatter(unittest.TestCase):
    def test_parse(self):
        meta, body = parse_frontmatter("---\nname: demo\n---\nBody\n")
        self.assertEqual(meta.get("name"), "demo")
        self.assertIn("Body", body)

    def test_no_frontmatter(self):
        meta, body = parse_frontmatter("Просто текст")
        self.assertEqual(meta, {})
        self.assertEqual(body, "Просто текст")


class TestRrf(unittest.TestCase):
    def test_fuse_same_chunk_preferred(self):
        fused = rrf_fuse([
            {"source": "bm25", "score": 0.8, "chunk_id": "a"},
            {"source": "vec", "score": 0.9, "chunk_id": "a"},
        ])
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["chunk_id"], "a")


if __name__ == "__main__":
    unittest.main()
