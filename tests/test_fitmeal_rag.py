"""Unit tests for FitMeal RAG Chatbot components."""

import os
import unittest
from app import _chunk_text, load_index, retrieve_top_k
from agent_harness import _heuristic_parse_command


class TestFitMealRAG(unittest.TestCase):

    def test_chunk_text(self):
        sample_text = "Section 1\n\nParagraph A line 1.\nParagraph A line 2.\n\nSection 2\n\nParagraph B line 1."
        chunks = _chunk_text(sample_text)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(any("Paragraph A" in c for c in chunks))

    def test_heuristic_parse_command(self):
        cmd = "บันทึกขายข้าวอกไก่ย่างซอสเทอริยากิ 2 กล่อง กล่องละ 119"
        parsed = _heuristic_parse_command(cmd)
        self.assertEqual(parsed["tool"], "log_sale")
        self.assertEqual(parsed["args"]["qty"], 2)
        self.assertEqual(parsed["args"]["price"], 119.0)

    def test_knowledge_base_exists(self):
        kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "menu_kb.md")
        self.assertTrue(os.path.exists(kb_path))
        with open(kb_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("FitMeal", content)
        self.assertIn("Low-Carb", content)
        self.assertIn("High-Protein", content)


if __name__ == "__main__":
    unittest.main()
