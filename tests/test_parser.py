"""Regression checks for HTML text extraction."""

import unittest

from src.parser import extract_text


class ExtractTextTests(unittest.TestCase):
    def test_nested_important_tags_count_each_occurrence_once(self):
        """Nested emphasis must not inflate important-word frequencies."""
        html = (
            "<p>cat</p>"
            "<h1>cat <strong>cat</strong></h1>"
            "<p><strong>dog <b>cat</b></strong></p>"
        )

        text, important_text = extract_text(html)

        self.assertEqual(text, "cat cat cat dog cat")
        self.assertEqual(important_text, "cat cat dog cat")

    def test_unclosed_tags_preserve_text_and_importance(self):
        """Missing closing tags should not crash or discard page text."""
        html = "<p>ordinary text</p><h2>broken <strong>important text"

        text, important_text = extract_text(html)

        self.assertEqual(text, "ordinary text broken important text")
        self.assertEqual(important_text, "broken important text")
