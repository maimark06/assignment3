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
