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

    def test_plain_text_has_no_important_text(self):
        """Pages without HTML tags still supply searchable text."""
        plain_text = "Hello world! ab12 and 3rd are searchable."

        text, important_text = extract_text(plain_text)

        self.assertEqual(text, plain_text)
        self.assertEqual(important_text, "")

    def test_scripts_styles_and_comments_are_excluded(self):
        """Exclude code and comments even inside an important tag."""
        html = (
            "<style>hidden stylesheet</style>"
            "<script>hidden script</script>"
            "<!-- hidden comment -->"
            "<p>ordinary text</p>"
            "<strong>important text"
            "<script>hidden emphasized script</script>"
            "<style>hidden emphasized stylesheet</style>"
            "<!-- hidden emphasized comment -->"
            "</strong>"
        )

        text, important_text = extract_text(html)

        self.assertEqual(text, "ordinary text important text")
        self.assertEqual(important_text, "important text")

    def test_empty_content_returns_empty_text(self):
        """Empty pages and whitespace-only tags contain no indexable text."""
        for html in ("", " \n\t ", "<p></p>", "<h1> \n </h1><strong></strong>"):
            with self.subTest(html=html):
                text, important_text = extract_text(html)

                self.assertEqual(text, "")
                self.assertEqual(important_text, "")

    def test_each_required_tag_marks_text_as_important(self):
        """All six tags required by the assignment contribute important text."""
        for tag in ("b", "strong", "h1", "h2", "h3", "title"):
            with self.subTest(tag=tag):
                html = f"<{tag}>important words</{tag}><p>ordinary words</p>"

                text, important_text = extract_text(html)

                self.assertEqual(text, "important words ordinary words")
                self.assertEqual(important_text, "important words")
