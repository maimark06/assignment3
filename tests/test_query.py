"""Check query processing against the real indexing tokenizer."""

import unittest

from src.query import parse_query
from src.tokenizer import tokenize


class ParseQueryTests(unittest.TestCase):
    def test_query_normalization_and_deduplication(self):
        """Normalize required queries and preserve first-seen unique stems."""
        cases = [
            ("cristina lopes", ["cristina", "lope"]),
            ("machine learning", ["machin", "learn"]),
            ("ACM", ["acm"]),
            ("master of software engineering", ["master", "of", "softwar", "engin"]),
            ("", []),
            (" \t\n", []),
            ("!!!", []),
            ("ACM acm ACM", ["acm"]),
            ("Running dog runs DOG", ["run", "dog"]),
            ("ab12 3rd AB12", ["ab12", "3rd"]),
            ("the of and", ["the", "of", "and"]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                result = parse_query(text)
                self.assertEqual(result, expected)
                self.assertEqual(result, list(dict.fromkeys(tokenize(text))))
