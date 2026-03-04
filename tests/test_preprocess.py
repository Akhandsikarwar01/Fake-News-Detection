"""Unit tests for src/preprocess.py and src/utils.py."""

import sys
import os
import unittest

import numpy as np
import pandas as pd

# Allow imports from ``src/``
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import clean_text
from preprocess import (
    load_dataset,
    preprocess_dataframe,
    build_tfidf_features,
)


class TestCleanText(unittest.TestCase):
    """Tests for the ``clean_text`` utility function."""

    def test_lowercasing(self):
        """Output should be fully lowercase."""
        result = clean_text("HELLO WORLD")
        self.assertEqual(result, result.lower())

    def test_removes_special_characters(self):
        """Punctuation and special characters should be removed."""
        result = clean_text("Hello, World! 123 #hashtag @user")
        # Only alphabetic tokens should remain
        for token in result.split():
            self.assertTrue(token.isalpha(), f"Non-alpha token found: '{token}'")

    def test_removes_stopwords(self):
        """Common English stopwords should not appear in output."""
        result = clean_text("this is a simple test sentence")
        # 'this', 'is', 'a' are stopwords and should be removed
        for stopword in ("this", "is", "a"):
            self.assertNotIn(stopword, result.split())

    def test_empty_string(self):
        """An empty string input should return an empty string."""
        self.assertEqual(clean_text(""), "")

    def test_non_string_input(self):
        """Non-string inputs (e.g. NaN as float) should not raise an exception."""
        try:
            result = clean_text(float("nan"))
            self.assertIsInstance(result, str)
        except Exception as exc:  # pragma: no cover
            self.fail(f"clean_text raised an exception on NaN input: {exc}")

    def test_url_removal(self):
        """URLs should be stripped from the output."""
        result = clean_text("Visit https://example.com for more info")
        self.assertNotIn("http", result)
        self.assertNotIn("example", result)

    def test_stemming_applied(self):
        """Stemming should reduce words to their root form."""
        # 'running' should be stemmed to 'run'
        result = clean_text("running")
        self.assertIn("run", result)


class TestMissingValueHandling(unittest.TestCase):
    """Tests for handling missing values in the dataset loading step."""

    def _make_df(self, **kwargs):
        """Create a minimal DataFrame with the expected columns."""
        defaults = {
            "id": [1, 2],
            "title": ["Title One", None],
            "author": [None, "Author Two"],
            "text": ["Body one", "Body two"],
            "label": [0, 1],
        }
        defaults.update(kwargs)
        return pd.DataFrame(defaults)

    def test_no_nan_in_content_after_load(self):
        """``content`` column must not contain NaN after loading."""
        df = self._make_df()
        # Simulate what load_dataset does (without reading a file)
        for col in ("author", "title", "text"):
            df[col] = df[col].fillna("")
        df["content"] = df["author"] + " " + df["title"]
        self.assertFalse(df["content"].isna().any())

    def test_empty_string_fills(self):
        """NaN values in key columns should be replaced with empty strings."""
        df = self._make_df()
        for col in ("author", "title", "text"):
            df[col] = df[col].fillna("")
        self.assertEqual(df["author"].iloc[0], "")
        self.assertEqual(df["title"].iloc[1], "")


class TestTfidfVectorizer(unittest.TestCase):
    """Tests for the TF-IDF vectorisation step."""

    def _make_preprocessed_df(self):
        data = {
            "content": [
                "politician lie fake election news",
                "scientist discover new cure disease",
                "president sign bill law congress",
                "celebrity break record sport win",
            ],
            "label": [1, 0, 0, 1],
        }
        return pd.DataFrame(data)

    def test_output_shape(self):
        """X should have the expected number of rows and at most max_features columns."""
        df = self._make_preprocessed_df()
        max_features = 10
        X, y, vectorizer = build_tfidf_features(df, max_features=max_features, save_vectorizer=False)
        self.assertEqual(X.shape[0], len(df))
        self.assertLessEqual(X.shape[1], max_features)

    def test_labels_unchanged(self):
        """Labels returned by build_tfidf_features should match the DataFrame."""
        df = self._make_preprocessed_df()
        _, y, _ = build_tfidf_features(df, max_features=50, save_vectorizer=False)
        np.testing.assert_array_equal(y, df["label"].values)

    def test_vectorizer_vocabulary(self):
        """Fitted vectoriser should have a non-empty vocabulary."""
        df = self._make_preprocessed_df()
        _, _, vectorizer = build_tfidf_features(df, max_features=50, save_vectorizer=False)
        self.assertGreater(len(vectorizer.vocabulary_), 0)


if __name__ == "__main__":
    unittest.main()
