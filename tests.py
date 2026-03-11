"""
tests.py
--------
Unit and integration tests for all modules of the plagiarism detector.

Run with:
    python tests.py
or:
    python -m pytest tests.py -v
"""

import sys
import time
import unittest
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocess import sanitize, load_and_sanitize
from src.kgrams import generate_kgrams
from src.rolling_hash import compute_hashes_from_text, compute_hashes, _char_val
from src.winnowing import winnow, fingerprint_to_hash_set, get_matching_positions
from src.similarity import jaccard_similarity, similarity_percentage
from utils import build_fingerprint, compare_documents


# ─────────────────────────────────────────────────────────────────────────────
#  1. Preprocessor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocess(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(sanitize("HELLO"), "hello")

    def test_strips_punctuation(self):
        self.assertEqual(sanitize("Hello, World!"), "helloworld")

    def test_strips_whitespace(self):
        self.assertEqual(sanitize("hello world"), "helloworld")

    def test_strips_newlines(self):
        self.assertEqual(sanitize("line1\nline2"), "line1line2")

    def test_retains_digits(self):
        self.assertEqual(sanitize("abc123"), "abc123")

    def test_empty_string(self):
        self.assertEqual(sanitize(""), "")

    def test_unicode_accents_stripped(self):
        # Accented chars normalise to base letters
        result = sanitize("café")
        self.assertIn("cafe", result)

    def test_all_punctuation(self):
        self.assertEqual(sanitize("!@#$%^&*()"), "")


# ─────────────────────────────────────────────────────────────────────────────
#  2. K-gram tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKgrams(unittest.TestCase):

    def test_basic(self):
        result = generate_kgrams("helloworld", k=5)
        self.assertEqual(result[0], "hello")
        self.assertEqual(result[-1], "world")

    def test_count(self):
        text = "abcdef"
        k = 3
        expected_count = len(text) - k + 1   # 4
        self.assertEqual(len(generate_kgrams(text, k)), expected_count)

    def test_k_equals_length(self):
        result = generate_kgrams("hello", k=5)
        self.assertEqual(result, ["hello"])

    def test_k_greater_than_length(self):
        result = generate_kgrams("hi", k=5)
        self.assertEqual(result, [])

    def test_k1(self):
        result = generate_kgrams("abc", k=1)
        self.assertEqual(result, ["a", "b", "c"])

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            generate_kgrams("hello", k=0)

    def test_contiguous(self):
        grams = generate_kgrams("abcde", k=3)
        for i in range(len(grams) - 1):
            # Consecutive k-grams overlap by k-1 chars
            self.assertEqual(grams[i][1:], grams[i + 1][:-1])


# ─────────────────────────────────────────────────────────────────────────────
#  3. Rolling hash tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingHash(unittest.TestCase):

    def test_hash_count_matches_kgrams(self):
        text = "helloworld"
        k = 5
        hashes = compute_hashes_from_text(text, k)
        self.assertEqual(len(hashes), len(text) - k + 1)

    def test_identical_kgrams_same_hash(self):
        # "abcab" — the kgrams "abc" at pos 0 and "abc" at pos ... won't both appear
        # here but two identical strings should produce identical hashes.
        text1 = "abcde"
        text2 = "abcde"
        h1 = compute_hashes_from_text(text1, k=3)
        h2 = compute_hashes_from_text(text2, k=3)
        self.assertEqual(h1, h2)

    def test_different_texts_different_hashes(self):
        h1 = compute_hashes_from_text("abcde", k=3)
        h2 = compute_hashes_from_text("xyzwq", k=3)
        # Very unlikely to collide with a Mersenne prime modulus
        self.assertNotEqual(h1, h2)

    def test_rolling_consistent_with_naive(self):
        """Rolling hash must agree with the list-based compute_hashes."""
        from src.kgrams import generate_kgrams
        text = "thequickbrownfox"
        k = 4
        kgrams = generate_kgrams(text, k)
        h_list = compute_hashes(kgrams, k)
        h_roll = compute_hashes_from_text(text, k)
        self.assertEqual(h_list, h_roll)

    def test_empty_text(self):
        self.assertEqual(compute_hashes_from_text("", k=5), [])

    def test_text_shorter_than_k(self):
        self.assertEqual(compute_hashes_from_text("hi", k=5), [])

    def test_all_hashes_positive(self):
        hashes = compute_hashes_from_text("abcdefghij", k=4)
        self.assertTrue(all(h >= 0 for h in hashes))


# ─────────────────────────────────────────────────────────────────────────────
#  4. Winnowing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWinnowing(unittest.TestCase):

    def test_returns_set(self):
        hashes = [77, 74, 42, 17, 98, 50, 17, 98, 8]
        result = winnow(hashes, w=4)
        self.assertIsInstance(result, set)

    def test_fingerprint_nonempty(self):
        hashes = [5, 3, 8, 1, 6, 2, 9, 4]
        result = winnow(hashes, w=3)
        self.assertGreater(len(result), 0)

    def test_minimum_selected(self):
        # Single window: minimum must be in fingerprint
        hashes = [10, 20, 5, 30]
        fp = winnow(hashes, w=4)
        hash_vals = fingerprint_to_hash_set(fp)
        self.assertIn(5, hash_vals)

    def test_duplicate_hashes_deduplicated(self):
        # Repeated minimum shouldn't bloat fingerprint
        hashes = [1, 1, 1, 1, 1]
        fp = winnow(hashes, w=3)
        hash_vals = fingerprint_to_hash_set(fp)
        self.assertEqual(hash_vals, {1})

    def test_empty_input(self):
        self.assertEqual(winnow([], w=4), set())

    def test_window_larger_than_hashes(self):
        # If window > hashes, we get no full windows → empty set
        result = winnow([1, 2, 3], w=10)
        self.assertEqual(result, set())

    def test_identical_texts_equal_fingerprints(self):
        text = "thequickbrownfoxjumps"
        h1 = compute_hashes_from_text(text, k=5)
        h2 = compute_hashes_from_text(text, k=5)
        fp1 = fingerprint_to_hash_set(winnow(h1, w=4))
        fp2 = fingerprint_to_hash_set(winnow(h2, w=4))
        self.assertEqual(fp1, fp2)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Similarity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilarity(unittest.TestCase):

    def _make_fp(self, hash_set):
        """Create a mock fingerprint from a plain set of integers."""
        from src.winnowing import FingerprintEntry
        return {FingerprintEntry(hash_value=h, position=i)
                for i, h in enumerate(hash_set)}

    def test_identical(self):
        fp = self._make_fp({1, 2, 3, 4})
        self.assertAlmostEqual(jaccard_similarity(fp, fp), 1.0)

    def test_no_overlap(self):
        fp_a = self._make_fp({1, 2, 3})
        fp_b = self._make_fp({4, 5, 6})
        self.assertAlmostEqual(jaccard_similarity(fp_a, fp_b), 0.0)

    def test_partial_overlap(self):
        fp_a = self._make_fp({1, 2, 3, 4})
        fp_b = self._make_fp({3, 4, 5, 6})
        # |{3,4}| / |{1,2,3,4,5,6}| = 2/6 ≈ 0.333
        score = jaccard_similarity(fp_a, fp_b)
        self.assertAlmostEqual(score, 2 / 6, places=3)

    def test_percentage_range(self):
        fp_a = self._make_fp({1, 2, 3})
        fp_b = self._make_fp({2, 3, 4})
        pct = similarity_percentage(fp_a, fp_b)
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)

    def test_both_empty(self):
        self.assertAlmostEqual(jaccard_similarity(set(), set()), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  6. Integration / performance test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.docs_dir = Path(__file__).parent / "documents"
        self.target = str(self.docs_dir / "target.txt")
        self.source1 = str(self.docs_dir / "source1.txt")
        self.source2 = str(self.docs_dir / "source2.txt")

    def test_high_similarity_detected(self):
        """target.txt is a paraphrase of source1.txt — expect significant overlap."""
        result = compare_documents(self.target, self.source1)
        self.assertGreater(result.similarity, 10.0,
                           "Paraphrased document should show >10% similarity")

    def test_low_similarity_different_topic(self):
        """target.txt (evolution) vs source2.txt (ML) should show low overlap."""
        result = compare_documents(self.target, self.source2)
        self.assertLess(result.similarity, 30.0,
                        "Unrelated documents should show <30% similarity")

    def test_self_comparison_is_100(self):
        """A document compared against itself must score 100%."""
        result = compare_documents(self.source1, self.source1)
        self.assertAlmostEqual(result.similarity, 100.0, places=1)

    def test_performance_10k_words(self):
        """
        Performance gate: fingerprinting a 10,000-word synthetic document
        must complete in under 1 second (per PRD requirement).
        """
        import tempfile, os
        # Generate ~10,000 word document (~60,000 chars)
        word = "naturalselectionevolution"
        big_text = (word + " ") * 10_000
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(big_text)
            tmp_path = f.name

        try:
            t0 = time.perf_counter()
            build_fingerprint(tmp_path, k=9, w=4)
            elapsed = time.perf_counter() - t0
            self.assertLess(
                elapsed, 1.0,
                f"Fingerprinting 10k-word doc took {elapsed:.3f}s (limit: 1.0s)"
            )
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [
        TestPreprocess,
        TestKgrams,
        TestRollingHash,
        TestWinnowing,
        TestSimilarity,
        TestIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
