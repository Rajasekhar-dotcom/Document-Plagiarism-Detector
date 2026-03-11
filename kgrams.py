"""
kgrams.py
---------
Generates contiguous k-gram substrings from a sanitized text string.

A k-gram (also called a k-shingle) is every contiguous sequence of k
characters.  They are the fundamental unit that both the rolling hash
and the Winnowing algorithm operate on.

Why characters instead of words?
  Character-level k-grams survive synonym substitution and partial word
  changes better than word-level n-grams, making them ideal for detecting
  lightly-obfuscated plagiarism.

Complexity: O(N) time, O(N) space for a text of length N.
"""

from typing import Generator


def generate_kgrams(text: str, k: int = 9) -> list[str]:
    """
    Produce all contiguous k-length substrings of *text*.

    For a text of length N, this yields N - k + 1 k-grams.
    If len(text) < k, returns an empty list.

    Args:
        text: Sanitized (lowercase, no punctuation/whitespace) string.
        k:    K-gram length. Typical values: 5–15.
              - Smaller k → more matches (higher false-positive rate).
              - Larger  k → fewer matches (misses short shared phrases).

    Returns:
        List of k-gram strings in order of their starting position.

    Example:
        >>> generate_kgrams("helloworld", k=5)
        ['hello', 'ellow', 'llowo', 'lowor', 'oworl', 'world']
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    if len(text) < k:
        return []

    # List comprehension is faster than a generator for our downstream
    # use-case (we always materialise the full list for hashing).
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def generate_kgrams_iter(text: str, k: int = 9) -> Generator[str, None, None]:
    """
    Memory-efficient generator variant of generate_kgrams.

    Useful when text is very large and you want to pipeline directly
    into the hash computation without storing all k-grams at once.

    Args:
        text: Sanitized string.
        k:    K-gram length.

    Yields:
        K-gram strings one at a time.
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    for i in range(len(text) - k + 1):
        yield text[i : i + k]
