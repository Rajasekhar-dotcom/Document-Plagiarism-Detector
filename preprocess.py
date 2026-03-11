"""
preprocess.py
-------------
Handles file I/O and text sanitization.

Sanitization pipeline:
  1. Read raw UTF-8 text from a .txt file
  2. Convert to lowercase
  3. Strip all punctuation characters
  4. Remove all whitespace (spaces, tabs, newlines)

Result is a single continuous string of alphabetic/numeric characters,
which eliminates trivial differences caused by formatting.
"""

import re
import unicodedata
from pathlib import Path


def read_file(filepath: str) -> str:
    """
    Read a plain-text file and return its raw contents.

    Args:
        filepath: Absolute or relative path to a .txt file.

    Returns:
        Raw string content of the file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not a .txt file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {path.suffix}")
    return path.read_text(encoding="utf-8", errors="replace")


def sanitize(text: str) -> str:
    """
    Sanitize raw text into a normalized, continuous character string.

    Steps:
        1. Lowercase everything.
        2. Normalize unicode (NFD) so accented chars decompose, then
           strip combining marks — keeps only base ASCII letters/digits.
        3. Remove every remaining non-alphanumeric character
           (punctuation, whitespace, symbols).

    Args:
        text: Raw input string.

    Returns:
        Sanitized string containing only [a-z0-9] characters.

    Example:
        >>> sanitize("Hello, World! 123")
        'helloworld123'
    """
    # Step 1: lowercase
    text = text.lower()

    # Step 2: unicode normalization — decompose accented chars
    text = unicodedata.normalize("NFD", text)

    # Step 3: keep only ASCII alphanumerics
    text = re.sub(r"[^a-z0-9]", "", text)

    return text


def load_and_sanitize(filepath: str) -> str:
    """
    Convenience function: read a file then sanitize its contents.

    Args:
        filepath: Path to a .txt file.

    Returns:
        Sanitized string ready for k-gram extraction.
    """
    raw = read_file(filepath)
    return sanitize(raw)
