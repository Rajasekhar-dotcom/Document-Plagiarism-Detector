"""
similarity.py
-------------
Computes the Jaccard similarity coefficient between two document fingerprints.

Jaccard Similarity:
    J(A, B) = |A ∩ B| / |A ∪ B|

    - Returns 0.0 when the documents share no fingerprint hashes (no overlap).
    - Returns 1.0 when the fingerprints are identical (exact copy).
    - Values in between represent partial structural similarity.

Why Jaccard?
    It is symmetric (J(A,B) = J(B,A)), bounded [0,1], and naturally handles
    sets of different sizes — perfect for comparing fingerprints of documents
    of different lengths.
"""

from .winnowing import FingerprintEntry, fingerprint_to_hash_set


def jaccard_similarity(
    fp_a: set[FingerprintEntry],
    fp_b: set[FingerprintEntry],
) -> float:
    """
    Compute Jaccard similarity between two Winnowing fingerprints.

    Args:
        fp_a: Fingerprint set for document A (FingerprintEntry objects).
        fp_b: Fingerprint set for document B (FingerprintEntry objects).

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if both fingerprints are empty.

    Example:
        >>> A = {1, 2, 3, 4}
        >>> B = {3, 4, 5, 6}
        >>> jaccard_similarity(A, B)
        0.333...   # |{3,4}| / |{1,2,3,4,5,6}|
    """
    set_a = fingerprint_to_hash_set(fp_a)
    set_b = fingerprint_to_hash_set(fp_b)

    if not set_a and not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    return len(intersection) / len(union)


def similarity_percentage(
    fp_a: set[FingerprintEntry],
    fp_b: set[FingerprintEntry],
) -> float:
    """
    Return Jaccard similarity as a percentage rounded to two decimal places.

    Args:
        fp_a: Fingerprint of document A.
        fp_b: Fingerprint of document B.

    Returns:
        Float in [0.0, 100.0].
    """
    return round(jaccard_similarity(fp_a, fp_b) * 100, 2)


def interpret_score(score: float) -> str:
    """
    Map a similarity percentage to a human-readable risk label.

    Args:
        score: Similarity percentage (0–100).

    Returns:
        Risk label string.
    """
    if score >= 75:
        return "HIGH  — likely plagiarism"
    if score >= 40:
        return "MEDIUM — significant overlap, review recommended"
    if score >= 15:
        return "LOW    — minor similarities, possibly coincidental"
    return "NONE   — documents appear original"
