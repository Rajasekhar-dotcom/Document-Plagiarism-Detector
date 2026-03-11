"""
winnowing.py
------------
Implements the Winnowing document fingerprinting algorithm.

Reference: Schleimer, Wilkerson & Aiken (2003) — "Winnowing: Local Algorithms
for Document Fingerprinting", SIGMOD.

Concept:
    Raw rolling hashes are dense (one per character position).  Comparing
    every hash across two documents would be O(N²) for N hashes.
    Winnowing compresses this down to a representative *fingerprint* —
    a small set of hashes that still provides reliable similarity detection.

Algorithm:
    1. Slide a window of size w over the list of hashes.
    2. In each window, record the *minimum* hash value (and its position).
    3. To avoid duplicates, a hash is only added to the fingerprint if it
       differs from the previously recorded minimum OR if the previous
       minimum has slid out of the window.
    4. The resulting set of (hash, position) pairs is the fingerprint.

Why minimum?
    The minimum in a window is a stable local representative: small changes
    to the document shift some windows but leave their minimum unchanged,
    making Winnowing robust to minor edits while still catching copied blocks.

Complexity: O(N) time using a monotonic deque for the sliding-window minimum.
"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FingerprintEntry:
    """A single selected hash and the text position it originated from."""
    hash_value: int
    position: int   # Index of the k-gram that produced this hash


def winnow(hashes: list[int], w: int = 4) -> set[FingerprintEntry]:
    """
    Apply the Winnowing algorithm to produce a document fingerprint.

    Uses a monotonic deque to maintain a sliding-window minimum in O(1)
    amortised time per step (overall O(N)).

    Args:
        hashes: Ordered list of rolling hash values (one per k-gram position).
        w:      Window size.  Larger w → smaller fingerprint (faster comparison,
                slightly reduced sensitivity to short matches).
                Typical values: 4–8.

    Returns:
        Set of FingerprintEntry objects representing the document fingerprint.
        Using a set automatically deduplicates identical hashes (important
        when the same minimum appears in many overlapping windows).

    Example:
        hashes = [77, 74, 42, 17, 98, 50, 17, 98, 8]  (w=4)
        windows:
          [77,74,42,17] → min=17 @ pos 3
          [74,42,17,98] → min=17 @ pos 3  (same, skip)
          [42,17,98,50] → min=17 @ pos 3  (same, skip)
          [17,98,50,17] → min=17 @ pos 6  (new position, add)
          [98,50,17,98] → min=17 @ pos 6  (same, skip)
          [50,17,98, 8] → min= 8 @ pos 8  (new min, add)
        fingerprint = {(17,3), (17,6), (8,8)}
    """
    if not hashes or w < 1:
        return set()

    fingerprint: set[FingerprintEntry] = set()

    # Monotonic deque stores (hash_value, position) in increasing hash order.
    # The front of the deque is always the current window minimum.
    dq: deque[tuple[int, int]] = deque()

    last_recorded_pos = -1

    for i, h in enumerate(hashes):
        # Remove elements from the back that are ≥ current hash
        # (they can never be the minimum while h is in the window).
        while dq and dq[-1][0] >= h:
            dq.pop()
        dq.append((h, i))

        # Remove elements that have slid out of the left edge of the window.
        # The window for index i covers positions [i - w + 1, i].
        while dq[0][1] < i - w + 1:
            dq.popleft()

        # We start recording only once the first full window is formed.
        if i >= w - 1:
            min_hash, min_pos = dq[0]
            # Only add if this is a new position to avoid duplicate entries
            # for the same hash found at the same position across windows.
            if min_pos != last_recorded_pos:
                fingerprint.add(FingerprintEntry(hash_value=min_hash, position=min_pos))
                last_recorded_pos = min_pos

    return fingerprint


def fingerprint_to_hash_set(fingerprint: set[FingerprintEntry]) -> set[int]:
    """
    Extract just the hash values from a fingerprint (discards positions).

    Used for Jaccard similarity computation, where positions are irrelevant.

    Args:
        fingerprint: Set of FingerprintEntry objects.

    Returns:
        Set of integer hash values.
    """
    return {entry.hash_value for entry in fingerprint}


def get_matching_positions(
    fp_a: set[FingerprintEntry],
    fp_b: set[FingerprintEntry],
) -> tuple[list[int], list[int]]:
    """
    Find positions in document A and B that correspond to shared hashes.

    Args:
        fp_a: Fingerprint of document A.
        fp_b: Fingerprint of document B.

    Returns:
        Tuple (positions_in_a, positions_in_b) — sorted lists of k-gram
        positions where matching fingerprint hashes were found.
    """
    shared_hashes = fingerprint_to_hash_set(fp_a) & fingerprint_to_hash_set(fp_b)
    pos_a = sorted(e.position for e in fp_a if e.hash_value in shared_hashes)
    pos_b = sorted(e.position for e in fp_b if e.hash_value in shared_hashes)
    return pos_a, pos_b
