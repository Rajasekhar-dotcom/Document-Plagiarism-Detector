"""
rolling_hash.py
---------------
Implements the Rabin-Karp polynomial rolling hash.

Mathematical formulation (from PRD):
    H_next = ((H_prev - c_old * b^(k-1)) * b + c_new)  mod M

Where:
    H_prev  = hash of the previous k-gram
    c_old   = ordinal value of the character leaving the window (leftmost)
    c_new   = ordinal value of the character entering the window (rightmost)
    b       = base (chosen as a prime ≈ alphabet size)
    k       = k-gram length
    M       = large prime modulus (prevents integer overflow & collisions)

Complexity:
    Naïve approach:  O(N * k)  — rehash every k-gram from scratch.
    Rolling hash:    O(N)      — each step does O(1) arithmetic.

Why these constants?
    b = 31   → standard for lowercase-only alphabets (fits in 5 bits)
    M = 2^61 - 1  → Mersenne prime; extremely low collision probability
                    and fast modular arithmetic on 64-bit hardware.
"""

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #

BASE: int = 31          # Polynomial base — prime, ~alphabet size
MODULUS: int = (1 << 61) - 1  # 2^61 − 1, a Mersenne prime


# --------------------------------------------------------------------------- #
#  Core functions                                                               #
# --------------------------------------------------------------------------- #

def _char_val(c: str) -> int:
    """
    Map a character to a positive integer ≥ 1.

    Using ord(c) - ord('a') + 1 keeps values in [1, 36] for [a-z0-9],
    ensuring no leading zero causes hash collisions between 'a...' strings.

    Args:
        c: Single character in [a-z0-9].

    Returns:
        Integer in [1, 36].
    """
    o = ord(c)
    if ord('a') <= o <= ord('z'):
        return o - ord('a') + 1          # 1..26
    if ord('0') <= o <= ord('9'):
        return o - ord('0') + 27         # 27..36
    # Fallback for any unexpected character
    return o % MODULUS


def compute_hashes(kgrams: list[str], k: int) -> list[int]:
    """
    Compute polynomial rolling hashes for all k-grams in O(N) time.

    Algorithm:
        1. Hash the first k-gram naively in O(k).
        2. For every subsequent k-gram, derive its hash from the previous
           one using the rolling formula — O(1) per step.

    Args:
        kgrams: Ordered list of k-gram strings (from kgrams.generate_kgrams).
        k:      Length of each k-gram (must match kgrams contents).

    Returns:
        List of integer hashes, one per k-gram, same order as input.
        Returns [] if kgrams is empty.

    Example:
        >>> hashes = compute_hashes(['hello', 'ellow', 'llowo'], k=5)
        >>> len(hashes)
        3
    """
    if not kgrams:
        return []

    # Pre-compute b^(k-1) mod M — used in every rolling step.
    high_base = pow(BASE, k - 1, MODULUS)   # Fast modular exponentiation O(log k)

    hashes: list[int] = []

    # ── Step 1: Hash the first k-gram from scratch ──────────────────────── #
    h = 0
    for ch in kgrams[0]:
        h = (h * BASE + _char_val(ch)) % MODULUS
    hashes.append(h)

    # ── Step 2: Roll the hash across the remaining k-grams ──────────────── #
    # Each iteration drops the leftmost char of the previous k-gram
    # and appends the rightmost char of the current k-gram.
    for i in range(1, len(kgrams)):
        prev_kg = kgrams[i - 1]
        curr_kg = kgrams[i]

        c_old = _char_val(prev_kg[0])          # character leaving the window
        c_new = _char_val(curr_kg[-1])         # character entering the window

        # Apply the PRD formula:
        # H_next = ((H_prev - c_old * b^(k-1)) * b + c_new)  mod M
        h = (h - c_old * high_base) % MODULUS
        h = (h * BASE + c_new) % MODULUS
        hashes.append(h)

    return hashes


def compute_hashes_from_text(text: str, k: int) -> list[int]:
    """
    Directly hash a sanitized text string without materialising k-gram strings.

    This is the most cache-friendly, highest-performance path:
    we slide a character window over the raw string and apply the
    rolling formula purely on character ordinals.

    Prefer this over compute_hashes() when performance is critical.

    Args:
        text: Sanitized string.
        k:    K-gram length.

    Returns:
        List of rolling hashes (len = len(text) - k + 1).
    """
    n = len(text)
    if n < k:
        return []

    high_base = pow(BASE, k - 1, MODULUS)

    # Bootstrap with chars[0..k-1]
    h = 0
    for i in range(k):
        h = (h * BASE + _char_val(text[i])) % MODULUS

    hashes = [h]

    # Slide the window
    for i in range(1, n - k + 1):
        c_old = _char_val(text[i - 1])
        c_new = _char_val(text[i + k - 1])
        h = (h - c_old * high_base) % MODULUS
        h = (h * BASE + c_new) % MODULUS
        hashes.append(h)

    return hashes
