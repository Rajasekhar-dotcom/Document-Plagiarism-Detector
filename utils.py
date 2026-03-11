"""
utils.py
--------
Shared utilities: the end-to-end fingerprinting pipeline, timing helpers,
and result pretty-printing.
"""

import time
from pathlib import Path
from dataclasses import dataclass

from src.preprocess import load_and_sanitize
from src.rolling_hash import compute_hashes_from_text
from src.winnowing import winnow, get_matching_positions, FingerprintEntry
from src.similarity import similarity_percentage, interpret_score


# --------------------------------------------------------------------------- #
#  Configuration defaults (can be overridden via CLI flags)                    #
# --------------------------------------------------------------------------- #

DEFAULT_K = 9    # K-gram length — good balance for English prose
DEFAULT_W = 4    # Winnowing window size


# --------------------------------------------------------------------------- #
#  Data classes                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class ComparisonResult:
    """Holds the outcome of comparing one target against one source document."""
    target_path: str
    source_path: str
    similarity: float           # Percentage [0, 100]
    risk_label: str
    matching_positions_target: list[int]
    matching_positions_source: list[int]
    elapsed_ms: float           # Wall-clock time for this comparison

    def summary_line(self) -> str:
        return (
            f"  {Path(self.target_path).name} vs "
            f"{Path(self.source_path).name:<25} → "
            f"{self.similarity:6.2f}%  [{self.risk_label}]  "
            f"({self.elapsed_ms:.1f} ms)"
        )


# --------------------------------------------------------------------------- #
#  Pipeline                                                                     #
# --------------------------------------------------------------------------- #

def build_fingerprint(
    filepath: str,
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
) -> tuple[set[FingerprintEntry], str]:
    """
    Full pipeline: file → sanitized text → hashes → Winnowing fingerprint.

    Args:
        filepath: Path to a .txt document.
        k:        K-gram length.
        w:        Winnowing window size.

    Returns:
        Tuple of (fingerprint set, sanitized text string).
    """
    text = load_and_sanitize(filepath)
    hashes = compute_hashes_from_text(text, k)
    fingerprint = winnow(hashes, w)
    return fingerprint, text


def compare_documents(
    target_path: str,
    source_path: str,
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
) -> ComparisonResult:
    """
    Compare a target document against one source document.

    Builds fingerprints for both files, computes Jaccard similarity,
    and collects the positions of matching k-gram blocks.

    Args:
        target_path: Path to the document being checked.
        source_path: Path to the reference/source document.
        k:           K-gram length.
        w:           Winnowing window size.

    Returns:
        ComparisonResult with similarity score, risk label, and match positions.
    """
    t0 = time.perf_counter()

    fp_target, _ = build_fingerprint(target_path, k, w)
    fp_source, _ = build_fingerprint(source_path, k, w)

    score = similarity_percentage(fp_target, fp_source)
    label = interpret_score(score)
    pos_t, pos_s = get_matching_positions(fp_target, fp_source)

    elapsed = (time.perf_counter() - t0) * 1000  # ms

    return ComparisonResult(
        target_path=target_path,
        source_path=source_path,
        similarity=score,
        risk_label=label,
        matching_positions_target=pos_t,
        matching_positions_source=pos_s,
        elapsed_ms=elapsed,
    )


# --------------------------------------------------------------------------- #
#  Display helpers                                                              #
# --------------------------------------------------------------------------- #

def print_banner() -> None:
    print("\n" + "═" * 65)
    print("  Document Plagiarism Detector  |  Winnowing + Rabin-Karp")
    print("═" * 65)


def print_result(result: ComparisonResult, show_positions: bool = False) -> None:
    """Pretty-print a single ComparisonResult."""
    print(result.summary_line())
    if show_positions and result.matching_positions_target:
        # Show at most 10 positions to keep output readable
        sample_t = result.matching_positions_target[:10]
        sample_s = result.matching_positions_source[:10]
        print(f"    Matching k-gram positions (target): {sample_t}" +
              (" ..." if len(result.matching_positions_target) > 10 else ""))
        print(f"    Matching k-gram positions (source): {sample_s}" +
              (" ..." if len(result.matching_positions_source) > 10 else ""))
