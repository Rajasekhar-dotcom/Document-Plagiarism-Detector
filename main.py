"""
main.py
-------
CLI entry point for the Document Plagiarism Detector.

Usage:
    python main.py <target.txt> <sources_dir/> [options]

Options:
    -k INT   K-gram length (default: 9)
    -w INT   Winnowing window size (default: 4)
    -v       Verbose — print matching k-gram positions
    --help   Show this help message

Examples:
    python main.py documents/target.txt documents/
    python main.py report.txt sources/ -k 7 -w 5 -v
"""

import sys
import time
import argparse
from pathlib import Path

from utils import (
    compare_documents,
    print_banner,
    print_result,
    DEFAULT_K,
    DEFAULT_W,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="plagiarism-detector",
        description="Detect structural similarity between documents using "
                    "Rabin-Karp rolling hash + Winnowing fingerprinting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "target",
        help="Path to the document being checked for plagiarism (.txt).",
    )
    parser.add_argument(
        "sources_dir",
        help="Directory containing reference/source .txt documents.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        metavar="INT",
        help=f"K-gram length (default: {DEFAULT_K}). Range 5–15 recommended.",
    )
    parser.add_argument(
        "-w",
        type=int,
        default=DEFAULT_W,
        metavar="INT",
        help=f"Winnowing window size (default: {DEFAULT_W}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print matching k-gram positions for each comparison.",
    )
    return parser.parse_args(argv)


def collect_sources(sources_dir: str, target_path: str) -> list[Path]:
    """
    Gather all .txt files in sources_dir, excluding the target itself.

    Args:
        sources_dir: Directory path to scan.
        target_path: Absolute path of the target file (to skip if present).

    Returns:
        Sorted list of Path objects for source documents.
    """
    d = Path(sources_dir)
    if not d.is_dir():
        print(f"[ERROR] Not a directory: {sources_dir}", file=sys.stderr)
        sys.exit(1)

    target_abs = Path(target_path).resolve()
    sources = sorted(
        p for p in d.glob("*.txt")
        if p.resolve() != target_abs
    )
    return sources


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    target_path = args.target
    if not Path(target_path).is_file():
        print(f"[ERROR] Target file not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    sources = collect_sources(args.sources_dir, target_path)
    if not sources:
        print(f"[ERROR] No .txt source files found in: {args.sources_dir}",
              file=sys.stderr)
        sys.exit(1)

    print_banner()
    print(f"\n  Target : {target_path}")
    print(f"  Sources: {len(sources)} file(s) in '{args.sources_dir}'")
    print(f"  Params : k={args.k}, w={args.w}\n")
    print("─" * 65)

    overall_start = time.perf_counter()

    for source in sources:
        try:
            result = compare_documents(
                target_path=target_path,
                source_path=str(source),
                k=args.k,
                w=args.w,
            )
            print_result(result, show_positions=args.verbose)
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {source.name} — {exc}")

    total_ms = (time.perf_counter() - overall_start) * 1000
    print("─" * 65)
    print(f"\n  Completed {len(sources)} comparison(s) in {total_ms:.1f} ms total.\n")


if __name__ == "__main__":
    main()
