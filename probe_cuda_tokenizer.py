"""Run the Windows tokenizer/vocabulary compatibility probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from onyx_cuda.tokenizer_probe import DEFAULT_MODEL_ID, format_probe_report, run_tokenizer_probe


def print_console_safe(value: str) -> None:
    """Print text without failing on legacy Windows console encodings."""
    encoding = sys.stdout.encoding or "utf-8"
    printable = value.encode(encoding, errors="backslashreplace").decode(encoding)
    print(printable)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that one Hugging Face tokenizer, the Rust grammar vocabulary, "
            "and the configured CUDA logits width share the same token-ID space. "
            "This does not load model weights or run inference."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID or path")
    parser.add_argument("--revision", default=None, help="Optional model repository revision")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already present in the Hugging Face cache",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the complete machine-readable report",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_tokenizer_probe(
            args.model,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print_console_safe(
            f"Tokenizer compatibility probe could not run: {type(exc).__name__}: {exc}"
        )
        return 2

    print_console_safe(format_probe_report(report))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print_console_safe(f"  JSON report: {args.json_output}")

    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
