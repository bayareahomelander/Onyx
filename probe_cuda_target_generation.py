"""Run bounded target-only CUDA constrained generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from onyx_cuda.real_logits_handoff import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from onyx_cuda.target_generation import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_REGEX,
    format_target_generation_report,
    run_target_only_generation,
)


def print_console_safe(value: str) -> None:
    """Print text without failing on legacy Windows console encodings."""
    encoding = sys.stdout.encoding or "utf-8"
    printable = value.encode(encoding, errors="backslashreplace").decode(encoding)
    print(printable)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the pinned INT4 Qwen model and run bounded regex-constrained "
            "target-only generation with the CUDA selector and KV cache."
        )
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
        help="Pinned Qwen model revision; defaults to the tokenizer-validated snapshot",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Prompt for bounded target-only generation"
    )
    parser.add_argument("--regex", default=DEFAULT_REGEX, help="Regex constraint for generation")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum constrained tokens to generate",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Optional stop string; repeat the flag for multiple stops",
    )
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device index")
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
        report = run_target_only_generation(
            DEFAULT_MODEL_ID,
            revision=args.revision,
            prompt=args.prompt,
            regex=args.regex,
            max_new_tokens=args.max_new_tokens,
            stop_strings=args.stop,
            local_files_only=args.local_files_only,
            device_index=args.device_index,
        )
    except Exception as exc:
        print_console_safe(
            f"Target-only CUDA generation could not run: {type(exc).__name__}: {exc}"
        )
        return 2

    print_console_safe(format_target_generation_report(report))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print_console_safe(f"  JSON report: {args.json_output}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
