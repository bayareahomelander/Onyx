"""Run the one-step quantized real-model CUDA logits handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from onyx_cuda.real_logits_handoff import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_PROMPT,
    DEFAULT_REGEX,
    format_real_logits_report,
    run_real_logits_handoff,
)


def print_console_safe(value: str) -> None:
    """Print text without failing on legacy Windows console encodings."""
    encoding = sys.stdout.encoding or "utf-8"
    printable = value.encode(encoding, errors="backslashreplace").decode(encoding)
    print(printable)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load one INT4 Qwen model, run one CUDA forward pass, and prove that "
            "the Rust grammar plus custom CUDA selector excludes an invalid raw argmax."
        )
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
        help="Pinned Qwen model revision; defaults to the tokenizer-validated snapshot",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Prompt for the single forward pass"
    )
    parser.add_argument("--regex", default=DEFAULT_REGEX, help="One-token regex constraint")
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
        report = run_real_logits_handoff(
            DEFAULT_MODEL_ID,
            revision=args.revision,
            prompt=args.prompt,
            regex=args.regex,
            local_files_only=args.local_files_only,
            device_index=args.device_index,
        )
    except Exception as exc:
        print_console_safe(
            f"Real-model CUDA logits handoff could not run: {type(exc).__name__}: {exc}"
        )
        return 2

    print_console_safe(format_real_logits_report(report))
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
