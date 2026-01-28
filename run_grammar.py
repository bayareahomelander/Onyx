"""
grammar-constrained generation verification script

this script verifies that the onyxengine correctly enforces regex constraints
during text generation. it tests:
1. simple digit pattern generation
2. more complex email format generation
3. performance comparison with unconstrained baseline

the script logs metrics to metrics_log.txt for documentation.
"""

import sys
import re
import time
from datetime import datetime
from typing import Tuple

sys.path.insert(0, ".")

from onyx.engine import OnyxEngine
import onyx


def verify_pattern(text: str, pattern: str) -> Tuple[bool, str]:
    """verify that text matches a regex pattern exactly. returns tuple of (matches, description)"""
    if re.fullmatch(pattern, text):
        return True, f"'{text}' matches pattern '{pattern}'"
    else:
        return False, f"'{text}' does NOT match pattern '{pattern}'"


def format_metrics_log(
    test_name: str,
    pattern: str,
    output: str,
    metrics: dict,
    baseline_metrics: dict = None,
) -> str:
    """format metrics as natural language for the log file."""
    lines = [
        f"\n--- Grammar-Constrained Generation Test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n",
        f"Test: {test_name}",
        f"Regex pattern: \"{pattern}\"",
        f"Generated output: \"{output}\"",
        f"Tokens generated: {metrics['generated_tokens']}",
        f"Generation time: {metrics['generation_time']*1000:.2f} milliseconds",
        f"Tokens per second: {metrics['tokens_per_second']:.1f}",
    ]
    
    if metrics.get('mask_time_avg'):
        lines.append(f"Average mask time: {metrics['mask_time_avg']*1_000_000:.2f} microseconds")
        lines.append(f"Total mask time: {metrics['mask_time_total']*1000:.3f} milliseconds")
        lines.append(f"Mask calls: {metrics.get('mask_calls', 0)}")
    
    if metrics.get('grammar_compile_time'):
        lines.append(f"Grammar compile time: {metrics['grammar_compile_time']*1000:.3f} milliseconds")
    
    if baseline_metrics:
        speedup = baseline_metrics['tokens_per_second'] / metrics['tokens_per_second']
        overhead_pct = (1 - 1/speedup) * 100
        lines.append(f"Baseline tokens per second: {baseline_metrics['tokens_per_second']:.1f}")
        lines.append(f"Grammar overhead: {overhead_pct:.1f}% slowdown vs baseline")
    
    lines.append("")
    return "\n".join(lines)


def test_digit_pattern(engine: OnyxEngine) -> Tuple[bool, dict]:
    """
    test 1: generate exactly 4 digits.
    
    prompt: "the year is "
    pattern: [0-9]{4}
    """
    print("test 1: 4-digit year generation")
    
    prompt = "The year is "
    pattern = r"[0-9]{4}"
    
    print(f"prompt: '{prompt}'")
    print(f"regex constraint: {pattern}")
    
    print("generating with grammar constraint...")
    output, metrics = engine.generate(
        prompt=prompt,
        max_tokens=10,
        temperature=0.0,
        regex=pattern,
    )
    
    print(f"generated: '{output}'")
    print(f"tokens: {metrics['generated_tokens']}")
    print(f"time: {metrics['generation_time']*1000:.2f}ms")
    print(f"speed: {metrics['tokens_per_second']:.1f} tok/s")
    if metrics.get('mask_time_avg'):
        print(f"avg mask time: {metrics['mask_time_avg']*1_000_000:.2f}µs")
    
    valid, description = verify_pattern(output, pattern)
    if valid:
        print(f"pass: {description}")
    else:
        print(f"fail: {description}")
    
    return valid, metrics


def test_email_pattern(engine: OnyxEngine) -> Tuple[bool, dict]:
    """
    test 2: generate a simple email-like pattern.
    
    pattern: [a-z]+@[a-z]+\\.com
    """
    print("test 2: email pattern generation")
    
    prompt = "Contact email: "
    pattern = r"[a-z]+@[a-z]+\.com"
    
    print(f"prompt: '{prompt}'")
    print(f"regex constraint: {pattern}")
    
    print("generating with grammar constraint...")
    output, metrics = engine.generate(
        prompt=prompt,
        max_tokens=30,
        temperature=0.0,
        regex=pattern,
    )
    
    print(f"generated: '{output}'")
    print(f"tokens: {metrics['generated_tokens']}")
    print(f"time: {metrics['generation_time']*1000:.2f}ms")
    print(f"speed: {metrics['tokens_per_second']:.1f} tok/s")
    if metrics.get('mask_time_avg'):
        print(f"avg mask time: {metrics['mask_time_avg']*1_000_000:.2f}µs")
    
    valid, description = verify_pattern(output, pattern)
    if valid:
        print(f"pass: {description}")
    else:
        print(f"fail: {description}")
    
    return valid, metrics


def test_baseline_comparison(engine: OnyxEngine) -> Tuple[dict, dict]:
    """
    compare grammar-constrained vs unconstrained generation performance.
    
    runs the same generation with and without grammar constraints to measure the overhead.
    """
    print("test 3: performance comparison")
    
    prompt = "The number is "
    pattern = r"[0-9]+"
    max_tokens = 20
    
    print("running baseline (unconstrained)...")
    baseline_output, baseline_metrics = engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        regex=None,
    )
    print(f"baseline output: '{baseline_output[:50]}...'")
    print(f"baseline speed: {baseline_metrics['tokens_per_second']:.1f} tok/s")
    
    print("running with grammar constraint...")
    constrained_output, constrained_metrics = engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        regex=pattern,
    )
    print(f"constrained output: '{constrained_output}'")
    print(f"constrained speed: {constrained_metrics['tokens_per_second']:.1f} tok/s")
    
    if constrained_metrics['tokens_per_second'] > 0:
        speedup = baseline_metrics['tokens_per_second'] / constrained_metrics['tokens_per_second']
        overhead_pct = (speedup - 1) * 100
        
        print(f"performance analysis:")
        print(f"baseline: {baseline_metrics['tokens_per_second']:.1f} tok/s")
        print(f"constrained: {constrained_metrics['tokens_per_second']:.1f} tok/s")
        print(f"overhead: {overhead_pct:.1f}%")
        
        if constrained_metrics.get('mask_time_avg'):
            mask_pct = (constrained_metrics['mask_time_total'] / constrained_metrics['generation_time']) * 100
            print(f"mask time percentage: {mask_pct:.1f}% of total generation time")
    
    return baseline_metrics, constrained_metrics


def test_streaming_grammar(engine: OnyxEngine) -> bool:
    """test 4: verify streaming generation with grammar constraints."""
    print("test 4: streaming grammar-constrained generation")
    
    prompt = "ID: "
    pattern = r"[A-Z]{3}-[0-9]{4}"
    
    print(f"prompt: '{prompt}'")
    print(f"regex constraint: {pattern}")
    
    print("streaming tokens: ", end="", flush=True)
    tokens = []
    final_metrics = None
    
    for token_text, metrics in engine.stream_generate(
        prompt=prompt,
        max_tokens=20,
        temperature=0.0,
        regex=pattern,
    ):
        if metrics is not None:
            final_metrics = metrics
        else:
            print(token_text, end="", flush=True)
            tokens.append(token_text)
    
    output = "".join(tokens)
    valid, description = verify_pattern(output, pattern)
    
    if valid:
        print(f"pass: {description}")
    else:
        print(f"fail: {description}")
    
    if final_metrics:
        print(f"tokens: {final_metrics['generated_tokens']}")
        print(f"speed: {final_metrics['tokens_per_second']:.1f} tok/s")
    
    return valid


def main():
    print("onyx grammar-constrained generation verification")
    
    print(f"rust backend available: {onyx.RUST_AVAILABLE}")
    if not onyx.RUST_AVAILABLE:
        print("rust backend not available")
        return 1
    
    engine = OnyxEngine()
    print(f"model loaded in {engine.load_time:.2f}s")
    print(f"vocabulary size: {len(engine.vocab_bytes):,} tokens")
    
    results = []
    all_metrics = []
    
    valid1, metrics1 = test_digit_pattern(engine)
    results.append(("4-Digit Year", valid1))
    all_metrics.append(("4-Digit Year", r"[0-9]{4}", metrics1))
    
    valid2, metrics2 = test_email_pattern(engine)
    results.append(("Email Pattern", valid2))
    all_metrics.append(("Email Pattern", r"[a-z]+@[a-z]+\.com", metrics2))
    
    baseline_metrics, constrained_metrics = test_baseline_comparison(engine)
    
    valid4 = test_streaming_grammar(engine)
    results.append(("Streaming ID Pattern", valid4))
    
    print("summary")
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("all passed")
    else:
        print("some failed")
    
    with open("metrics_log.txt", "a") as f:
        f.write("\n--- Grammar-Constrained Generation Benchmark: " + 
                datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " ---\n\n")
        
        f.write(f"vocabulary bytes pre-calculated: {len(engine.vocab_bytes):,} tokens\n\n")
        
        for test_name, pattern, metrics in all_metrics:
            f.write(f"test: {test_name}\n")
            f.write(f"pattern: \"{pattern}\"\n")
            f.write(f"tokens generated: {metrics['generated_tokens']}\n")
            f.write(f"generation time: {metrics['generation_time']*1000:.2f} milliseconds\n")
            f.write(f"tokens per second: {metrics['tokens_per_second']:.1f}\n")
            if metrics.get('mask_time_avg'):
                f.write(f"average mask time: {metrics['mask_time_avg']*1_000_000:.2f} microseconds\n")
            f.write("\n")
        
        f.write(f"baseline (unconstrained) speed: {baseline_metrics['tokens_per_second']:.1f} tokens per second\n")
        f.write(f"constrained speed: {constrained_metrics['tokens_per_second']:.1f} tokens per second\n")
        
        if constrained_metrics['tokens_per_second'] > 0:
            overhead = (baseline_metrics['tokens_per_second'] / constrained_metrics['tokens_per_second'] - 1) * 100
            f.write(f"grammar constraint overhead: {overhead:.1f}% slowdown\n")
        
        if constrained_metrics.get('mask_time_avg'):
            f.write(f"average mask computation: {constrained_metrics['mask_time_avg']*1_000_000:.2f} microseconds per token\n")
        
        f.write("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
