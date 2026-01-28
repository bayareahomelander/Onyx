#!/usr/bin/env python3
"""
ffi benchmark script

this script benchmarks the python-rust ffi overhead to verify that
our hybrid architecture is viable for high-frequency token generation loops.

we compare:
1. python's native `re` module with pre-compiled patterns
2. rust's `regex` crate via pyo3 with pre-compiled patterns (regexvalidator)
3. One-shot Rust regex validation (for comparison)
"""

import sys
import re
import time
from datetime import datetime
from typing import Tuple

sys.path.insert(0, ".")

import onyx
from onyx import RegexValidator, RUST_AVAILABLE


def benchmark_python_regex(pattern: str, test_string: str, iterations: int) -> Tuple[float, int]:
    """
    benchmark Python's native re module with pre-compiled pattern.
    
    returns tuple of (total_time_seconds, match_count)
    """
    compiled = re.compile(pattern)
    match_count = 0
    
    start = time.perf_counter()
    for _ in range(iterations):
        if compiled.match(test_string):
            match_count += 1
    end = time.perf_counter()
    
    return end - start, match_count


def benchmark_rust_regex(pattern: str, test_string: str, iterations: int) -> Tuple[float, int]:
    """
    benchmark Rust's regex crate via RegexValidator class.
    
    returns tuple of (total_time_seconds, match_count)
    """
    validator = RegexValidator(pattern)
    match_count = 0
    
    start = time.perf_counter()
    for _ in range(iterations):
        if validator.validate(test_string):
            match_count += 1
    end = time.perf_counter()
    
    return end - start, match_count


def benchmark_rust_oneshot(pattern: str, test_string: str, iterations: int) -> Tuple[float, int]:
    """
    benchmark Rust one-shot regex validation (compiles each time).
    
    returns tuple of (total_time_seconds, match_count)
    """
    match_count = 0
    
    start = time.perf_counter()
    for _ in range(iterations):
        if onyx.validate_regex_oneshot(test_string, pattern):
            match_count += 1
    end = time.perf_counter()
    
    return end - start, match_count


def format_metrics_log(
    iterations: int,
    python_time_us: float,
    rust_time_us: float,
    ffi_overhead_us: float,
    overhead_percent: float,
    pattern: str,
) -> str:
    """format metrics as natural language for the log file."""
    lines = [
        f"ffi benchmark run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"benchmarked {iterations:,} regex validation calls to measure ffi overhead.",
        f"test pattern: \"{pattern}\"",
        f"python native re module averaged {python_time_us:.3f} microseconds per call.",
        f"rust regexvalidator via ffi averaged {rust_time_us:.3f} microseconds per call.",
        f"ffi round-trip latency overhead is {ffi_overhead_us:.3f} microseconds ({overhead_percent:.1f}% overhead).",
    ]
    
    if ffi_overhead_us < 1.0:
        lines.append("ffi overhead is sub-microsecond, hybrid architecture viable for high-frequency token generation.")
    elif ffi_overhead_us < 5.0:
        lines.append("ffi overhead < 5 microseconds, acceptable for token generation loops.")
    elif ffi_overhead_us < 10.0:
        lines.append("ffi overhead < 10 microseconds, acceptable but may need optimization.")
    else:
        lines.append("ffi overhead > 10 microseconds")
    
    lines.append("")
    return "\n".join(lines)


def main():
    print("onyx ffi benchmark: python vs rust regex validation")
    
    # verify rust backend
    print(f"rust backend available: {RUST_AVAILABLE}")
    if not RUST_AVAILABLE:
        print("error: rust backend not available. run 'maturin develop' first.")
        return 1
    print()
    
    # test configuration
    iterations = 100_000
    
    # pattern: simple alphanumeric with boundaries (common for token validation)
    pattern = r"^A[0-9]+Z$"
    
    # test strings
    test_match = "A12345Z"      # should match
    test_no_match = "B12345Y"   # should not match
    
    print(f"configuration:")
    print(f"iterations: {iterations:,}")
    print(f"pattern: {pattern}")
    print(f"test (match): '{test_match}'")
    print(f"test (no match): '{test_no_match}'")
    print()
    
    # verify correctness first
    print("correctness verification:")
    py_compiled = re.compile(pattern)
    rust_validator = RegexValidator(pattern)
    
    py_match = bool(py_compiled.match(test_match))
    py_no_match = bool(py_compiled.match(test_no_match))
    rust_match = rust_validator.validate(test_match)
    rust_no_match = rust_validator.validate(test_no_match)
    
    print(f"python: match={py_match}, no_match={py_no_match}")
    print(f"rust: match={rust_match}, no_match={rust_no_match}")
    
    if py_match != rust_match or py_no_match != rust_no_match:
        print("error: python and rust results don't match")
        return 1
    print("results match")
    
    # warmup runs
    benchmark_python_regex(pattern, test_match, 1000)
    benchmark_rust_regex(pattern, test_match, 1000)
    print()
    
    # benchmark with matching string
    print(f"benchmarking {iterations:,} iterations (matching string)")
    
    py_time, py_matches = benchmark_python_regex(pattern, test_match, iterations)
    rust_time, rust_matches = benchmark_rust_regex(pattern, test_match, iterations)
    
    py_time_us = (py_time / iterations) * 1_000_000
    rust_time_us = (rust_time / iterations) * 1_000_000
    
    print(f"python re (compiled):")
    print(f"total time: {py_time*1000:.2f} ms")
    print(f"per call: {py_time_us:.3f} µs")
    print(f"matches: {py_matches:,}")
    print()
    
    print(f"rust regexvalidator via ffi:")
    print(f"total time: {rust_time*1000:.2f} ms")
    print(f"per call: {rust_time_us:.3f} µs")
    print(f"matches: {rust_matches:,}")
    print()
    
    # calculate overhead
    ffi_overhead_us = rust_time_us - py_time_us
    if py_time_us > 0:
        overhead_percent = (ffi_overhead_us / py_time_us) * 100
    else:
        overhead_percent = 0
    
    print(f"ffi round-trip analysis:")
    print(f"python baseline: {py_time_us:.3f} µs/call")
    print(f"rust via ffi: {rust_time_us:.3f} µs/call")
    print(f"ffi overhead: {ffi_overhead_us:.3f} µs/call")
    print(f"overhead percentage: {overhead_percent:.1f}%")
    
    # one-shot (compile each time), additional comparison
    print("one-shot rust (compiles each call)")
    oneshot_time, oneshot_matches = benchmark_rust_oneshot(pattern, test_match, iterations // 10) # fewer iterations since it's slower
    oneshot_time_us = (oneshot_time / (iterations // 10)) * 1_000_000
    print(f"per call: {oneshot_time_us:.3f} µs (includes compilation)")
    
    # evaluation
    if ffi_overhead_us < 1.0:
        print("ffi overhead is submicrosecond")
        print("hybrid architecture is viable for token generation")
    elif ffi_overhead_us < 5.0:
        print("ffi overhead is < 5 microseconds")
        print("hybrid architecture is viable for token generation")
    elif ffi_overhead_us < 10.0:
        print("ffi overhead is < 10 microseconds")
        print("optimization needed for high throughput")
    else:
        print("ffi overhead > 10 microseconds")
    
    # log metrics
    metrics_entry = format_metrics_log(
        iterations=iterations,
        python_time_us=py_time_us,
        rust_time_us=rust_time_us,
        ffi_overhead_us=ffi_overhead_us,
        overhead_percent=overhead_percent,
        pattern=pattern,
    )
    
    with open("metrics_log.txt", "a") as f:
        f.write(metrics_entry)
    
    # Context for token generation
    print("context for token generation:")
    tokens_per_second = 200
    time_per_token_us = (1 / tokens_per_second) * 1_000_000
    ffi_overhead_pct_of_token = (ffi_overhead_us / time_per_token_us) * 100
    print(f"at {tokens_per_second} tok/s, each token takes ~{time_per_token_us:.0f} µs")
    print(f"ffi overhead of {ffi_overhead_us:.3f} µs is {ffi_overhead_pct_of_token:.2f}% of token time")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
