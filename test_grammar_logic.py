"""
grammar logic verification script

this script verifies the dfa traversal logic in the rust grammarconstraint class.
it loads a real tokenizer vocabulary, compiles a test regex pattern, and simulates
a generation loop to ensure the vocabulary filtering works correctly.

key tests:
1. initialize grammarconstraint with the qwen tokenizer vocabulary
2. compile a regex pattern to dfa
3. simulate token-by-token generation with state tracking
4. verify that valid token filtering works at each step
5. measure mask computation time for performance metrics
"""

import sys
import time
from datetime import datetime
from typing import List, Tuple, Optional

sys.path.insert(0, ".")

import onyx
from onyx import RUST_AVAILABLE


def load_tokenizer_vocabulary(model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit") -> Tuple[List[bytes], 'mlx_lm.tokenizer_utils.TokenizerWrapper']:
    """
    load the tokenizer and extract the vocabulary as byte sequences.
    
    returns tuple of (vocabulary as list of bytes, tokenizer object)
    """
    from mlx_lm import load
    
    print(f"loading tokenizer from {model_name}")
    start = time.perf_counter()
    
    _, tokenizer = load(model_name)
    
    load_time = time.perf_counter() - start
    print(f"tokenizer loaded in {load_time:.2f}s")
    
    vocab_size = tokenizer.vocab_size
    print(f"vocab size: {vocab_size:,}")
    
    vocabulary: List[bytes] = []
    
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            token_bytes = token_str.encode('utf-8')
            vocabulary.append(token_bytes)
        except Exception:
            vocabulary.append(b"")
    
    print(f"extracted {len(vocabulary):,} token byte sequences")
    
    return vocabulary, tokenizer


def find_token_id(tokenizer, text: str) -> Optional[int]:
    """find the token id for a given text string."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def find_token_ids_containing(tokenizer, vocab_size: int, substring: str) -> List[Tuple[int, str]]:
    """find all tokens containing a given substring."""
    results = []
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            if substring in token_str:
                results.append((token_id, token_str))
        except Exception:
            pass
    return results


def verify_digit_tokens(tokenizer, vocab_size: int, valid_token_ids: List[int]) -> Tuple[int, int, List[str]]:
    """
    verify that valid tokens at the digit phase contain digit tokens.
    
    returns tuple of (digit_token_count, total_valid_count, sample_tokens)
    """
    digit_count = 0
    sample_tokens = []
    
    for token_id in valid_token_ids:
        try:
            token_str = tokenizer.decode([token_id])
            if token_str and token_str[0].isdigit():
                digit_count += 1
                if len(sample_tokens) < 20:
                    sample_tokens.append(f"'{token_str}' (id={token_id})")
        except Exception:
            pass
    
    return digit_count, len(valid_token_ids), sample_tokens


def format_metrics_log(
    vocab_size: int,
    compilation_time_ms: float,
    avg_mask_time_us: float,
    min_mask_time_us: float,
    max_mask_time_us: float,
    pattern: str,
) -> str:
    """format metrics as natural language for the log file."""
    lines = [
        f"grammar dfa benchmark run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"tested dfa-based vocabulary filtering with regex pattern: \"{pattern}\"",
        f"vocabulary size: {vocab_size:,} tokens",
        f"grammar compilation time: {compilation_time_ms:.3f} milliseconds",
        f"average mask computation time: {avg_mask_time_us:.2f} microseconds per call",
        f"min mask time: {min_mask_time_us:.2f} microseconds, max mask time: {max_mask_time_us:.2f} microseconds",
    ]
    
    tokens_per_second = 200
    time_per_token_us = (1 / tokens_per_second) * 1_000_000
    mask_overhead_pct = (avg_mask_time_us / time_per_token_us) * 100
    
    lines.append(f"at 200 tokens/second, mask computation adds {mask_overhead_pct:.2f}% overhead per token.")
    
    if avg_mask_time_us < 100:
        lines.append("mask computation < 100µs, excellent for real time generation.")
    elif avg_mask_time_us < 500:
        lines.append("mask computation < 500µs, acceptable mostly.")
    elif avg_mask_time_us < 1000:
        lines.append("mask computation < 1ms, may cause noticeable latency.")
    else:
        lines.append("mask computation > 1ms. optimization.")
    
    lines.append("")
    return "\n".join(lines)


def main():
    print("onyx grammar logic verification: dfa traversal")
    
    print(f"rust backend available: {RUST_AVAILABLE}")
    if not RUST_AVAILABLE:
        print("rust backend not available. run 'maturin develop'")
        return 1
    
    try:
        from onyx._rust import GrammarConstraint
    except ImportError as e:
        print(f"failed to import GrammarConstraint: {e}")
        return 1
    
    vocabulary, tokenizer = load_tokenizer_vocabulary()
    vocab_size = len(vocabulary)
    
    print("initializing GrammarConstraint with vocab")
    start = time.perf_counter()
    constraint = GrammarConstraint(vocabulary)
    init_time = time.perf_counter() - start
    print(f"initialization time: {init_time*1000:.2f}ms")
    print(f"GrammarConstraint: {constraint}")
    
    test_pattern = r"The year is [0-9]{4}"
    print(f"compiling regex pattern: \"{test_pattern}\"")
    start = time.perf_counter()
    constraint.compile_regex(test_pattern)
    compile_time = time.perf_counter() - start
    compile_time_ms = compile_time * 1000
    print(f"compilation time: {compile_time_ms:.3f}ms")
    
    mask_times = []
    
    print("step 1: get initial state")
    state = constraint.init_state()
    print(f"initial state ID: {state}")
    print(f"is dead state: {constraint.is_dead_state(state)}")
    print(f"is match state: {constraint.is_match_state(state)}")
    
    print("step 2: get valid tokens at initial state")
    start = time.perf_counter()
    valid_tokens = constraint.get_valid_token_ids(state)
    mask_time = (time.perf_counter() - start) * 1_000_000
    mask_times.append(mask_time)
    
    print(f"valid token count: {len(valid_tokens):,}")
    print(f"mask computation time: {mask_time:.2f}µs")
    
    the_token_id = find_token_id(tokenizer, "The")
    print(f"token 'The' ID: {the_token_id}")
    if the_token_id is not None and the_token_id in valid_tokens:
        print("'The' is in valid tokens")
    else:
        print("'The' NOT in valid tokens, checking alternatives")
        sample = [tokenizer.decode([tid]) for tid in valid_tokens[:10]]
        print(f"sample valid tokens: {sample}")
    
    print("step 3: advance state with 'The'")
    if the_token_id is not None:
        state = constraint.advance_state(state, the_token_id)
        print(f"new state after 'The': {state}")
        print(f"is dead state: {constraint.is_dead_state(state)}")
    else:
        print("could not find 'The' token")
        return 1
    
    print("step 4: get valid tokens after 'The'")
    start = time.perf_counter()
    valid_tokens = constraint.get_valid_token_ids(state)
    mask_time = (time.perf_counter() - start) * 1_000_000
    mask_times.append(mask_time)
    
    print(f"valid token count: {len(valid_tokens):,}")
    print(f"mask computation time: {mask_time:.2f}µs")
    
    year_token_id = find_token_id(tokenizer, " year")
    print(f"token ' year' ID: {year_token_id}")
    if year_token_id is not None and year_token_id in valid_tokens:
        print("' year' is in valid tokens")
    else:
        print("' year' NOT in valid tokens, checking alternatives")
        year_tokens = find_token_ids_containing(tokenizer, vocab_size, "year")
        print(f"tokens containing 'year': {year_tokens[:5]}")
    
    print("step 5: advance state with ' year'")
    if year_token_id is not None:
        state = constraint.advance_state(state, year_token_id)
        print(f"new state after ' year': {state}")
        print(f"is dead state: {constraint.is_dead_state(state)}")
    
    print("step 6: get valid tokens after 'The year'")
    start = time.perf_counter()
    valid_tokens = constraint.get_valid_token_ids(state)
    mask_time = (time.perf_counter() - start) * 1_000_000
    mask_times.append(mask_time)
    
    print(f"valid token count: {len(valid_tokens):,}")
    print(f"mask computation time: {mask_time:.2f}µs")
    
    is_token_id = find_token_id(tokenizer, " is")
    print(f"token ' is' ID: {is_token_id}")
    if is_token_id is not None and is_token_id in valid_tokens:
        print("' is' is in valid tokens")
    
    print("step 7: advance state with ' is'")
    if is_token_id is not None:
        state = constraint.advance_state(state, is_token_id)
        print(f"new state after ' is': {state}")
        print(f"is dead state: {constraint.is_dead_state(state)}")
    
    print("step 8: get valid tokens after 'The year is'")
    start = time.perf_counter()
    valid_tokens = constraint.get_valid_token_ids(state)
    mask_time = (time.perf_counter() - start) * 1_000_000
    mask_times.append(mask_time)
    
    print(f"valid token count: {len(valid_tokens):,}")
    print(f"mask computation time: {mask_time:.2f}µs")
    
    space_token_id = find_token_id(tokenizer, " ")
    print(f"token ' ' (space) ID: {space_token_id}")
    if space_token_id is not None and space_token_id in valid_tokens:
        print("' ' (space) is in valid tokens")
    
    print("step 9: advance state with ' ' (space)")
    if space_token_id is not None:
        state = constraint.advance_state(state, space_token_id)
        print(f"new state after space: {state}")
        print(f"is dead state: {constraint.is_dead_state(state)}")
    
    print("step 10: critical verification - digit token filtering")
    print("at this point, only digit tokens should be valid.")
    
    start = time.perf_counter()
    valid_tokens = constraint.get_valid_token_ids(state)
    mask_time = (time.perf_counter() - start) * 1_000_000
    mask_times.append(mask_time)
    
    print(f"valid token count: {len(valid_tokens):,}")
    print(f"mask computation time: {mask_time:.2f}µs")
    
    digit_count, total_count, sample_digits = verify_digit_tokens(
        tokenizer, vocab_size, valid_tokens
    )
    
    print(f"digit tokens in valid set: {digit_count} out of {total_count}")
    print(f"sample digit tokens: {sample_digits[:10]}")
    
    letter_a_id = find_token_id(tokenizer, "a")
    if letter_a_id is not None:
        if letter_a_id not in valid_tokens:
            print("non-digit token 'a' correctly excluded")
        else:
            print("non-digit token 'a' incorrectly in valid tokens")
    
    if digit_count > 0:
        print("passed: valid tokens at digit phase contain digit tokens")
    else:
        print("failed: no digit tokens found in valid set")
        return 1
    
    print("additional mask time measurements")
    
    for i in range(10):
        start = time.perf_counter()
        _ = constraint.get_valid_token_ids(state)
        mask_time = (time.perf_counter() - start) * 1_000_000
        mask_times.append(mask_time)
    
    avg_mask_time = sum(mask_times) / len(mask_times)
    min_mask_time = min(mask_times)
    max_mask_time = max(mask_times)
    
    print(f"Total measurements: {len(mask_times)}")
    print(f"Average mask time: {avg_mask_time:.2f}µs")
    print(f"Min mask time: {min_mask_time:.2f}µs")
    print(f"Max mask time: {max_mask_time:.2f}µs")
    
    tokens_per_second = 200
    time_per_token_us = (1 / tokens_per_second) * 1_000_000
    mask_overhead_pct = (avg_mask_time / time_per_token_us) * 100
    
    print(f"at {tokens_per_second} tok/s, each token takes ~{time_per_token_us:.0f}µs")
    print(f"mask computation of {avg_mask_time:.2f}µs adds {mask_overhead_pct:.2f}% overhead")
    
    print(f"vocab size: {vocab_size:,}")
    print(f"grammar compilation time: {compile_time_ms:.3f}ms")
    print(f"avg mask time: {avg_mask_time:.2f}µs")
    
    if avg_mask_time < 100:
        print("excellent: mask computation < 100µs")
    elif avg_mask_time < 500:
        print("good: mask computation < 500µs")
    elif avg_mask_time < 1000:
        print("acceptable: mask computation < 1ms")
    else:
        print("optimization: mask computation > 1ms")
    
    metrics_entry = format_metrics_log(
        vocab_size=vocab_size,
        compilation_time_ms=compile_time_ms,
        avg_mask_time_us=avg_mask_time,
        min_mask_time_us=min_mask_time,
        max_mask_time_us=max_mask_time,
        pattern=test_pattern,
    )
    
    with open("metrics_log.txt", "a") as f:
        f.write(metrics_entry)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
