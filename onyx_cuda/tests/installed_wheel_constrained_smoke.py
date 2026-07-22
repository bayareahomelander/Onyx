"""Model-free constrained generation and streaming qualification for an installed native wheel."""

from __future__ import annotations

import importlib
import json
import math
import re
import sys


LIFECYCLE_ITERATIONS = 100
VOCABULARY = (b"P", b"a", b"null", b"", b"x", b"1")
EOS_TOKEN_ID = 3


class ReferenceMask:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.calls: list[tuple[int, ...]] = []

    def apply(
        self,
        logits: tuple[float, ...],
        valid_token_ids: tuple[int, ...],
        /,
    ) -> tuple[float, ...]:
        self.calls.append(valid_token_ids)
        return tuple(
            value if token_id in valid_token_ids else -math.inf
            for token_id, value in enumerate(logits)
        )


class PieceTokenizer:
    tokenizer_id = "installed-native-piece-tokenizer"
    vocab_size = len(VOCABULARY)

    def encode(self, text: str, /) -> tuple[int, ...]:
        if text != "P":
            raise ValueError("installed smoke accepts only the fixed prompt")
        return (0,)

    def decode(self, token_ids, /) -> str:
        return b"".join(VOCABULARY[token_id] for token_id in token_ids).decode("utf-8")


def main() -> None:
    forbidden_prefixes = (
        "onyx",
        "mlx",
        "torch",
        "transformers",
        "tokenizers",
        "bitsandbytes",
    )
    loaded_before = set(sys.modules)
    onyx_cuda = importlib.import_module("onyx_cuda")
    testing = importlib.import_module("onyx_cuda.testing")

    newly_loaded = set(sys.modules) - loaded_before
    unexpected = sorted(
        module_name
        for module_name in newly_loaded
        for prefix in forbidden_prefixes
        if (module_name == prefix or module_name.startswith(f"{prefix}."))
        and not module_name.startswith("onyx_cuda")
    )
    if unexpected:
        raise AssertionError(f"normal constrained imports loaded forbidden runtimes: {unexpected}")

    regex_visible = _run_cycles(
        onyx_cuda,
        testing,
        grammar_type="regex",
        source="^a$",
        expected_content_token_id=1,
    )
    if re.fullmatch("^a$", regex_visible.decode("utf-8")) is None:
        raise AssertionError("native regex constrained output did not fully match")

    json_visible = _run_cycles(
        onyx_cuda,
        testing,
        grammar_type="json_schema",
        source='{"type":"null"}',
        expected_content_token_id=2,
    )
    if json.loads(json_visible.decode("utf-8")) is not None:
        raise AssertionError("native JSON constrained output did not produce null")

    print(
        "installed constrained-generation smoke passed: "
        f"abi={onyx_cuda.NATIVE_GRAMMAR_ABI_VERSION} "
        f"cycles={LIFECYCLE_ITERATIONS * 2}"
    )


def _run_cycles(
    onyx_cuda,
    testing,
    *,
    grammar_type: str,
    source: str,
    expected_content_token_id: int,
) -> bytes:
    final_visible = b""
    for _ in range(LIFECYCLE_ITERATIONS):
        timing_session = testing.create_deterministic_grammar_timing_session()
        with timing_session.compilation():
            if grammar_type == "regex":
                constraint = onyx_cuda.compile_native_regex(VOCABULARY, source)
            else:
                constraint = onyx_cuda.compile_native_json_schema(VOCABULARY, source)
        mask = ReferenceMask(len(VOCABULARY))
        backend = testing.FakeAutoregressiveBackend(
            (
                (99.0, 5.0, 4.0, -1.0, 98.0, 97.0),
                (99.0, 98.0, 97.0, 5.0, 96.0, 95.0),
            )
        )
        result = onyx_cuda.generate_constrained_target(
            backend,
            (0,),
            max_new_tokens=2,
            select_token=onyx_cuda.select_highest_logit,
            grammar_context=onyx_cuda.GrammarGenerationContext(
                constraint=constraint,
                logit_mask=mask,
                eos_token_id=EOS_TOKEN_ID,
                timing_session=timing_session,
            ),
            metrics_session=testing.create_deterministic_metrics_session(),
        )

        if result.finish_reason != "grammar_complete":
            raise AssertionError(f"unexpected finish reason {result.finish_reason!r}")
        if result.token_ids != (expected_content_token_id, EOS_TOKEN_ID):
            raise AssertionError(f"unexpected sampled token IDs {result.token_ids!r}")
        if result.visible_token_ids != (expected_content_token_id,):
            raise AssertionError(f"unexpected visible token IDs {result.visible_token_ids!r}")
        if mask.calls != [(expected_content_token_id,), (EOS_TOKEN_ID,)]:
            raise AssertionError(f"native valid tuples did not drive masking: {mask.calls!r}")
        if backend.cache_length != 2:
            raise AssertionError("terminal EOS was unexpectedly decoded into the backend cache")
        _assert_grammar_timing(onyx_cuda, result)
        final_visible = b"".join(VOCABULARY[token_id] for token_id in result.visible_token_ids)

        created_constraints = []

        def create_context(grammar):
            request_timing = testing.create_deterministic_grammar_timing_session()
            with request_timing.compilation():
                if grammar_type == "regex":
                    fresh_constraint = onyx_cuda.compile_native_regex(
                        VOCABULARY,
                        grammar.pattern,
                    )
                else:
                    fresh_constraint = onyx_cuda.compile_native_json_schema(
                        VOCABULARY,
                        grammar.schema,
                    )
            created_constraints.append(fresh_constraint)
            return onyx_cuda.GrammarGenerationContext(
                constraint=fresh_constraint,
                logit_mask=ReferenceMask(len(VOCABULARY)),
                eos_token_id=EOS_TOKEN_ID,
                timing_session=request_timing,
            )

        grammar = (
            onyx_cuda.RegexGrammar(source)
            if grammar_type == "regex"
            else onyx_cuda.JsonSchemaGrammar(source)
        )
        engine = onyx_cuda.TargetTextEngine(
            backend,
            PieceTokenizer(),
            select_token=onyx_cuda.select_highest_logit,
            create_metrics_session=testing.create_deterministic_metrics_session,
            create_grammar_context=create_context,
        )
        expected = engine.generate_constrained(
            "P",
            grammar=grammar,
            max_new_tokens=2,
        )
        _assert_grammar_timing(onyx_cuda, expected.generation)
        _assert_completed_stream(
            onyx_cuda,
            engine.stream_constrained(
                "P",
                grammar=grammar,
                max_new_tokens=2,
            ),
            expected,
        )

        partial = engine.stream_constrained(
            "P",
            grammar=grammar,
            max_new_tokens=2,
        )
        first_event = next(partial)
        if not isinstance(first_event, onyx_cuda.TextGenerationDelta):
            raise AssertionError("partial constrained stream did not yield a text delta")
        partial.close()
        if backend.cache_length != 0:
            raise AssertionError("cancelled constrained stream retained backend cache state")

        _assert_completed_stream(
            onyx_cuda,
            engine.stream_constrained(
                "P",
                grammar=grammar,
                max_new_tokens=2,
            ),
            expected,
        )
        if len(created_constraints) != 4 or len({id(item) for item in created_constraints}) != 4:
            raise AssertionError("constrained calls did not compile four fresh native constraints")
    return final_visible


def _assert_completed_stream(onyx_cuda, stream, expected) -> None:
    events = list(stream)
    deltas = [event for event in events if isinstance(event, onyx_cuda.TextGenerationDelta)]
    completions = [
        event for event in events if isinstance(event, onyx_cuda.TextGenerationComplete)
    ]
    if len(completions) != 1 or events[-1] is not completions[0]:
        raise AssertionError("constrained stream did not end with exactly one completion")
    if any(not event.text for event in deltas):
        raise AssertionError("constrained stream emitted an empty text delta")
    if "".join(event.text for event in deltas) != expected.text:
        raise AssertionError("constrained stream deltas did not reproduce completed text")
    if completions[0].result != expected:
        raise AssertionError("constrained stream terminal result differed from non-streaming")
    if expected.generation.grammar_completion_token_id != EOS_TOKEN_ID:
        raise AssertionError("constrained stream did not retain completion EOS metadata")
    _assert_grammar_timing(onyx_cuda, completions[0].result.generation)
    if "".join(event.text for event in deltas).endswith("!"):
        raise AssertionError("constrained stream exposed completion EOS text")


def _assert_grammar_timing(onyx_cuda, generation) -> None:
    timing = generation.metrics.grammar_timing
    expected = onyx_cuda.GrammarTimingMetrics(
        compilation_time=1.0,
        state_scan_time=2.0,
        valid_index_transfer_time=0.0,
        mask_application_time=2.0,
    )
    if timing != expected:
        raise AssertionError(f"unexpected constrained grammar timing {timing!r}")


if __name__ == "__main__":
    main()
