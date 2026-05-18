"""
adaptive speculative decoding experiment

This module keeps adaptive speculation separate from the stable
SpeculativeEngine path. It is intended for benchmarking and for making the
inference-systems tradeoffs explicit: the best draft length depends on recent
acceptance rate and where time is being spent.
"""

from typing import List, Optional, Tuple
import time

import mlx.core as mx

from onyx.adaptive_controller import AdaptiveGammaConfig, AdaptiveGammaController
from onyx.speculative import SpeculativeEngine, _GrammarConstraint


class AdaptiveSpeculativeEngine(SpeculativeEngine):
    """
    Experimental adaptive-gamma variant of SpeculativeEngine.

    The inherited fixed-gamma generate() method is left untouched. Call
    generate_adaptive() to use the controller in this module.
    """

    def generate_adaptive(
        self,
        prompt: str,
        max_tokens: int = 50,
        stop_tokens: Optional[List[int]] = None,
        regex: Optional[str] = None,
        json_schema: Optional[str] = None,
        draft_grammar_aware: bool = True,
        controller_config: Optional[AdaptiveGammaConfig] = None,
    ) -> Tuple[str, dict]:
        if self.draft_model is None or self.target_model is None:
            self.load_models()

        self._reset_caches()

        if stop_tokens is None:
            stop_tokens = []
            if hasattr(self.tokenizer, "eos_token_id"):
                eos = self.tokenizer.eos_token_id
                if isinstance(eos, int):
                    stop_tokens.append(eos)
                elif isinstance(eos, list):
                    stop_tokens.extend(eos)

        prompt_tokens = self.tokenizer.encode(prompt)
        grammar_active = regex is not None or json_schema is not None
        controller = AdaptiveGammaController(controller_config or AdaptiveGammaConfig())

        metrics = {
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "acceptance_rate": 0.0,
            "speculative_iterations": 0,
            "ttft": None,
            "generation_time": 0.0,
            "tokens_per_second": 0.0,
            "cache_mode": self.cache_mode,
            "jit_compiled": self.use_compile,
            "grammar_constrained": grammar_active,
            "draft_grammar_aware": draft_grammar_aware and grammar_active,
            "mask_time_total": 0.0,
            "mask_time_avg": 0.0,
            "gamma_mode": "adaptive",
            "gamma_initial": controller.current_gamma,
            "gamma_min": controller.config.min_gamma,
            "gamma_max": controller.config.max_gamma,
            "gamma_history": [],
            "adaptive_adjustments": 0,
            "draft_time_total": 0.0,
            "verify_time_total": 0.0,
            "controller_time_total": 0.0,
        }

        grammar_constraint = None
        grammar_state = None

        if grammar_active:
            if _GrammarConstraint is None:
                raise RuntimeError("Grammar constraints require the Rust backend")

            grammar_constraint = _GrammarConstraint(self.vocab_bytes)

            compile_start = time.perf_counter()
            if json_schema is not None:
                grammar_constraint.compile_json_schema(json_schema)
            else:
                grammar_constraint.compile_regex(regex)
            metrics["grammar_compile_time"] = time.perf_counter() - compile_start

            grammar_state = grammar_constraint.init_state()

        generated_tokens = []
        generation_start = time.perf_counter()
        mask_times: List[float] = []

        input_ids = mx.array([prompt_tokens])

        draft_logits = self.draft_model(input_ids, cache=self.draft_cache)
        target_logits = self.target_model(input_ids, cache=self.target_cache)
        mx.eval(draft_logits, target_logits)

        first_token_logits = target_logits[:, -1, :]

        if grammar_constraint is not None:
            mask_start = time.perf_counter()
            valid_tokens = grammar_constraint.get_valid_token_ids(grammar_state)
            mask_times.append(time.perf_counter() - mask_start)

            if valid_tokens:
                first_token_logits = self._apply_grammar_mask(first_token_logits, valid_tokens)
            else:
                raise ValueError(f"no valid tokens at initial state. Pattern: {regex}")

        first_token = self._sample_greedy(first_token_logits)
        mx.eval(first_token)

        metrics["ttft"] = time.perf_counter() - generation_start

        token_id = first_token.item()
        generated_tokens.append(token_id)

        grammar_complete = False
        if grammar_constraint is not None:
            previous_state = grammar_state
            grammar_state = grammar_constraint.advance_state(grammar_state, token_id)
            grammar_constraint.release_state(previous_state)
            if grammar_constraint.is_match_state(grammar_state):
                grammar_complete = True

        if token_id not in stop_tokens and not grammar_complete:
            while len(generated_tokens) < max_tokens:
                metrics["speculative_iterations"] += 1

                gamma = controller.current_gamma
                metrics["gamma_history"].append(gamma)

                cache_position_before_draft = self._get_cache_size(self.draft_cache)
                target_cache_position = self._get_cache_size(self.target_cache)

                draft_tokens = []
                current_token = token_id
                draft_grammar_state = grammar_state
                draft_temp_states = []
                mask_count_before_iteration = len(mask_times)

                draft_start = time.perf_counter()
                draft_input = mx.array([[current_token]])
                draft_logits = self.draft_model(draft_input, cache=self.draft_cache)
                mx.eval(draft_logits)

                for _ in range(gamma):
                    draft_last_logits = draft_logits[:, -1, :]

                    if grammar_constraint is not None and draft_grammar_aware:
                        mask_start = time.perf_counter()
                        valid_tokens = grammar_constraint.get_valid_token_ids(draft_grammar_state)
                        mask_times.append(time.perf_counter() - mask_start)

                        if not valid_tokens:
                            break

                        draft_last_logits = self._apply_grammar_mask(draft_last_logits, valid_tokens)

                    next_draft = self._sample_greedy(draft_last_logits)
                    mx.eval(next_draft)
                    draft_token = next_draft.item()
                    draft_tokens.append(draft_token)

                    if grammar_constraint is not None and draft_grammar_aware:
                        draft_grammar_state = grammar_constraint.advance_state(
                            draft_grammar_state, draft_token
                        )
                        draft_temp_states.append(draft_grammar_state)
                        if grammar_constraint.is_match_state(draft_grammar_state):
                            break

                    if draft_token in stop_tokens:
                        break

                    draft_input = mx.array([[draft_token]])
                    draft_logits = self.draft_model(draft_input, cache=self.draft_cache)
                    mx.eval(draft_logits)

                draft_time = time.perf_counter() - draft_start
                metrics["draft_time_total"] += draft_time
                metrics["draft_tokens_proposed"] += len(draft_tokens)

                if not draft_tokens:
                    controller_start = time.perf_counter()
                    controller.observe(
                        proposed=0,
                        accepted=0,
                        draft_time=draft_time,
                        verify_time=0.0,
                        mask_time=0.0,
                    )
                    metrics["controller_time_total"] += time.perf_counter() - controller_start
                    break

                verify_sequence = [current_token] + draft_tokens
                verify_input = mx.array([verify_sequence])

                verify_start = time.perf_counter()
                target_logits = self.target_model(verify_input, cache=self.target_cache)
                mx.eval(target_logits)
                verify_time = time.perf_counter() - verify_start
                metrics["verify_time_total"] += verify_time

                accepted_count = 0
                verify_grammar_state = grammar_state
                verify_temp_states = []

                for i, draft_token in enumerate(draft_tokens):
                    target_pos_logits = target_logits[:, i : i + 1, :].squeeze(1)

                    if grammar_constraint is not None:
                        mask_start = time.perf_counter()
                        valid_tokens = grammar_constraint.get_valid_token_ids(verify_grammar_state)
                        mask_times.append(time.perf_counter() - mask_start)

                        if not valid_tokens:
                            break

                        target_pos_logits = self._apply_grammar_mask(target_pos_logits, valid_tokens)

                    target_pred = self._sample_greedy(target_pos_logits).item()

                    if draft_token == target_pred:
                        accepted_count += 1
                        generated_tokens.append(draft_token)

                        if grammar_constraint is not None:
                            verify_grammar_state = grammar_constraint.advance_state(
                                verify_grammar_state, draft_token
                            )
                            verify_temp_states.append(verify_grammar_state)
                            if grammar_constraint.is_match_state(verify_grammar_state):
                                grammar_complete = True
                                break

                        if draft_token in stop_tokens:
                            break
                    else:
                        generated_tokens.append(target_pred)

                        if grammar_constraint is not None:
                            verify_grammar_state = grammar_constraint.advance_state(
                                verify_grammar_state, target_pred
                            )
                            verify_temp_states.append(verify_grammar_state)
                            if grammar_constraint.is_match_state(verify_grammar_state):
                                grammar_complete = True
                        break

                metrics["draft_tokens_accepted"] += accepted_count

                if grammar_constraint is not None:
                    release_states = list(draft_temp_states)
                    release_states.extend(s for s in verify_temp_states if s != verify_grammar_state)
                    if grammar_state != verify_grammar_state:
                        release_states.append(grammar_state)
                    grammar_state = verify_grammar_state
                    if release_states:
                        grammar_constraint.release_states(release_states)

                tokens_added = min(accepted_count + 1, len(draft_tokens))
                if generated_tokens and generated_tokens[-1] in stop_tokens:
                    tokens_added = accepted_count + (
                        1 if accepted_count < len(draft_tokens) else 0
                    )

                valid_draft_length = cache_position_before_draft + 1 + accepted_count
                self._rollback_cache(self.draft_cache, valid_draft_length)

                valid_target_length = target_cache_position + tokens_added
                self._rollback_cache(self.target_cache, valid_target_length)

                iteration_mask_time = sum(mask_times[mask_count_before_iteration:])
                controller_start = time.perf_counter()
                controller.observe(
                    proposed=len(draft_tokens),
                    accepted=accepted_count,
                    draft_time=draft_time,
                    verify_time=verify_time,
                    mask_time=iteration_mask_time,
                )
                metrics["controller_time_total"] += time.perf_counter() - controller_start

                if generated_tokens:
                    token_id = generated_tokens[-1]

                    if token_id in stop_tokens or grammar_complete:
                        break

                if len(generated_tokens) >= max_tokens:
                    break

        generation_end = time.perf_counter()
        output_text = self.tokenizer.decode(generated_tokens)

        metrics["generated_tokens"] = len(generated_tokens)
        metrics["generation_time"] = generation_end - generation_start

        if metrics["generated_tokens"] > 0:
            metrics["tokens_per_second"] = (
                metrics["generated_tokens"] / metrics["generation_time"]
            )

        if metrics["draft_tokens_proposed"] > 0:
            metrics["acceptance_rate"] = (
                metrics["draft_tokens_accepted"] / metrics["draft_tokens_proposed"]
            ) * 100

        if mask_times:
            metrics["mask_time_total"] = sum(mask_times)
            metrics["mask_time_avg"] = sum(mask_times) / len(mask_times)
            metrics["mask_calls"] = len(mask_times)

        metrics["gamma_final"] = controller.current_gamma
        metrics["adaptive_adjustments"] = controller.adjustments

        if self.cache_mode == "paged":
            metrics["cache_stats"] = self._get_cache_stats()

        return output_text, metrics


__all__ = [
    "AdaptiveGammaConfig",
    "AdaptiveGammaController",
    "AdaptiveSpeculativeEngine",
]
