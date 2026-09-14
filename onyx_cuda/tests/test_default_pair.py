"""GPU regressions for the 8B default, tokenizer extensions, and chunked context."""
import re

import pytest
import torch

from onyx_cuda.model import load_model, load_model_pair
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt
from onyx_cuda.speculative import generate_speculative
from onyx_cuda.vocabulary import build_token_byte_vocabulary

pytestmark = pytest.mark.gpu


def test_chunked_prefill_matches_single_forward_and_preserves_all_tokens():
    loaded = load_model()
    ids = loaded.tokenizer.encode('CUDA processes parallel computations. ' * 24)
    whole = prefill(loaded.model, ids, chunk_size=len(ids))
    chunked = prefill(loaded.model, ids, chunk_size=32)
    assert chunked.past_key_values.get_seq_length() == len(ids)
    assert torch.equal(chunked.token_id, whole.token_id)
    # FP16 reductions use different matrix shapes across chunks. Bound logit
    # drift as well as requiring the same greedy next token and full cache.
    torch.testing.assert_close(chunked.logits, whole.logits, rtol=1e-3, atol=5e-2)


def test_target_only_marker_ids_and_thinking_preserve_target_oracle():
    pair = load_model_pair()
    assert pair.draft.tokenizer is pair.target.tokenizer
    tokenizer = pair.target.tokenizer
    vocabulary = build_token_byte_vocabulary(tokenizer, pair.target.model.config.vocab_size)
    messages = [{'role':'user', 'content':'Reply with a short answer.'}]
    for thinking in (False, True):
        prompt = format_prompt(tokenizer, messages, enable_thinking=thinking).token_ids
        for marker in ('<think>ok</think>', '<tool_response>ok</tool_response>'):
            signatures = []
            for gamma in (0, 2):
                result = generate_speculative(pair.draft.model, pair.target.model, prompt, 32, gamma,
                    tokenizer.eos_token_id, regex=re.escape(marker), token_byte_vocabulary=vocabulary)
                assert result.finish_reason == 'stop'
                assert tokenizer.decode(result.token_ids, skip_special_tokens=True) == marker
                signatures.append(result.token_ids.copy())
                del result
            assert signatures[0] == signatures[1]
        signatures = []
        for gamma in (0, 2):
            result = generate_speculative(pair.draft.model, pair.target.model, prompt, 24, gamma,
                                          tokenizer.eos_token_id)
            signatures.append((result.token_ids.copy(), result.finish_reason))
            del result
        assert signatures[0] == signatures[1]
