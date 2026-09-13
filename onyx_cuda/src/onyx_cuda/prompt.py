"""Tokenizer chat-prompt formatting."""

from typing import NamedTuple

from transformers import PreTrainedTokenizerBase


class FormattedPrompt(NamedTuple):
    text: str
    token_ids: list[int]


def format_prompt(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]], *,
    enable_thinking: bool | None = None,
) -> FormattedPrompt:
    """Apply the model chat template and preserve its generation prompt."""
    options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **options
    )
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **options
    )
    return FormattedPrompt(text, token_ids)
