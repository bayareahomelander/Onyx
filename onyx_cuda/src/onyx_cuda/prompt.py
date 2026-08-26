"""Tokenizer chat-prompt formatting."""

from typing import NamedTuple

from transformers import PreTrainedTokenizerBase


class FormattedPrompt(NamedTuple):
    text: str
    token_ids: list[int]


def format_prompt(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]
) -> FormattedPrompt:
    """Apply the model chat template and preserve its generation prompt."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return FormattedPrompt(text, token_ids)
