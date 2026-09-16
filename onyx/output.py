"""Shared completion status and incremental text handling, independent of MLX."""


def finish_reason(tokens, stop_sequences, grammar_complete, max_tokens):
    if any(sequence and tokens[-len(sequence):] == sequence for sequence in stop_sequences):
        return "stop"
    if grammar_complete:
        return "grammar_complete"
    if len(tokens) >= max_tokens:
        return "length"
    raise RuntimeError("Generation ended without a stop, completed grammar, or exhausted budget")


class TokenTextStream:
    """Decode cumulative prefixes without leaking partial stops or UTF-8 text."""

    def __init__(self, tokenizer, stop_sequences):
        self.tokenizer = tokenizer
        self.stops = stop_sequences
        self.emitted = ""

    def update(self, tokens, final=False):
        visible = tokens
        matches = [len(s) for s in self.stops if s and tokens[-len(s):] == s]
        if matches:
            visible = tokens[:-max(matches)]
        elif not final:
            retain = max((n for s in self.stops for n in range(1, len(s))
                          if tokens[-n:] == s[:n]), default=0)
            if retain:
                visible = tokens[:-retain]
        text = self.tokenizer.decode(visible) if visible else ""
        if not final and "\ufffd" in text:
            text = text[:text.index("\ufffd")]
        if not text.startswith(self.emitted):
            raise RuntimeError("Tokenizer changed already-streamed text")
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta
