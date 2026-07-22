"""Pinned production model identities for measured Windows development."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QwenModelProfile:
    """Immutable Hugging Face identity for one measured Qwen model revision."""

    model_id: str
    revision: str

    def __post_init__(self) -> None:
        for field_name, value in (("model_id", self.model_id), ("revision", self.revision)):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty")

    @property
    def pinned_id(self) -> str:
        return f"{self.model_id}@{self.revision}"


DEFAULT_TARGET_PROFILE = QwenModelProfile(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)


QWEN_3B_CANDIDATE_PROFILE = QwenModelProfile(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
)
