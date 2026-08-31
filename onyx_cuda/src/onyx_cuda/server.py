"""OpenAI-shaped request and response models."""

from time import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "onyx-speculative"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    stream: bool = False
    regex: str | None = None
    json_schema: dict[str, Any] | None = None
    compact_json: bool = True
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    n: int = Field(default=1, ge=1)
    stop: list[str] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OnyxMetrics(BaseModel):
    tokens_per_second: float | None = None
    acceptance_rate: float | None = None
    ttft_ms: float | None = None
    grammar_constrained: bool | None = None
    speculative_iterations: int | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    onyx_metrics: OnyxMetrics | None = None


class ChatCompletionChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
