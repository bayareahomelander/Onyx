"""OpenAI-shaped request and response models plus the HTTP application factory."""

import gc
import sys
from contextlib import asynccontextmanager
from time import time
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = "onyx-speculative"
SERVICE_VERSION = "0.1.0"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
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


def _load_configured_engine() -> Any:
    from onyx_cuda.model import load_model_pair

    return load_model_pair()


def _release_cuda_memory() -> None:
    gc.collect()
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    cuda.empty_cache()


def get_engine(app: FastAPI, model: str = MODEL_ID) -> Any:
    engines = getattr(app.state, "engines", {})
    if model not in engines:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(engines)}",
        )
    return engines[model]


def create_app(
    *,
    engine: Any | None = None,
    load_engine: Callable[[], Any] | None = None,
) -> FastAPI:
    loader = load_engine
    if loader is None:
        loaded = engine

        def loader() -> Any:
            if loaded is not None:
                return loaded
            return _load_configured_engine()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engines = {MODEL_ID: loader()}
        try:
            yield
        finally:
            app.state.engines.clear()
            _release_cuda_memory()

    app = FastAPI(
        title="Onyx CUDA API",
        description="OpenAI-shaped API for CUDA grammar-aware speculative decoding",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return {
            "status": "ok",
            "service": "Onyx CUDA API",
            "version": SERVICE_VERSION,
            "endpoints": ["/", "/v1/models"],
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "onyx-cuda",
                }
                for model_id in getattr(app.state, "engines", {})
            ],
        }

    return app
