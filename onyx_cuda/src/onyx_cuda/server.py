"""OpenAI-shaped request and response models plus the HTTP application factory."""

import gc
import json
import sys
from contextlib import asynccontextmanager
from time import time
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = "onyx-speculative"
SERVICE_VERSION = "0.1.0"
GAMMA = 4


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


def format_messages_as_prompt(messages: list[ChatMessage]) -> str:
    parts = []
    for message in messages:
        if message.role == "system":
            parts.append(f"System: {message.content}")
        elif message.role == "user":
            parts.append(f"User: {message.content}")
        elif message.role == "assistant":
            parts.append(f"Assistant: {message.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _encode_text(tokenizer, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return list(token_ids)


def format_request_messages(messages: list[ChatMessage], tokenizer) -> tuple[str, list[int]]:
    chat_messages = [{"role": message.role, "content": message.content} for message in messages]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is not None:
        try:
            text = apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            token_ids = apply_chat_template(
                chat_messages, tokenize=True, add_generation_prompt=True
            )
            return text, list(token_ids)
        except TypeError:
            pass
    text = format_messages_as_prompt(messages)
    return text, _encode_text(tokenizer, text)


def resolve_stop_sequences(stop: list[str] | None, tokenizer) -> list[list[int]] | None:
    if not stop:
        return None
    sequences = []
    for sequence in stop:
        if not sequence:
            continue
        token_ids = _encode_text(tokenizer, sequence)
        if token_ids:
            sequences.append(token_ids)
    return sequences or None


def prepare_generation(request: ChatCompletionRequest, engine) -> dict[str, Any]:
    tokenizer = engine.target.tokenizer
    _text, prompt_token_ids = format_request_messages(request.messages, tokenizer)
    json_schema = json.dumps(request.json_schema) if request.json_schema is not None else None
    arguments = {
        "draft_model": engine.draft.model,
        "target_model": engine.target.model,
        "prompt_token_ids": prompt_token_ids,
        "max_tokens": request.max_tokens,
        "gamma": GAMMA,
        "eos_token_ids": tokenizer.eos_token_id,
        "stop_sequences": resolve_stop_sequences(request.stop, tokenizer),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "regex": request.regex,
        "json_schema": json_schema,
    }
    if request.regex is not None or json_schema is not None:
        arguments["token_byte_vocabulary"] = _build_vocabulary(
            tokenizer, engine.target.model.config.vocab_size
        )
    return arguments


def truncate_at_stop(text: str, stop: list[str] | None) -> str:
    positions = [text.find(sequence) for sequence in (stop or []) if sequence]
    positions = [position for position in positions if position >= 0]
    return text[: min(positions)] if positions else text


def _build_metrics(timings, grammar_constrained: bool) -> OnyxMetrics:
    if timings is None:
        return OnyxMetrics(grammar_constrained=grammar_constrained)
    return OnyxMetrics(
        tokens_per_second=timings.decode_tokens_per_second,
        acceptance_rate=timings.acceptance_rate,
        ttft_ms=timings.time_to_first_token_seconds * 1000,
        grammar_constrained=grammar_constrained,
        speculative_iterations=timings.speculative_iteration_count,
    )


def _generate(arguments: dict[str, Any]):
    from onyx_cuda.speculative import generate_speculative

    return generate_speculative(**arguments)


def _build_vocabulary(tokenizer, logits_vocab_size):
    from onyx_cuda.vocabulary import build_token_byte_vocabulary

    return build_token_byte_vocabulary(tokenizer, logits_vocab_size)


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
            "endpoints": ["/", "/v1/models", "/v1/chat/completions"],
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

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not implemented")

        engine = get_engine(app, request.model)
        tokenizer = engine.target.tokenizer
        arguments = prepare_generation(request, engine)
        arguments["measure"] = True
        prompt_tokens = len(arguments["prompt_token_ids"])
        completion_tokens = 0
        choices = []
        last_timings = None

        for index in range(request.n):
            result = _generate(arguments)
            completion_tokens += len(result.token_ids)
            last_timings = result.timings
            output = tokenizer.decode(result.token_ids, skip_special_tokens=True)
            output = truncate_at_stop(output, request.stop)
            if request.json_schema is not None and request.compact_json:
                try:
                    output = json.dumps(json.loads(output), separators=(",", ":"))
                except json.JSONDecodeError:
                    pass
            choices.append(
                ChatCompletionChoice(
                    index=index,
                    message=ChatMessage(role="assistant", content=output),
                    finish_reason=result.finish_reason,
                )
            )

        return ChatCompletionResponse(
            model=request.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            onyx_metrics=_build_metrics(
                last_timings,
                request.regex is not None or request.json_schema is not None,
            ),
        )

    return app
