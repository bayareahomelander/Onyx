"""OpenAI-shaped request and response models plus the HTTP application factory."""

import asyncio
import gc
import json
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from decimal import Decimal
from time import time
from typing import Annotated, Any, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from onyx_cuda.config import initialize_greedy_backend, resolve_greedy_backend

MODEL_ID = "onyx-speculative"
SERVICE_VERSION = "0.1.0"
GAMMA = 0
# Conservative single-device API envelope. Programmatic generators are separate.
MAX_OUTPUT_TOKENS = 1024
MAX_CONTEXT_TOKENS = 2048
MAX_CHOICES = 4
MAX_ACTIVE_REQUESTS = 8
STREAM_BUFFER_CHUNKS = 64


class RequestCapacityMiddleware:
    """Count requests through completion, including streaming and cancellation."""

    def __init__(self, app, *, capacity: int):
        self.app = app
        self.capacity = capacity
        self.active = 0

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") != "/v1/chat/completions"):
            return await self.app(scope, receive, send)
        # No await between checking and incrementing: admission is atomic on
        # the application's event loop. Each process owns its own model/counter.
        if self.active >= self.capacity:
            response = JSONResponse(
                status_code=429, content={"detail": "Generation capacity reached; retry later"},
                headers={"Retry-After": "1"},
            )
            return await response(scope, receive, send)
        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str


class TextResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text"]


class ResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, serialize_by_alias=True)
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: Literal[True] = True

    @field_validator("strict", mode="before")
    @classmethod
    def require_boolean(cls, value):
        if not isinstance(value, bool):
            raise ValueError("strict must be a boolean")
        return value


class SchemaResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["json_schema"]
    json_schema: ResponseSchema


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(
        default=256, ge=1, le=MAX_OUTPUT_TOKENS,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    stream: bool = False
    regex: str | None = None
    json_schema: dict[str, Any] | None = None
    response_format: (
        Annotated[TextResponseFormat | SchemaResponseFormat, Field(discriminator="type")] | None
    ) = None
    compact_json: bool = True
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=-(2**63), le=2**64 - 1)
    n: int = Field(default=1, ge=1, le=MAX_CHOICES)
    stop: list[str] | None = Field(default=None, max_length=4)

    @field_validator("stop", mode="before")
    @classmethod
    def normalize_stop(cls, value):
        return [value] if isinstance(value, str) else value

    @field_validator("stop")
    @classmethod
    def reject_empty_stops(cls, value):
        if value is not None and (not value or any(not item for item in value)):
            raise ValueError("stop must contain one to four nonempty strings")
        return value

    @property
    def effective_json_schema(self) -> dict[str, Any] | None:
        if isinstance(self.response_format, SchemaResponseFormat):
            return self.response_format.json_schema.schema_
        return self.json_schema

    @model_validator(mode="after")
    def validate_options(self):
        if sum(len(message.content) for message in self.messages) > 32768:
            raise ValueError("Message content exceeds 32768 characters")
        if self.stream and self.n != 1:
            raise ValueError("stream=true supports only n=1")
        if (
            sum(value is not None for value in (self.regex, self.json_schema, self.response_format))
            > 1
        ):
            raise ValueError("regex, json_schema, and response_format are mutually exclusive")
        if self.effective_json_schema is not None and self.stop is not None:
            raise ValueError(
                "stop is unsupported with JSON output; the schema determines completion"
            )
        if "compact_json" in self.model_fields_set and self.compact_json and self.stream:
            raise ValueError("compact_json=true is unsupported with streaming")
        return self


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


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
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


def _load_configured_engine(gamma: int = GAMMA) -> Any:
    from onyx_cuda.model import load_model_pair

    return load_model_pair(include_draft=gamma > 0)


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


def _request_prompt_token_ids(request: ChatCompletionRequest, engine) -> list[int]:
    _text, token_ids = format_request_messages(request.messages, engine.target.tokenizer)
    limit = MAX_CONTEXT_TOKENS
    for loaded in (engine.target, engine.draft):
        if loaded is not None:
            context = getattr(getattr(loaded.model, "config", None), "max_position_embeddings", None)
            if isinstance(context, int) and context > 0:
                limit = min(limit, context)
    if not token_ids or len(token_ids) + request.max_tokens > limit:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt plus max_tokens must fit within {limit} tokens (prompt: {len(token_ids)})",
        )
    return token_ids


def prepare_generation(request: ChatCompletionRequest, engine, *, gamma: int = GAMMA,
                       prompt_token_ids: list[int] | None = None) -> dict[str, Any]:
    tokenizer = engine.target.tokenizer
    if prompt_token_ids is None:
        prompt_token_ids = _request_prompt_token_ids(request, engine)
    json_schema = (
        json.dumps(request.effective_json_schema)
        if request.effective_json_schema is not None
        else None
    )
    arguments = {
        "draft_model": engine.draft.model if engine.draft is not None else None,
        "target_model": engine.target.model,
        "prompt_token_ids": prompt_token_ids,
        "max_tokens": request.max_tokens,
        "gamma": gamma,
        "eos_token_ids": tokenizer.eos_token_id,
        "stop_sequences": resolve_stop_sequences(request.stop, tokenizer),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
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


def _validate_json_response(schema: dict[str, Any], output: str) -> str:
    from onyx_cuda import _rust

    return _rust.validate_json_output(json.dumps(schema), output)


def _validate_schema_number_precision(value: Any) -> None:
    if isinstance(value, Decimal):
        if not value.is_finite() or Decimal(str(float(value))) != value:
            raise ValueError("JSON schema number loses precision in the HTTP request parser")
    elif isinstance(value, dict):
        for child in value.values():
            _validate_schema_number_precision(child)
    elif isinstance(value, list):
        for child in value:
            _validate_schema_number_precision(child)


def _generate(arguments: dict[str, Any]):
    from onyx_cuda.speculative import generate_speculative

    return generate_speculative(**arguments)


async def _off_event_loop(fn, *args, executor=None):
    worker = asyncio.get_running_loop().run_in_executor(executor, fn, *args)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Repeated cancellation must not detach live model work from its lock.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()  # Retrieve a failure even when the caller disconnected.
        raise


async def _generate_off_event_loop(arguments: dict[str, Any], executor=None):
    return await _off_event_loop(_generate, arguments, executor=executor)


def _completion_event_iter(arguments: dict[str, Any], tokenizer, stop: list[str] | None):
    from onyx_cuda.speculative import decode_speculative_events, generate_speculative_events

    return decode_speculative_events(
        generate_speculative_events(**arguments),
        tokenizer,
        stop=stop,
    )


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


def _chunk_json(
    completion_id: str,
    created: int,
    model: str,
    *,
    role: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
) -> str:
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    ).model_dump_json()


def _stream_error_payload(error: BaseException) -> str:
    if isinstance(error, ValueError):
        return json.dumps({"error": {"message": str(error), "type": "invalid_request"}})
    if isinstance(error, (OSError, RuntimeError)):
        return json.dumps(
            {
                "error": {
                    "message": "Model or CUDA service unavailable",
                    "type": "service_unavailable",
                }
            }
        )
    return json.dumps({"error": {"message": "Internal server error", "type": "server_error"}})


def _sse_events(request: ChatCompletionRequest, engine, *, gamma: int = GAMMA,
                greedy_backend: str | None = None, prompt_token_ids: list[int] | None = None):
    completion_id = f"chatcmpl-{uuid4().hex[:12]}"
    created = int(time())
    model = request.model
    events = None
    try:
        yield _sse(_chunk_json(completion_id, created, model, role="assistant"))
        arguments = prepare_generation(request, engine, gamma=gamma, prompt_token_ids=prompt_token_ids)
        arguments["greedy_backend"] = greedy_backend
        arguments["measure"] = True
        events = _completion_event_iter(arguments, engine.target.tokenizer, request.stop)
        finish_reason = None
        json_parts = []
        for event in events:
            result = getattr(event, "result", None)
            if result is not None:
                finish_reason = "stop" if result.finish_reason == "eos" else result.finish_reason
                continue
            text = getattr(event, "text", None)
            if text:
                if request.effective_json_schema is not None:
                    json_parts.append(text)
                yield _sse(_chunk_json(completion_id, created, model, content=text))
        if finish_reason is None:
            raise Exception("Generation ended without a terminal event")
        if request.effective_json_schema is not None and finish_reason != "length":
            _validate_json_response(request.effective_json_schema, "".join(json_parts))
        yield _sse(_chunk_json(completion_id, created, model, finish_reason=finish_reason))
        yield _sse("[DONE]")
    except Exception as error:
        yield _sse(_stream_error_payload(error))
        yield _sse("[DONE]")
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()


async def _stream_chat_completion(app: FastAPI, request: ChatCompletionRequest, engine,
                                  prompt_token_ids: list[int] | None = None):
    lock = app.state.engine_locks[request.model]
    await lock.acquire()
    cancelled = threading.Event()
    items: queue.Queue[str | None] = queue.Queue(maxsize=STREAM_BUFFER_CHUNKS)

    def put(chunk: str | None) -> bool:
        while not cancelled.is_set():
            try:
                items.put(chunk, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        stream = _sse_events(
            request, engine, gamma=app.state.speculative_gamma,
            greedy_backend=app.state.greedy_backend,
            prompt_token_ids=prompt_token_ids,
        )
        try:
            for chunk in stream:
                if not put(chunk):
                    break
        except Exception as error:
            put(_sse(_stream_error_payload(error)))
            put(_sse("[DONE]"))
        finally:
            stream.close()
            put(None)

    future = None
    try:
        future = app.state.inference_executor.submit(worker)
        while True:
            try:
                chunk = await asyncio.to_thread(items.get, True, 0.05)
            except queue.Empty:
                continue
            if chunk is None:
                break
            yield chunk
    finally:
        cancelled.set()
        # Keep model ownership until the producer has closed its iterator.
        while future is not None and not future.done():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue
        lock.release()


def _build_vocabulary(tokenizer, logits_vocab_size):
    from onyx_cuda.vocabulary import get_token_byte_vocabulary

    return get_token_byte_vocabulary(tokenizer, logits_vocab_size)


async def _invalid_request(_request, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


async def _service_unavailable(_request, _error: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Model or CUDA service unavailable"},
    )


async def _unexpected_error(_request, _error: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def create_app(
    *,
    engine: Any | None = None,
    load_engine: Callable[[], Any] | None = None,
    gamma: int | None = None,
    greedy_backend: str | None = None,
) -> FastAPI:
    greedy_backend = resolve_greedy_backend(greedy_backend)
    if gamma is None:
        try:
            gamma = int(os.environ.get("ONYX_SPECULATIVE_GAMMA", str(GAMMA)))
        except ValueError as error:
            raise ValueError("ONYX_SPECULATIVE_GAMMA must be a nonnegative integer") from error
    if isinstance(gamma, bool) or not isinstance(gamma, int) or gamma < 0:
        raise ValueError("gamma must be a nonnegative integer (0 disables speculation)")
    loader = load_engine
    if loader is None:
        loaded = engine
        injected = engine is not None

        def loader() -> Any:
            nonlocal loaded
            if injected:
                if loaded is None:
                    raise RuntimeError("Injected engine already consumed; use load_engine to restart the app")
                # Transfer ownership to app.state. Keeping this closure's copy
                # would retain GPU weights even after lifespan shutdown.
                result, loaded = loaded, None
                return result
            return _load_configured_engine(gamma)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_greedy_backend(greedy_backend)
        app.state.engines = {MODEL_ID: loader()}
        app.state.engine_locks = {model_id: asyncio.Lock() for model_id in app.state.engines}
        # Reuse CUDA/cuBLAS thread-local state across every generation mode.
        app.state.inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onyx-inference")
        try:
            yield
        finally:
            app.state.inference_executor.shutdown(wait=True)
            del app.state.inference_executor
            app.state.engine_locks.clear()
            app.state.engines.clear()
            _release_cuda_memory()

    app = FastAPI(
        title="Onyx CUDA API",
        description="OpenAI-shaped API for CUDA grammar-aware speculative decoding",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.state.speculative_gamma = gamma
    app.state.greedy_backend = greedy_backend
    app.add_middleware(RequestCapacityMiddleware, capacity=MAX_ACTIVE_REQUESTS)
    app.add_exception_handler(ValueError, _invalid_request)
    app.add_exception_handler(OSError, _service_unavailable)
    app.add_exception_handler(RuntimeError, _service_unavailable)
    app.add_exception_handler(Exception, _unexpected_error)

    @app.get("/")
    async def root():
        return {
            "status": "ok",
            "service": "Onyx CUDA API",
            "version": SERVICE_VERSION,
            "speculative_gamma": app.state.speculative_gamma,
            "greedy_backend": app.state.greedy_backend,
            "limits": {
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "max_context_tokens": MAX_CONTEXT_TOKENS,
                "max_choices": MAX_CHOICES,
                "max_active_requests": MAX_ACTIVE_REQUESTS,
                "stream_buffer_chunks": STREAM_BUFFER_CHUNKS,
            },
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
    async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):
        engine = get_engine(app, request.model)
        prompt_token_ids = _request_prompt_token_ids(request, engine)
        if request.regex is not None:
            from onyx_cuda import _rust

            _rust.GrammarConstraint([b"x"]).compile_regex(request.regex)
        if request.effective_json_schema is not None:
            body = json.loads(await http_request.body(), parse_float=Decimal)
            raw_schema = (
                body["response_format"]["json_schema"]["schema"]
                if isinstance(request.response_format, SchemaResponseFormat)
                else body["json_schema"]
            )
            _validate_schema_number_precision(raw_schema)
            from onyx_cuda import _rust

            _rust.validate_json_schema(json.dumps(request.effective_json_schema))
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(app, request, engine, prompt_token_ids),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        async with app.state.engine_locks[request.model]:
            tokenizer = engine.target.tokenizer
            arguments = prepare_generation(request, engine, gamma=app.state.speculative_gamma,
                                           prompt_token_ids=prompt_token_ids)
            arguments["greedy_backend"] = app.state.greedy_backend
            arguments["measure"] = True
            prompt_tokens = len(arguments["prompt_token_ids"])
            completion_tokens = 0
            choices = []
            last_timings = None

            for index in range(request.n):
                result = await _generate_off_event_loop(arguments, app.state.inference_executor)
                completion_tokens += len(result.token_ids)
                last_timings = result.timings
                output = tokenizer.decode(result.token_ids, skip_special_tokens=True)
                truncated = truncate_at_stop(output, request.stop)
                finish_reason = (
                    "stop"
                    if result.finish_reason == "eos" or truncated != output
                    else result.finish_reason
                )
                output = truncated
                if request.effective_json_schema is not None and result.finish_reason != "length":
                    compact_output = _validate_json_response(request.effective_json_schema, output)
                    if request.compact_json:
                        output = compact_output
                choices.append(
                    ChatCompletionChoice(
                        index=index,
                        message=ChatMessage(role="assistant", content=output),
                        finish_reason=finish_reason,
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
                    request.regex is not None or request.effective_json_schema is not None,
                ),
            )

    return app
