import asyncio
import json
import time
import uuid
import os
from typing import List, Optional, Dict, Any, Generator, Literal, Annotated
from contextlib import asynccontextmanager, closing

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(
        default="onyx-speculative",
        description="Model identifier. Use 'onyx-speculative' for grammar-aware speculation."
    )
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="List of messages in the conversation"
    )
    max_tokens: int = Field(
        default=256,
        strict=True, gt=0,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.0,
        strict=True, ge=0, le=2, allow_inf_nan=False,
        description="Sampling temperature (0 = greedy, higher = more random)"
    )
    stream: bool = Field(
        default=False,
        strict=True,
        description="Whether to stream the response token by token"
    )
    regex: Optional[str] = Field(
        default=None,
        description="Regex pattern to constrain the output (Onyx extension)"
    )
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema to constrain the output (Onyx extension)"
    )
    compact_json: bool = Field(default=False, strict=True, deprecated=True,
                              description="Deprecated; only false is supported. Format JSON in the client.")
    top_p: float = Field(default=1.0, strict=True, gt=0, le=1, allow_inf_nan=False)
    n: int = Field(default=1, strict=True, ge=1, le=1, deprecated=True,
                   description="Compatibility field; only one completion is supported.")
    stop: Optional[List[Annotated[str, Field(min_length=1)]]] = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.regex is not None and self.json_schema is not None:
            raise ValueError("Specify either regex or json_schema, not both")
        if self.__dict__["compact_json"]:
            raise ValueError("compact_json=true is no longer supported; format JSON in the client")
        return self


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = Field(
        default="stop",
        description="Reason for completion (stop, length, grammar_complete)"
    )


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

    onyx_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Onyx-specific performance metrics"
    )


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


_engines: Dict[str, Any] = {}
# An engine keeps its KV caches on itself, so one request must own it at a time.
# Requests wait on the event loop: a waiter blocking a worker thread could starve
# the owning stream of the threads it needs to produce its next chunk.
_engine_locks: Dict[str, asyncio.Lock] = {}


def get_engine(model: str = "onyx-speculative"):
    if model not in _engines:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(_engines.keys())}",
        )
    return _engines[model]


def engine_lock(model: str) -> asyncio.Lock:
    return _engine_locks.setdefault(model, asyncio.Lock())


def ensure_loaded(engine) -> None:
    # Called while owning the engine, so a lazy engine loads exactly once.
    if engine.draft_model is None:
        engine.load_models()


def generate_loaded(request: ChatCompletionRequest, engine):
    ensure_loaded(engine)
    return engine.generate(**prepare_generation(request, engine))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engines
    print("=" * 60)
    print("  ONYX API SERVER - Starting up...")
    print("=" * 60)
    
    from onyx.speculative import SpeculativeEngine
    
    print("\n[Initializing SpeculativeEngine: onyx-speculative]")
    _engines["onyx-speculative"] = SpeculativeEngine(
        cache_mode="paged",
        use_compile=True,
        lazy_load=False,
    )

    print("\n[Registering SpeculativeEngine: onyx-speculative-8b]")
    print("  8B target is lazy-loaded on first request to avoid startup memory cost.")
    _engines["onyx-speculative-8b"] = SpeculativeEngine(
        target_model_path="mlx-community/Qwen3-8B-4bit",
        cache_mode="paged",
        use_compile=True,
        lazy_load=True,
    )
    
    print("\n" + "=" * 60)
    print("  ONYX API SERVER - Ready to serve requests")
    print("  Endpoint: POST /v1/chat/completions")
    print("=" * 60 + "\n")
    
    yield
    
    print("\n[Shutting down Onyx API server...]")
    _engines = {}


app = FastAPI(
    title="Onyx API",
    description="OpenAI-compatible API for grammar-aware speculative decoding",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if os.environ.get("ONYX_API_DOCS", "1") == "1" else None,
    redoc_url=None,
)


def format_messages_as_prompt(messages: List[ChatMessage]) -> str:
    prompt_parts = []
    
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    
    prompt_parts.append("Assistant:")
    
    return "\n".join(prompt_parts)


def format_messages_for_engine(messages: List[ChatMessage], tokenizer) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return format_messages_as_prompt(messages)

    chat_messages = [
        {
            "role": msg.role,
            "content": msg.content,
        }
        for msg in messages
    ]

    try:
        return apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        return format_messages_as_prompt(messages)


def resolve_stop_tokens(
    stop: Optional[List[str]],
    tokenizer,
) -> Optional[List[List[int]]]:
    eos = getattr(tokenizer, "eos_token_id", None)
    stop_sequences = [[token] for token in (eos if isinstance(eos, list) else [eos])
                      if token is not None]
    for sequence in stop or []:
        ids = tokenizer.encode(sequence)
        if ids:
            stop_sequences.append(ids)

    return stop_sequences or None


def prepare_generation(request: ChatCompletionRequest, engine) -> dict:
    return dict(
        prompt=format_messages_for_engine(request.messages, engine.tokenizer),
        max_tokens=request.max_tokens, gamma=4,
        regex=request.regex,
        json_schema=json.dumps(request.json_schema) if request.json_schema is not None else None,
        draft_grammar_aware=True, temperature=request.temperature, top_p=request.top_p,
        stop_tokens=resolve_stop_tokens(request.stop, engine.tokenizer),
    )


def completion_reason(metrics: Dict[str, Any], stopped: bool = False) -> str:
    if stopped:
        return "stop"
    reason = metrics.get("finish_reason")
    if reason not in ("stop", "length", "grammar_complete"):
        raise RuntimeError("Generation did not report a valid finish reason")
    return reason


def truncate_at_stop(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return text

    positions = [text.find(sequence) for sequence in stop if sequence]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text
    return text[:min(positions)]


def flush_stream_text(pending: str, stop: Optional[List[str]]) -> tuple[str, str, bool]:
    active_stops = [sequence for sequence in (stop or []) if sequence]
    if not active_stops:
        return pending, "", False

    earliest_position = None
    for sequence in active_stops:
        position = pending.find(sequence)
        if position >= 0 and (earliest_position is None or position < earliest_position):
            earliest_position = position

    if earliest_position is not None:
        return pending[:earliest_position], "", True

    retain = max(len(sequence) for sequence in active_stops) - 1
    if retain <= 0:
        return pending, "", False
    if len(pending) <= retain:
        return "", pending, False

    flush_len = len(pending) - retain
    return pending[:flush_len], pending[flush_len:], False


def build_onyx_metrics(last_metrics: Dict[str, Any], grammar_active: bool) -> Dict[str, Any]:
    return {
        "tokens_per_second": last_metrics.get("tokens_per_second", 0),
        "acceptance_rate": last_metrics.get("acceptance_rate", 0),
        "ttft_ms": last_metrics.get("ttft", 0) * 1000 if last_metrics.get("ttft") else None,
        "grammar_constrained": grammar_active,
        "speculative_iterations": last_metrics.get("speculative_iterations", 0),
        "jit_compiled": last_metrics.get("jit_compiled", False),
        "compile_requested": last_metrics.get("compile_requested", False),
        "compile_reason": last_metrics.get("compile_reason", "disabled"),
    }


def create_streaming_response(request: ChatCompletionRequest, engine) -> Generator[str, None, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(delta, reason=None):
        payload = ChatCompletionChunk(
            id=completion_id, created=created, model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=reason)],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    yield chunk(ChatCompletionChunkDelta(role="assistant"))
    try:
        pending = ""
        stopped = False
        terminal = None
        ensure_loaded(engine)
        with closing(engine.stream_generate(**prepare_generation(request, engine))) as events:
            for text, metrics in events:
                if metrics is not None:
                    terminal = metrics
                    break
                pending += text
                text, pending, stopped = flush_stream_text(pending, request.stop)
                if text:
                    yield chunk(ChatCompletionChunkDelta(content=text))
                if stopped:
                    break
        reason = completion_reason(terminal or {}, stopped)
        if pending and not stopped:
            yield chunk(ChatCompletionChunkDelta(content=pending))
        yield chunk(ChatCompletionChunkDelta(), reason)
    except Exception as error:
        payload = {"error": {"message": str(error), "type": "server_error"}}
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


async def exclusive_stream(request: ChatCompletionRequest, engine):
    """Own the engine for a whole stream; each generation step runs off the event loop."""
    async with engine_lock(request.model):
        chunks = create_streaming_response(request, engine)
        try:
            async for chunk in iterate_in_threadpool(chunks):
                yield chunk
        finally:
            # A disconnect cancels this task. Close generation before the next
            # request can own the engine, even while that cancellation is pending.
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(chunks.close)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Onyx API",
        "version": "0.2.0",
        "endpoints": ["/v1/chat/completions", "/v1/models"],
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "onyx-speculative",
                "object": "model",
                "created": 1706745600,
                "owned_by": "onyx",
                "description": "Grammar-aware speculative decoding (0.5B draft + 1.5B target)",
            },
            {
                "id": "onyx-speculative-8b",
                "object": "model", 
                "created": 1706745600,
                "owned_by": "onyx",
                "description": "Grammar-aware speculative decoding (0.5B draft + 8B target)",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    engine = get_engine(request.model)
    if request.stream:
        return StreamingResponse(
            exclusive_stream(request, engine), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        async with engine_lock(request.model):
            # Off the event loop, so other requests and streams keep being served.
            output, metrics = await run_in_threadpool(generate_loaded, request, engine)
        text = truncate_at_stop(output, request.stop)
        reason = completion_reason(metrics, text != output)
        prompt_tokens = metrics.get("prompt_tokens", 0)
        completion_tokens = metrics.get("generated_tokens", 0)
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0, message=ChatMessage(role="assistant", content=text), finish_reason=reason,
            )],
            usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens),
            onyx_metrics=build_onyx_metrics(
                metrics, request.regex is not None or request.json_schema is not None,
            ),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
