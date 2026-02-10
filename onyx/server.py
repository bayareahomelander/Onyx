import json
import time
import uuid
from typing import List, Optional, Dict, Any, Generator, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default="onyx-speculative",
        description="Model identifier. Use 'onyx-speculative' for grammar-aware speculation."
    )
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation"
    )
    max_tokens: Optional[int] = Field(
        default=256,
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        description="Sampling temperature (0 = greedy, higher = more random)"
    )
    stream: Optional[bool] = Field(
        default=False,
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
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(default=1, description="Number of completions to generate")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")


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


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized. Server is still starting up."
        )
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    print("=" * 60)
    print("  ONYX API SERVER - Starting up...")
    print("=" * 60)
    
    from onyx.speculative import SpeculativeEngine
    
    print("\n[Initializing SpeculativeEngine...]")
    _engine = SpeculativeEngine(
        cache_mode="paged",
        use_compile=True,
        lazy_load=False,
    )
    
    print("\n" + "=" * 60)
    print("  ONYX API SERVER - Ready to serve requests")
    print("  Endpoint: POST /v1/chat/completions")
    print("=" * 60 + "\n")
    
    yield
    
    print("\n[Shutting down Onyx API server...]")
    _engine = None


app = FastAPI(
    title="Onyx API",
    description="OpenAI-compatible API for grammar-aware speculative decoding",
    version="0.1.0",
    lifespan=lifespan,
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


def create_streaming_response(
    request: ChatCompletionRequest,
    engine,
) -> Generator[str, None, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model
    
    prompt = format_messages_as_prompt(request.messages)
    
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"
    
    try:
        json_schema_str = json.dumps(request.json_schema) if request.json_schema else None
        
        output, metrics = engine.generate(
            prompt=prompt,
            max_tokens=request.max_tokens or 256,
            gamma=4,
            regex=request.regex,
            json_schema=json_schema_str,
            draft_grammar_aware=True,
        )
        
        for char in output:
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=char),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Onyx API",
        "version": "0.1.0",
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
                "id": "onyx-speculative-7b",
                "object": "model", 
                "created": 1706745600,
                "owned_by": "onyx",
                "description": "Grammar-aware speculative decoding (0.5B draft + 7B target)",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    engine = get_engine()
    
    if request.stream:
        return StreamingResponse(
            create_streaming_response(request, engine),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    try:
        prompt = format_messages_as_prompt(request.messages)
        json_schema_str = json.dumps(request.json_schema) if request.json_schema else None
        
        grammar_active = request.regex is not None or request.json_schema is not None
        
        output, metrics = engine.generate(
            prompt=prompt,
            max_tokens=request.max_tokens or 256,
            gamma=4,
            regex=request.regex,
            json_schema=json_schema_str,
            draft_grammar_aware=True,
        )
        
        response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output),
                    finish_reason="grammar_complete" if grammar_active else "stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=metrics.get("prompt_tokens", 0),
                completion_tokens=metrics.get("generated_tokens", 0),
                total_tokens=metrics.get("prompt_tokens", 0) + metrics.get("generated_tokens", 0),
            ),
            onyx_metrics={
                "tokens_per_second": metrics.get("tokens_per_second", 0),
                "acceptance_rate": metrics.get("acceptance_rate", 0),
                "ttft_ms": metrics.get("ttft", 0) * 1000 if metrics.get("ttft") else None,
                "grammar_constrained": grammar_active,
                "speculative_iterations": metrics.get("speculative_iterations", 0),
            },
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
