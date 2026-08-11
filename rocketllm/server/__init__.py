"""The OpenAI-compatible server.

FastAPI, uvicorn and pydantic are an optional extra (``pip install rocketllm[server]``) rather than
base dependencies: a user who only ever calls ``generate()`` from Python should not have a web
framework installed for it. Importing this package without them raises with the command that fixes
it, following the same pattern as the other optional components.
"""
_INSTALL_HINT = (
    "the RocketLLM server needs FastAPI, uvicorn and pydantic, which are an optional extra. "
    "Install them with:  pip install 'rocketllm[server]'")

try:
    from .protocol import (FINISH_LENGTH, FINISH_STOP, FINISH_TOOL_CALLS, ChatCompletionRequest,
                           CompletionRequest, RequestError, SamplingSettings)
    from .toolcalls import FAMILIES, ToolCall, ToolCallStream, select_parser
    from .app import GenerationEngine, RequestQueue, create_app, serve
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(f"{_INSTALL_HINT}  (missing: {exc.name or exc})") from exc

__all__ = [
    "ChatCompletionRequest",
    "CompletionRequest",
    "FAMILIES",
    "FINISH_LENGTH",
    "FINISH_STOP",
    "FINISH_TOOL_CALLS",
    "GenerationEngine",
    "RequestError",
    "RequestQueue",
    "SamplingSettings",
    "ToolCall",
    "ToolCallStream",
    "create_app",
    "select_parser",
    "serve",
]
