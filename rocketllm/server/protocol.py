"""The OpenAI wire format: request schemas, response schemas, and SSE framing.

Nothing in this module touches a model, a device or the hardware profile. It is the shape of what
goes over HTTP and the normalisation of a client's sampling request into something ``generate()``
understands -- which makes it the half of the server that can be tested without a GPU, and most of
what tests/test_server.py exercises.

Three things clients genuinely depend on, and each is easy to get subtly wrong:

  * The SSE frame. ``data: {json}\\n\\n`` per event and a literal ``data: [DONE]\\n\\n`` to close.
    A stream that ends without [DONE] leaves well-behaved clients waiting for the socket to time
    out rather than returning.
  * ``finish_reason``. "stop" means the model chose to end (EOS or a stop string), "length" means we
    cut it off at max_tokens, "tool_calls" means it asked for a tool. An agentic loop branches on
    this, so reporting "stop" where the truth is "length" makes a truncated answer look deliberate
    and the loop proceeds on half an answer.
  * ``usage``. Real counts, not estimates. Clients bill and budget from them.

Requests accept unknown fields rather than rejecting them. Real clients (LangChain, llama-index,
various IDE plugins) send fields OpenAI itself ignores, and a 422 over one of those looks to the
user like the server is broken. Fields we DO understand are validated properly.
"""
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---- finish reasons ---------------------------------------------------------------------------

FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"
FINISH_REASONS = (FINISH_STOP, FINISH_LENGTH, FINISH_TOOL_CALLS)

# ---- SSE --------------------------------------------------------------------------------------

#: The terminator. A client that has read this knows the stream ended on purpose.
SSE_DONE = "data: [DONE]\n\n"


def sse(payload):
    """One SSE event. Accepts a model or a plain dict."""
    if isinstance(payload, BaseModel):
        payload = payload.to_payload() if hasattr(payload, "to_payload") else payload.model_dump()
    # Separators without spaces: these frames are the hot path of a streaming response and the
    # whitespace is pure bytes on the wire.
    return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


# ---- errors -----------------------------------------------------------------------------------

class RequestError(ValueError):
    """A client mistake, carrying everything the OpenAI error envelope needs.

    Raised from validation and caught at the endpoint, because the useful message is usually known
    where the value is checked and reconstructing it at the boundary loses the parameter name.
    """

    def __init__(self, message, param=None, status_code=400, err_type="invalid_request_error",
                 code=None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.status_code = status_code
        self.err_type = err_type
        self.code = code

    def payload(self):
        return error_payload(self.message, err_type=self.err_type, param=self.param, code=self.code)


def error_payload(message, err_type="invalid_request_error", param=None, code=None):
    """The error envelope openai-python and its lookalikes parse: everything under "error"."""
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


# ---- shared request pieces --------------------------------------------------------------------

class _Permissive(BaseModel):
    # protected_namespaces=() because OpenAI's own schema has a field called `model`, which pydantic
    # would otherwise warn about for colliding with its `model_` attribute namespace.
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class FunctionCall(BaseModel):
    name: str
    #: A JSON *string*, not an object. That is the OpenAI wire format and clients json.loads it.
    arguments: str = ""


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(_Permissive):
    role: str
    #: Either a plain string or the multi-part list newer clients send. Kept as-is here and
    #: flattened only when a prompt is built, so a template that understands parts still sees them.
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def text(self):
        """The message's content as a single string, for a template that cannot take parts."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for part in self.content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in (None, "text") and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)


class StreamOptions(_Permissive):
    #: OpenAI's opt-in for a final usage-only chunk. Off by default, because a client that does not
    #: expect the extra chunk can choke on choices being empty.
    include_usage: bool = False


class _SamplingRequest(_Permissive):
    """Everything the two completion endpoints share."""

    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    #: Not in OpenAI's schema. Exposed because transformers has it and it is genuinely useful on
    #: small models; a client that does not send it is unaffected.
    top_k: Optional[int] = None
    n: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    user: Optional[str] = None

    def _validate_common(self):
        if self.n is not None and self.n != 1:
            raise RequestError(
                "n must be 1. This server runs one forward pass at a time by design -- there is one "
                "model instance streaming its weights, so a second sequence would cost a second "
                "full pass rather than riding along in a batch. Send n separate requests.",
                param="n")
        if self.temperature is not None and self.temperature < 0:
            raise RequestError("temperature must be >= 0", param="temperature")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise RequestError("top_p must be in (0, 1]", param="top_p")
        if self.top_k is not None and self.top_k < 0:
            raise RequestError("top_k must be >= 0", param="top_k")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise RequestError("max_tokens must be >= 1", param="max_tokens")

    def stop_sequences(self):
        if self.stop is None:
            return ()
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop)
        return tuple(s for s in stops if isinstance(s, str) and s)

    def wants_usage_chunk(self):
        return bool(self.stream_options and self.stream_options.include_usage)


class ChatCompletionRequest(_SamplingRequest):
    messages: List[ChatMessage]
    #: Newer name for max_tokens. Preferred when both are sent, which is what OpenAI does.
    max_completion_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    add_generation_prompt: bool = True
    #: Not OpenAI. Lets a caller bypass the chat template entirely, which is how you reproduce a bad
    #: generation from a bug report without having to recreate the exact message list.
    continue_final_message: bool = False

    def validate_request(self):
        self._validate_common()
        if not self.messages:
            raise RequestError("messages must not be empty", param="messages")
        for index, message in enumerate(self.messages):
            if not message.role:
                raise RequestError(f"messages[{index}].role is required", param="messages")
        if self.max_completion_tokens is not None and self.max_completion_tokens < 1:
            raise RequestError("max_completion_tokens must be >= 1", param="max_completion_tokens")
        if self.continue_final_message and self.add_generation_prompt:
            raise RequestError(
                "continue_final_message and add_generation_prompt are mutually exclusive",
                param="continue_final_message")
        return self

    def token_limit(self):
        return self.max_completion_tokens or self.max_tokens


class CompletionRequest(_SamplingRequest):
    #: OpenAI's prompt has four shapes: a string, a token-id list, a list of strings and a list of
    #: token-id lists. The last two are batches, and batching is the thing this server deliberately
    #: does not do -- but they are accepted by the SCHEMA so that validation can refuse them with an
    #: explanation. Narrowing the annotation instead would hand the client pydantic's union-mismatch
    #: dump, which says nothing about why a batch is not served here.
    prompt: Union[str, List[int], List[str], List[List[int]]]
    echo: bool = False
    best_of: Optional[int] = None
    suffix: Optional[str] = None

    def validate_request(self):
        self._validate_common()
        if isinstance(self.prompt, list):
            if not self.prompt:
                raise RequestError("prompt must not be empty", param="prompt")
            if not all(isinstance(token, int) for token in self.prompt):
                raise RequestError(
                    "prompt must be a single string or a single list of token ids. A list of "
                    "prompts is a batch, and this server runs one forward pass at a time by design "
                    "-- send them as separate requests.",
                    param="prompt")
        elif not isinstance(self.prompt, str):
            raise RequestError("prompt must be a string or a list of token ids", param="prompt")
        if self.best_of is not None and self.best_of != 1:
            raise RequestError("best_of must be 1; this server generates one sequence per request",
                               param="best_of")
        if self.suffix is not None:
            raise RequestError("suffix is not supported (it needs a fill-in-the-middle template "
                               "this server does not render)", param="suffix")
        return self

    def token_limit(self):
        return self.max_tokens


# ---- sampling ---------------------------------------------------------------------------------

@dataclass
class SamplingSettings:
    """A client's sampling request, normalised into ``generate()`` keyword arguments.

    Kept apart from the request models because the mapping has judgement in it, and that judgement
    is worth reading in one place:

    ``temperature == 0`` is greedy, and greedy means ``do_sample=False`` with no warping arguments
    at all rather than ``temperature=0``. Passing a zero temperature would divide by it, and passing
    top_p alongside do_sample=False makes transformers warn on every single request about settings
    it is ignoring.

    ``stop`` is NOT forwarded. transformers implements stop strings as a stopping criterion inside
    its own generation loop, and this engine has a second loop -- speculative decoding -- that never
    sees it, so a stop string would work on one path and be silently ignored on the other. The
    server detects stop sequences itself while decoding the stream, which is one implementation for
    both paths and also gives the exact character offset to truncate the reply at.
    """

    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    stop: tuple = ()
    repetition_penalty: Optional[float] = None
    #: Set when the client asked for something that is accepted but not applied, so the server can
    #: say so once instead of pretending it took effect.
    ignored: tuple = ()

    @property
    def greedy(self):
        return self.temperature <= 0.0

    @classmethod
    def from_request(cls, request, max_new_tokens):
        ignored = []
        for name in ("presence_penalty", "frequency_penalty"):
            value = getattr(request, name, None)
            if value:
                ignored.append(name)
        temperature = 1.0 if request.temperature is None else float(request.temperature)
        return cls(
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            top_p=1.0 if request.top_p is None else float(request.top_p),
            top_k=0 if request.top_k is None else int(request.top_k),
            seed=request.seed,
            stop=request.stop_sequences(),
            repetition_penalty=(float(request.repetition_penalty)
                                if request.repetition_penalty else None),
            ignored=tuple(ignored))

    def to_generate_kwargs(self):
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": not self.greedy}
        if not self.greedy:
            kwargs["temperature"] = self.temperature
            if 0.0 < self.top_p < 1.0:
                kwargs["top_p"] = self.top_p
            if self.top_k:
                kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs


# ---- responses --------------------------------------------------------------------------------

def _new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex}"


def chat_id():
    return _new_id("chatcmpl")


def completion_id():
    return _new_id("cmpl")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def of(cls, prompt_tokens, completion_tokens):
        return cls(prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens),
                   total_tokens=int(prompt_tokens) + int(completion_tokens))


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class DeltaMessage(BaseModel):
    """A streaming increment. Only the fields that actually changed are sent.

    OpenAI's first chunk carries the role and no content, every middle chunk carries content alone,
    and the last carries an empty delta beside the finish_reason. Clients merge them key by key, so
    sending ``"content": null`` in the role chunk is not harmless -- some assemblers overwrite the
    accumulated string with it. See :meth:`_ChunkBase.to_payload` for where the nulls are dropped.
    """

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class _Payload(BaseModel):
    def to_payload(self):
        return self.model_dump()


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ResponseMessage
    #: Always present, never null on a completed choice.
    finish_reason: str = FINISH_STOP
    logprobs: Optional[Any] = None


class ChatCompletionResponse(_Payload):
    id: str = Field(default_factory=chat_id)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    #: Null until the final chunk. The key is always present because some clients read it directly
    #: rather than with a default.
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(_Payload):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionChunkChoice] = Field(default_factory=list)
    #: Only on the final chunk, and only when the client asked for it.
    usage: Optional[Usage] = None

    def to_payload(self):
        data = self.model_dump()
        if data.get("usage") is None:
            data.pop("usage", None)
        for choice in data["choices"]:
            delta = choice.get("delta") or {}
            choice["delta"] = {k: v for k, v in delta.items() if v is not None}
        return data


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = FINISH_STOP
    logprobs: Optional[Any] = None


class CompletionResponse(_Payload):
    id: str = Field(default_factory=completion_id)
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[CompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class CompletionChunk(_Payload):
    """The /v1/completions stream. Same envelope as the non-streaming response by design: the legacy
    endpoint streams whole choice objects rather than deltas, and clients parse them with the same
    code."""

    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[CompletionChoice] = Field(default_factory=list)
    usage: Optional[Usage] = None

    def to_payload(self):
        data = self.model_dump()
        if data.get("usage") is None:
            data.pop("usage", None)
        return data


# ---- models listing ---------------------------------------------------------------------------

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "rocketllm"
    #: Not OpenAI. What this model costs to run here, so `GET /v1/models` is worth reading on a
    #: machine you did not configure yourself.
    rocketllm: Optional[Dict[str, Any]] = None


class ModelList(_Payload):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)
