"""The OpenAI-compatible HTTP server.

One model instance, one forward pass at a time. That is not a simplification to be removed later:
the engine streams a model's weights through the device on every token, so two concurrent sequences
do not share a pass the way a batched server's do -- they interleave, evict each other's layers from
the cache, and both run slower than either would alone. Continuous batching is deliberately out of
scope. Requests queue, and each is told how many are ahead of it.

The threading is worth stating plainly, because everything else follows from it:

  * The event loop runs HTTP. It never touches the model or the tokenizer.
  * One worker thread runs generation, start to finish, for one request at a time.
  * The two meet at :class:`_Job`, which carries events from the worker into the loop with
    ``call_soon_threadsafe`` and carries cancellation back the other way through a plain
    ``threading.Event``.

Cancellation matters more here than in a batched server. An abandoned request holds the *only*
worker, so a client that hangs up mid-answer would otherwise stall every other request behind it for
as long as the generation had left to run -- minutes, on a storage-bound model. The cancel flag is
checked inside the streamer, which is the one place both generation loops (transformers' own, and
the speculative one) pass through on every token.
"""
import asyncio
import collections
import logging
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from ..hw import caps
from .protocol import (FINISH_LENGTH, FINISH_STOP, FINISH_TOOL_CALLS, SSE_DONE,
                       ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChoice,
                       ChatCompletionRequest, ChatCompletionResponse, CompletionChoice,
                       CompletionChunk, CompletionRequest, CompletionResponse, DeltaMessage,
                       ModelCard, ModelList, RequestError, ResponseMessage, SamplingSettings, Usage,
                       chat_id, completion_id, error_payload, sse)
from .toolcalls import (ContentDelta, ToolCallDelta, ToolCallStream, ToolSetup, render_message,
                        resolve_tools, select_parser)
from . import multimodal
from . import prefix_cache as prefixes

log = logging.getLogger("rocketllm.server")

#: Placed on every response so a queued client can see it is queued rather than stalled.
QUEUE_POSITION_HEADER = "x-rocketllm-queue-position"
REQUEST_ID_HEADER = "x-rocketllm-request-id"


class _StopGeneration(Exception):
    """A stop sequence completed. Raised out of the streamer to end the generation loop.

    Raising is the only way to stop a loop this server does not own. It is safe precisely where it
    is raised: the streamer is called between forward passes, so every module's pre-hook has already
    been matched by its post-hook and the weight cache's refcounts are balanced. Raising from
    anywhere inside a forward would strand an acquired entry and it could never be evicted again.
    """


class _Cancelled(Exception):
    """The client went away. Same mechanism, no result."""


# ---- events between the worker thread and the event loop --------------------------------------

@dataclass
class Delta:
    text: str


@dataclass
class Completed:
    finish_reason: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    #: Finished tool calls, for the non-streaming path. The streaming path has already sent these
    #: as deltas; both come from the same parse, so they cannot disagree.
    tool_calls: tuple = ()


@dataclass
class Failed:
    error: BaseException


class _End:
    """Sentinel: no more events for this job."""


_END = _End()


# ---- incremental detokenisation ----------------------------------------------------------------

class StreamDecoder:
    """Token ids in, printable text out, one step at a time.

    Two things make this less trivial than calling ``decode`` per token. A token is not a character:
    a multi-byte codepoint is routinely split across two tokens and decodes to U+FFFD until its last
    byte arrives, so emitting eagerly would put a permanent replacement character in the client's
    output. And a token's rendering depends on the ones before it -- SentencePiece resolves a
    leading space from context -- so tokens cannot be decoded in isolation either.

    So the whole run is decoded, not each token, and the result is diffed against what has already
    been sent. To keep that from being quadratic over a long generation the sequence is committed in
    segments at newlines, which is a boundary no tokenizer renders across.
    """

    def __init__(self, tokenizer, hold_back=0):
        self.tokenizer = tokenizer
        #: Characters kept back from the client. See :class:`_Sink` for why a stop sequence needs it.
        self.hold_back = int(hold_back)
        self._committed = ""
        self._segment = []
        #: Everything decoded so far, sent or not.
        self.text = ""
        #: How much of `text` has been handed to the client.
        self.emitted = 0

    def push(self, ids):
        self._segment.extend(ids)
        decoded = self.tokenizer.decode(self._segment, skip_special_tokens=True)
        if decoded.endswith("\n"):
            self._committed += decoded
            self._segment = []
            decoded = ""
        self.text = self._committed + decoded

    def take(self):
        """The next chunk that is safe to send."""
        limit = len(self.text)
        while limit > self.emitted and self.text[limit - 1] == "�":
            limit -= 1
        limit = max(self.emitted, limit - self.hold_back)
        chunk = self.text[self.emitted:limit]
        self.emitted = limit
        return chunk

    def flush(self):
        """Everything still held back. Called once, when the generation has actually ended."""
        chunk = self.text[self.emitted:].rstrip("�")
        self.emitted = len(self.text)
        return chunk

    def stable_text(self):
        """The decoded text without a trailing replacement character.

        What the tool-call parser reads. It manages its own cursor over the whole text rather than
        taking chunks, so it needs the part that will not change under it -- and a partial codepoint
        at the tail is precisely the part that will.
        """
        return self.text.rstrip("�")

    def truncate(self, length):
        """Cut the text short, which is what a stop sequence does."""
        self.text = self.text[:length]
        self.emitted = min(self.emitted, len(self.text))


def _first_stop(text, stops, start=0):
    """Where the earliest stop sequence begins, or None."""
    found = None
    for stop in stops:
        at = text.find(stop, start)
        if at != -1 and (found is None or at < found):
            found = at
    return found


def _as_ids(value):
    """Token ids out of whatever the generation loop handed the streamer."""
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        value = tolist()
    if isinstance(value, int):
        return [value]
    ids = []
    stack = [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            ids.append(int(item))
    return ids


class _Sink:
    """transformers' streamer protocol, with stop sequences and cancellation folded in.

    Both generation loops in this project call ``put`` once per step and ``end`` at the finish, so
    this is the single place where the server can watch tokens appear regardless of which loop
    produced them. Doing stop sequences here rather than through transformers'
    ``stopping_criteria`` is what makes them work on both: the speculative loop is not transformers'
    loop and never sees a stopping criterion.

    The hold-back is the subtle part. A stop sequence is a string, so it can straddle a token
    boundary, and by the time it is recognised its first characters may already have been streamed
    to the client -- who cannot un-print them. Holding back as many characters as the longest stop
    sequence guarantees that cannot happen: at every step before the sequence completes, the text is
    shorter than its end offset, so nothing at or past its start has been sent.
    """

    def __init__(self, tokenizer, settings, emit, cancel, tools=None):
        self.decoder = StreamDecoder(
            tokenizer, hold_back=max((len(s) for s in settings.stop), default=0))
        self.stop = settings.stop
        self.max_new_tokens = settings.max_new_tokens
        #: A ToolCallStream, or None when this request did not ask for tools. When it is set, it
        #: owns content emission -- it is the only thing that knows which of the text is an answer
        #: and which is a call the client must never see as prose.
        self.tools = tools
        self._emit = emit
        self._cancel = cancel
        self.token_count = 0
        self.last_token = None
        self.stopped_on_sequence = False
        self._prompt_seen = False
        self._finished = False
        #: When the first generated token appeared. Everything before it is prefill, so this is
        #: what makes the prefill cost a measurement rather than an estimate.
        self.first_token_at = None
        #: Optional PrefixSession, told about each token so it can key a checkpoint by them.
        self.prefix = None

    @property
    def text(self):
        return self.tools.content if self.tools is not None else self.decoder.text

    @property
    def tool_calls(self):
        return tuple(self.tools.calls) if self.tools is not None else ()

    def put(self, value):
        # generate() hands the streamer the prompt before the first generated token. It is input,
        # not output, and every streamer in transformers drops it the same way.
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        if self._cancel.is_set():
            raise _Cancelled()

        ids = _as_ids(value)
        if not ids:
            return
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        self.token_count += len(ids)
        self.last_token = ids[-1]
        if self.prefix is not None:
            # Before the text is decoded, because the prefix cache keys checkpoints by token ids
            # and the next forward may reach a length whose key needs this one.
            self.prefix.observe_tokens(ids)
        self.decoder.push(ids)

        at = _first_stop(self.decoder.text, self.stop)
        if at is not None:
            self.decoder.truncate(at)
            self.stopped_on_sequence = True
            self.end()
            raise _StopGeneration()
        self._pump()

    def _pump(self):
        if self.tools is None:
            self._send(self.decoder.take())
            return
        for event in self.tools.push(self.decoder.stable_text()):
            self._forward(event)

    def end(self):
        # Guarded because there are two ways here -- transformers' loop calls end() when it finishes,
        # and a stop sequence flushes on its way out -- and flushing twice would replay the tail.
        if self._finished:
            return
        self._finished = True
        if self.tools is None:
            self._send(self.decoder.flush())
            return
        for event in self.tools.finish(self.decoder.stable_text()):
            self._forward(event)

    def _forward(self, event):
        if isinstance(event, ContentDelta):
            self._send(event.text)
        else:
            self._emit(event)

    def _send(self, text):
        if text:
            self._emit(Delta(text))

    def finish_reason(self, eos_ids):
        """What actually ended this generation.

        Order matters twice over. A model that emits its end-of-sequence token on the very last
        token it was allowed has stopped of its own accord, not been cut off, and reporting "length"
        there would tell an agentic client to ask for a continuation of an answer that is already
        complete.

        And "length" outranks "tool_calls". A call that was still being written when the budget ran
        out has arguments that are not valid JSON, so announcing it as a tool call hands the client
        something it cannot parse and no reason why. "length" is both the truth and the signal an
        agentic loop already knows how to handle.
        """
        ended_itself = (self.stopped_on_sequence
                        or (self.last_token is not None and self.last_token in eos_ids))
        if not ended_itself and self.token_count >= self.max_new_tokens:
            return FINISH_LENGTH
        if self.tool_calls:
            return FINISH_TOOL_CALLS
        return FINISH_STOP


# ---- the queue ----------------------------------------------------------------------------------

class _Job:
    """One generation: a callable for the worker, an event stream for the endpoint.

    Events travel worker -> loop through ``call_soon_threadsafe``; cancellation travels loop ->
    worker through a ``threading.Event`` the streamer polls. Neither side ever blocks on the other.
    """

    def __init__(self, run, loop, label=""):
        self.id = uuid.uuid4().hex[:12]
        self.label = label
        self._run = run
        self._loop = loop
        self._events = asyncio.Queue()
        self.cancel = threading.Event()
        self.position = 0
        self.enqueued_at = time.time()
        self.started_at = None
        self.finished_at = None
        #: How this job ended. NOT ``cancel.is_set()``: the endpoints set that flag on the way out
        #: of every request, including the ones that finished perfectly, so reading the flag counted
        #: each ordinary reply as a cancellation and /health reported a server dropping every
        #: request it served.
        self.cancelled = False
        self.failed = False

    def emit(self, event):
        try:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)
        except RuntimeError:
            # The loop has closed under us -- the server is shutting down, or the client's task was
            # torn down before the worker noticed. Nothing can read these events any more, so stop
            # producing them rather than generating into a void that still holds the worker.
            self.cancel.set()

    def execute(self):
        try:
            if self.cancel.is_set():
                # Abandoned while it was still queued. Never touch the model for it.
                self.cancelled = True
                return
            self._run(self)
        except _Cancelled:
            self.cancelled = True
        except BaseException as exc:  # noqa: BLE001 - one bad request must not kill the worker
            self.failed = True
            log.exception("request %s failed", self.id)
            self.emit(Failed(exc))
        finally:
            self.emit(_END)

    async def events(self):
        while True:
            event = await self._events.get()
            if event is _END:
                return
            yield event


class RequestQueue:
    """A single worker thread, and the queue in front of it.

    There is exactly one worker because there is exactly one model. Adding a second would not add
    throughput -- the two would take turns inside the same weight cache, each evicting what the
    other is about to need, which is the pathological access pattern the cache's policy exists to
    avoid in the first place.
    """

    def __init__(self, name="rocketllm-generate"):
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._pending = collections.deque()
        self._current = None
        self._closed = False
        self.completed = 0
        self.cancelled = 0
        self.failed = 0
        self._thread = threading.Thread(target=self._work, name=name, daemon=True)
        self._thread.start()

    def submit(self, job):
        """Enqueue, and return how many requests have to finish before this one starts."""
        with self._ready:
            if self._closed:
                raise RequestError("the server is shutting down", status_code=503,
                                   err_type="server_error")
            ahead = len(self._pending) + (1 if self._current is not None else 0)
            job.position = ahead
            self._pending.append(job)
            self._ready.notify()
        if ahead:
            log.info("request %s queued at position %d", job.id, ahead)
        return ahead

    def _work(self):
        while True:
            with self._ready:
                while not self._pending and not self._closed:
                    self._ready.wait()
                if not self._pending:
                    return
                job = self._pending.popleft()
                self._current = job
            job.started_at = time.time()
            try:
                job.execute()
            finally:
                job.finished_at = time.time()
                with self._ready:
                    self._current = None
                    self.completed += 1
                    self.cancelled += bool(job.cancelled)
                    self.failed += bool(job.failed)

    def stats(self):
        with self._lock:
            running = self._current
            return {
                "pending": len(self._pending),
                "depth": len(self._pending) + (1 if running is not None else 0),
                "running": running.id if running is not None else None,
                "running_seconds": (round(time.time() - running.started_at, 3)
                                    if running is not None and running.started_at else None),
                "completed": self.completed,
                "cancelled": self.cancelled,
                "failed": self.failed,
            }

    def close(self, timeout=10.0):
        with self._ready:
            self._closed = True
            for job in self._pending:
                job.cancel.set()
            if self._current is not None:
                self._current.cancel.set()
            self._ready.notify_all()
        self._thread.join(timeout)


# ---- the engine ---------------------------------------------------------------------------------

def _int_or_none(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _eos_ids(model, tokenizer):
    ids = set()
    for source in (getattr(model, "generation_config", None), tokenizer):
        value = getattr(source, "eos_token_id", None)
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            ids.update(int(v) for v in value if isinstance(v, int))
    return frozenset(ids)


class GenerationEngine:
    """One OpenAI request, one pass of one model.

    Everything in here that touches the model or the tokenizer runs on the queue's worker thread.
    That is not only about the model: a fast tokenizer is a Rust object with no documented thread
    safety, and encoding a new prompt on the event loop while the worker is detokenising the current
    answer would be two threads inside it at once.
    """

    def __init__(self, model, model_id=None, max_tokens=None, tool_parser=None,
                 prefix_cache="auto", prefix_cache_bytes=None):
        self.model = model
        self.tokenizer = model.tokenizer
        #: Which raw tool-call syntax this model emits, read off its chat template rather than its
        #: name. `tool_parser=` overrides it for a checkpoint whose template does not say.
        self.tool_parser = select_parser(self.tokenizer, getattr(model, "config", None),
                                         override=tool_parser)
        self.model_id = model_id or _default_model_id(model)
        #: An optional server-wide ceiling on a single reply, for a shared deployment. Without one
        #: the limit is the model's own context, which is the honest default.
        self.max_tokens_cap = _int_or_none(max_tokens)
        self.context_length = _context_length(model, self.tokenizer)
        self.eos_ids = _eos_ids(model, self.tokenizer)
        self.started_at = time.time()
        self._tools_render_checked = False

        self.layer_count = _layer_count(model)
        # Seeded with everything that changes what a token's KV actually is. Two models sharing a
        # spill directory would otherwise restore each other's caches -- same tokens, entirely
        # different numbers, and no error anywhere.
        seed = prefixes.namespace_seed(
            self.model_id, getattr(model, "running_dtype", ""),
            getattr(model, "kv_cache_choice", ""), self.layer_count,
            prefixes.FORMAT_VERSION)
        wanted, reason = prefixes.resolve(
            prefix_cache if isinstance(prefix_cache, str) else
            (prefixes.PREFIX_ON if prefix_cache else prefixes.PREFIX_OFF),
            weight_bytes=getattr(model, "_weight_bytes", None),
            device_bytes=getattr(model, "_device_budget", None))
        self.prefixes = prefixes.build(
            profile=getattr(model, "profile", None), seed=seed,
            enabled=wanted and self.layer_count > 0, capacity_bytes=prefix_cache_bytes)
        if self.prefixes.enabled:
            print(f"prefix cache: on -- {reason}. "
                  f"{self.prefixes.capacity_bytes / 1024 ** 3:.1f}GB host, "
                  f"{self.prefixes.spill_bytes / 1024 ** 3:.1f}GB spill, "
                  f"{self.prefixes.block_size}-token blocks")
        else:
            if wanted:
                reason = ("the profile left it no host budget, or the model's layer count could "
                          "not be read, so nothing could be checkpointed")
            print(f"prefix cache: off -- {reason}")

    # -- prompts ----------------------------------------------------------------------------------

    def _encode_chat(self, request, setup):
        """``(input_ids, extra model inputs)`` for a chat request.

        The second element is empty for every text request and carries a vision-language model's
        pixel tensors for the rest. It is kept separate from the token ids all the way through
        because two things branch on whether it is empty: prefix reuse, which keys on token ids and
        would happily hand back a cache built from a different picture, and speculative decoding,
        whose draft never sees it.
        """
        if multimodal.has_images(request.messages):
            return self._encode_multimodal(request, setup)

        messages = [render_message(m) for m in request.messages]
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            try:
                return self._apply_template(messages, request, setup), {}
            except Exception as exc:  # noqa: BLE001 - a broken template must not be a 500
                caps.announce_once(
                    "server-chat-template",
                    f"this model's chat template could not render the request ({exc}); falling "
                    f"back to a plain role: content transcript. Replies will be worse than the "
                    f"model is capable of -- the template is what it was trained to see.",
                    logging.WARNING)
        else:
            caps.announce_once(
                "server-no-chat-template",
                "this tokenizer ships no chat template, so /v1/chat/completions renders a plain "
                "role: content transcript. Use /v1/completions if you want to control the prompt "
                "exactly.", logging.WARNING)
        return self._encode_text(_plain_transcript(messages), add_special_tokens=True), {}

    def _encode_multimodal(self, request, setup):
        """A request carrying images, through the checkpoint's own processor.

        The processor does both halves and neither can be done without it: it renders the template
        with the model's image placeholders in the right places, and it expands each placeholder to
        the number of tokens THIS checkpoint's patch and merge sizes produce for THAT image's
        dimensions. Encoding the text separately and pasting pixels beside it gets the count wrong
        and the model reads the picture at the wrong offset.
        """
        processor = getattr(self.model, "processor", None)
        if processor is None:
            raise RequestError(
                "this request contains an image, and the loaded model has no multimodal processor "
                "-- it is a text-only checkpoint, or its processor could not be built (the load "
                "log says which). Send text-only messages, or serve a vision-language checkpoint.",
                param="messages", code="model_not_multimodal")
        if not (getattr(processor, "chat_template", None)
                or getattr(self.tokenizer, "chat_template", None)):
            raise RequestError(
                "this model ships no chat template, so there is nowhere to put an image: the "
                "template is what renders the placeholder tokens the model was trained to read a "
                "picture from. Use /v1/completions for exact control of a text prompt.",
                param="messages", code="no_chat_template")

        images = multimodal.collect_images(request.messages)
        messages = [render_message(m, content=multimodal.template_content(m))
                    for m in request.messages]
        kwargs = dict(add_generation_prompt=request.add_generation_prompt, tokenize=False)
        if request.continue_final_message:
            kwargs["add_generation_prompt"] = False
            kwargs["continue_final_message"] = True
        if setup.tools:
            kwargs["tools"] = setup.tools
            self._check_template_renders_tools(messages, request, setup)
        try:
            text = processor.apply_chat_template(messages, **kwargs)
            encoded = processor(text=[text], images=images, return_tensors="pt")
        except RequestError:
            raise
        except Exception as exc:  # noqa: BLE001 - a client's picture must not be a 500
            raise RequestError(
                f"this model's processor could not render the request ({exc}). The usual cause is "
                f"an image the model's own template does not expect -- a count of images the "
                f"conversation's turns do not account for, or a size the processor rejects.",
                param="messages") from exc

        # attention_mask is rebuilt in _generate from the ids, and input_ids is returned on its own,
        # so everything else the processor produced is what the model needs beside the tokens.
        extra = {name: value for name, value in encoded.items()
                 if name not in ("input_ids", "attention_mask")}
        log.info("multimodal request: %d image(s), %d prompt tokens, extra inputs %s",
                 len(images), int(encoded["input_ids"].shape[-1]), sorted(extra))
        return encoded["input_ids"], extra

    def _apply_template(self, messages, request, setup):
        kwargs = dict(add_generation_prompt=request.add_generation_prompt,
                      tokenize=True, return_tensors="pt", return_dict=True)
        if request.continue_final_message:
            kwargs["add_generation_prompt"] = False
            kwargs["continue_final_message"] = True
        # Tool definitions go through the template's own tool support, which is the only rendering
        # that matches what the model was trained on. Which tools that is, is tool_choice's business
        # -- see resolve_tools.
        if setup.tools:
            kwargs["tools"] = setup.tools
            self._check_template_renders_tools(messages, request, setup)
        encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        # return_dict=True gives a BatchEncoding, which is a UserDict -- a Mapping, but NOT a dict
        # subclass, so an isinstance(dict) test here silently returns the container itself and the
        # first thing to touch .shape fails somewhere unrelated. Without return_dict, and on older
        # transformers, the tensor comes back on its own.
        if isinstance(encoded, Mapping):
            return encoded["input_ids"]
        return encoded

    def _check_template_renders_tools(self, messages, request, setup):
        """Say so, once, if this model's template quietly drops the tools it was handed.

        Not every chat template has tool support, and Jinja ignores a variable it does not
        reference rather than complaining about it. So a client can send perfectly good tool
        definitions to a model that never sees them, get prose back, and have nothing anywhere to
        explain why. Rendering it both ways once and comparing is the only reliable test -- there is
        no flag on a template that declares this.
        """
        if self._tools_render_checked:
            return
        self._tools_render_checked = True
        try:
            probe = dict(add_generation_prompt=request.add_generation_prompt, tokenize=False)
            with_tools = self.tokenizer.apply_chat_template(messages, tools=setup.tools, **probe)
            without_tools = self.tokenizer.apply_chat_template(messages, **probe)
        except Exception:  # noqa: BLE001 - a diagnostic must never be what fails a request
            return
        if with_tools == without_tools:
            caps.announce_once(
                "server-template-ignores-tools",
                "this model's chat template has no tool support, so the tool definitions in this "
                "request never reach the prompt and the model cannot know the tools exist. Replies "
                "will be prose. Nothing here can fix that -- the template is part of the "
                "checkpoint -- but a model trained for tool use will have one that renders them.",
                logging.WARNING)

    def _encode_text(self, text, add_special_tokens=True):
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
        return encoded["input_ids"]

    def _encode_completion(self, request):
        if isinstance(request.prompt, str):
            return self._encode_text(request.prompt, add_special_tokens=True)
        import torch

        return torch.tensor([list(request.prompt)], dtype=torch.long)

    def _token_budget(self, request, prompt_tokens):
        """How many tokens this reply may have, and why.

        The default is what is left of the model's context, which is the only limit that is a
        property of the model rather than a number someone picked. A request may ask for less; the
        server-wide cap, if one was set, may impose less again.
        """
        if prompt_tokens >= self.context_length:
            raise RequestError(
                f"the prompt is {prompt_tokens} tokens but this model's context is "
                f"{self.context_length}; there is no room left to generate a reply",
                param="messages", code="context_length_exceeded")
        available = self.context_length - prompt_tokens
        wanted = request.token_limit() or available
        if self.max_tokens_cap:
            wanted = min(wanted, self.max_tokens_cap)
        return max(1, min(int(wanted), available))

    # -- generation -------------------------------------------------------------------------------

    def chat_job(self, request, loop):
        return _Job(lambda job: self._run(job, request, chat=True), loop, label="chat")

    def completion_job(self, request, loop):
        return _Job(lambda job: self._run(job, request, chat=False), loop, label="completion")

    def _run(self, job, request, chat):
        # Settled before the prompt is built, because it decides what goes INTO the prompt as well
        # as what is read back out of the reply. /v1/completions has no tools field at all.
        setup = resolve_tools(request) if chat else ToolSetup()
        if setup.unenforced:
            caps.announce_once(f"server-tool-choice-{setup.unenforced[:24]}", setup.unenforced,
                               logging.INFO)

        input_ids, model_inputs = (self._encode_chat(request, setup) if chat
                                   else (self._encode_completion(request), {}))
        prompt_tokens = int(input_ids.shape[-1])
        settings = SamplingSettings.from_request(request, self._token_budget(request, prompt_tokens))
        if settings.ignored:
            caps.announce_once(
                "server-ignored-sampling",
                f"{', '.join(settings.ignored)} are accepted for client compatibility but not "
                f"applied -- transformers has no equivalent with the same meaning. Use "
                f"repetition_penalty if you want the effect they are usually reached for.",
                logging.INFO)

        # Only a request that offered tools gets its reply parsed for them. A model that writes
        # <tool_call> in an ordinary answer is quoting, not calling, and swallowing that as a call
        # would lose the text -- which is also what keeps the marker-less generic parser away from
        # prose it has no business touching.
        tools = None
        if setup.parse:
            tools = ToolCallStream(self.tool_parser,
                                   hold=max((len(s) for s in settings.stop), default=0))

        sink = _Sink(self.tokenizer, settings, job.emit, job.cancel, tools=tools)
        # A request carrying images opts out of prefix reuse, and that is a correctness rule rather
        # than a tuning choice: the cache is keyed by token ids, and one image's placeholder tokens
        # are identical to another's of the same size. Two different pictures would key the same
        # checkpoint and the second request would answer about the first one's, with nothing
        # anywhere to show it had happened.
        store = None if model_inputs else self.prefixes
        session = prefixes.PrefixSession(
            store, input_ids[0].tolist(), config=getattr(self.model, "kv_cache_config", None),
            layers=self.layer_count, device=getattr(self.model, "device", None))
        sink.prefix = session

        started = time.perf_counter()
        self._generate(input_ids, settings, sink, session, model_inputs)
        prefill_seconds = ((sink.first_token_at - started)
                           if sink.first_token_at is not None else None)
        session.finish(prefill_seconds=prefill_seconds)
        self._log_prefix(job, session, prefill_seconds)

        if job.cancel.is_set():
            raise _Cancelled()
        job.emit(Completed(finish_reason=sink.finish_reason(self.eos_ids), text=sink.text,
                           prompt_tokens=prompt_tokens, completion_tokens=sink.token_count,
                           tool_calls=sink.tool_calls))

    def _log_prefix(self, job, session, prefill_seconds):
        # session.cache is None for a request that opted out -- reporting a "miss" for one that was
        # never looked up would put it in the hit-rate arithmetic a reader does in their head.
        if not self.prefixes.enabled or session.cache is None:
            return
        summary = session.summary()
        measured = f"{prefill_seconds * 1000:.0f}ms" if prefill_seconds is not None else "unknown"
        if summary["hit"]:
            log.info(
                "request %s: prefix %s hit, %d of %d prompt tokens reused, %d prefilled in %s"
                "%s",
                job.id, summary["tier"], summary["tokens_reused"], summary["prompt_tokens"],
                summary["tokens_prefilled"], measured,
                "" if summary["seconds_saved"] is None else
                f" ({summary['seconds_saved']:.2f}s faster than a measured full prefill of this "
                f"size)")
        else:
            log.info("request %s: prefix miss, %d prompt tokens prefilled in %s",
                     job.id, summary["prompt_tokens"], measured)

    def _generate(self, input_ids, settings, sink, session=None, model_inputs=None):
        import torch

        if settings.seed is not None:
            # Seeded globally rather than through a generator object, because that is the only lever
            # that reaches both loops: transformers' generate() takes no generator, and the
            # speculative decoder draws from the global RNG when it is not given one. Safe because
            # the queue guarantees nothing else is generating at the same time.
            torch.manual_seed(settings.seed)

        kwargs = settings.to_generate_kwargs()
        kwargs["streamer"] = sink
        kwargs["use_cache"] = True
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        if pad is not None:
            kwargs["pad_token_id"] = pad

        device = getattr(self.model, "device", None)
        if device is not None:
            input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        for name, value in (model_inputs or {}).items():
            # The pixel tensors travel to the device with the ids. A float one is cast to the
            # running dtype as well: the processor produces float32 whatever the model runs in, and
            # a vision tower whose weights are bf16 refuses a float32 input in its first matmul.
            if isinstance(value, torch.Tensor):
                if device is not None:
                    value = value.to(device)
                if value.is_floating_point():
                    dtype = getattr(self.model, "running_dtype", None)
                    if dtype is not None:
                        value = value.to(dtype)
            kwargs[name] = value

        if session is not None and self.prefixes.enabled:
            # Handing generate() a pre-filled cache with the FULL input_ids is how transformers is
            # meant to be told a prefix is already done: it takes the cache's length as the start
            # of cache_position and forwards only the tokens past it.
            cache = session.begin(self._new_cache)
            if cache is not None:
                kwargs["past_key_values"] = cache
        try:
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        except _StopGeneration:
            # A stop sequence completed. The reply is finished and correct; only the loop had to be
            # interrupted, because neither generation loop knows about server-side stop sequences.
            pass

    def _new_cache(self):
        """A fresh cache for a request the prefix cache could not serve.

        Falls back to a DynamicCache when the model's own choice is full precision, because that
        path normally lets transformers build its own and there would be nothing to checkpoint.
        """
        build = getattr(self.model, "_new_kv_cache", None)
        cache = build() if callable(build) else None
        if cache is not None:
            return cache
        from transformers.cache_utils import DynamicCache

        return DynamicCache()

    # -- introspection ----------------------------------------------------------------------------

    def model_card(self):
        model = self.model
        return ModelCard(id=self.model_id, rocketllm={
            "context_length": self.context_length,
            "device": str(getattr(model, "running_device", "") or ""),
            "dtype": str(getattr(model, "running_dtype", "") or ""),
            "kv_cache": getattr(model, "kv_cache_choice", None),
            "speculation": bool(getattr(model, "spec", None)),
            "tool_parser": self.tool_parser.family,
            "prefix_cache": self.prefixes.enabled,
            "max_tokens_cap": self.max_tokens_cap,
            # Whether this deployment accepts image parts at all. A client that gets a 400 on its
            # first picture should be able to find out here rather than by reading the load log.
            "multimodal": getattr(model, "processor", None) is not None,
        })

    def health(self, queue, verbose=False):
        """Everything a performance bug report needs, in one payload."""
        model = self.model
        payload = {
            "status": "ok",
            "model": self.model_id,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "queue": queue.stats(),
            "generation": {
                "context_length": self.context_length,
                # First thing to check when tool calls come back as prose: the wrong family here
                # means the reply was parsed for a syntax the model does not emit.
                "tool_parser": self.tool_parser.family,
                "kv_cache": getattr(model, "kv_cache_choice", None),
                "pin_policy": getattr(model, "pin_policy", None),
                "expert_residency": getattr(model, "expert_residency", None),
                "prefetching": getattr(model, "prefetching", None),
            },
            "hardware": _hardware_summary(model),
            "cache": _cache_summary(model),
            "budget": _budget_summary(model),
            "prefix_cache": self.prefixes.report(),
        }
        for name, source in (("speculation", "speculation_report"), ("experts", "expert_report")):
            reporter = getattr(model, source, None)
            if callable(reporter):
                try:
                    payload[name] = reporter()
                except Exception:  # noqa: BLE001 - a report must never fail a health check
                    payload[name] = None
        if verbose:
            profile = getattr(model, "profile", None)
            payload["profile_text"] = profile.describe() if profile is not None else None
        return payload


def _default_model_id(model):
    """A name for the loaded model, for clients that echo it and humans who read it.

    The model only knows where its weights ended up, and for a downloaded checkpoint that is
    ``.../models--org--name/snapshots/<commit>`` -- whose last component is a commit hash, which is
    a terrible thing to call a model in a response. The repo id is recoverable from the cache
    directory above it, so recover it. `rocketllm serve` passes the id the user actually typed and
    never reaches this.
    """
    path = getattr(model, "model_local_path", None) or getattr(model, "name_or_path", None)
    if not path:
        return "rocketllm"
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    for part in reversed(parts):
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/")
    return parts[-1] if parts else "rocketllm"


def _layer_count(model):
    """How many decoder layers the KV cache will have.

    The prefix cache needs it to know which `update` call ends a step: that is the only moment
    every layer is consistent at the new length, and so the only moment a snapshot is exact.
    """
    config = getattr(model, "config", None)
    for source in (config, getattr(config, "text_config", None)):
        found = _int_or_none(getattr(source, "num_hidden_layers", None))
        if found:
            return found
    layers = getattr(model, "layers", None)
    if layers:
        # RocketModel's list is embed + decoder layers + norm + lm_head.
        return max(0, len(layers) - 3)
    return 0


def _context_length(model, tokenizer):
    """The model's context window, from the model rather than from a constant."""
    config = getattr(model, "config", None)
    for source in (config, getattr(config, "text_config", None)):
        for name in ("max_position_embeddings", "seq_length", "n_positions", "model_max_length"):
            found = _int_or_none(getattr(source, name, None))
            if found:
                return found
    # model_max_length is a very large sentinel on tokenizers that do not know their limit, so it is
    # only trusted when it looks like a real window.
    declared = _int_or_none(getattr(tokenizer, "model_max_length", None))
    if declared and declared < 1_000_000:
        return declared
    # Last resort, and a property of transformer checkpoints rather than of any machine: 2048 is
    # the smallest window in common use, so a model whose config states nothing is assumed to have
    # the shortest one rather than a longer one it might not support. Nothing about the hardware
    # enters here -- how much context actually fits is the KV cache's decision, made against the
    # measured device budget.
    return _int_or_none(getattr(model, "max_seq_len", None)) or 2048


def _assistant_message(completed):
    """The reply, as OpenAI shapes it.

    Content goes to null rather than "" when the model only called a tool. That is what OpenAI
    sends, and clients branch on it: one that tests ``if message.content`` would otherwise print an
    empty assistant turn into the conversation it is building, and then send that back next turn.
    """
    tool_calls = [call.to_payload() for call in completed.tool_calls] or None
    content = completed.text
    if tool_calls and not (content or "").strip():
        content = None
    return ResponseMessage(role="assistant", content=content, tool_calls=tool_calls)


def _plain_transcript(messages):
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def _hardware_summary(model):
    profile = getattr(model, "profile", None)
    if profile is None:
        return {"available": False,
                "reason": "no hardware profile was built for this run; run `rocketllm profile`"}
    storage = profile.storage or {}
    return {
        "available": True,
        "fingerprint": profile.fingerprint,
        "probed_at": profile.probed_at,
        "backend": profile.backend,
        "device": profile.device_name,
        "compute_capability": profile.compute_capability,
        "device_memory_bytes": {"total": profile.device_total_bytes,
                                "free_at_probe": profile.device_free_bytes},
        "host_memory_bytes": {"total": profile.host_total_bytes,
                              "available_at_probe": profile.host_available_bytes},
        "cpu_count": profile.cpu_count,
        "bandwidth_bytes_per_s": {
            "device_memory": profile.device_memory_bandwidth,
            "host_to_device_pinned": profile.host_to_device_pinned_bandwidth,
            "host_to_device_pageable": profile.host_to_device_pageable_bandwidth,
            "storage_queue_depth_1": storage.get("queue_depth_1_bytes_per_s"),
        },
        "storage": {"path": storage.get("path"), "rotational": storage.get("rotational"),
                    "saturating_concurrency": storage.get("saturating_concurrency")},
        "capabilities": {"dtypes": profile.dtypes, "pinned_memory": profile.pinned_memory,
                         "async_copy_streams": profile.async_copy_streams,
                         "triton": profile.triton,
                         "fused_4bit": profile.fused_4bit.get("any_usable")},
        "derived": {name: derivation.value for name, derivation in profile.derived.items()},
        "warnings": list(profile.warnings or ()),
    }


def _cache_summary(model):
    cache = getattr(model, "cache", None)
    if cache is None:
        return None
    try:
        return cache.report()
    except Exception:  # noqa: BLE001
        return None


def _budget_summary(model):
    budget = getattr(model, "budget", None)
    if budget is None:
        return None
    try:
        return {"target_bytes": budget.target(), "current_bytes": budget.current(),
                "reserve_bytes": budget.reserve_bytes,
                "external": dict(getattr(budget, "external", {}) or {})}
    except Exception:  # noqa: BLE001
        return None


# ---- the app -------------------------------------------------------------------------------------

async def _watch_for_disconnect(request, job):
    """Wait on the receive channel for the client to hang up, and cancel the generation when it does.

    An explicit watcher, rather than relying on the ASGI server to tear the response down, and that
    is not belt-and-braces -- it is the only thing that works. Starlette does cancel the task
    iterating a streaming response when the client disconnects, but cancelling that task does NOT
    run the ``finally`` in the async generator it was iterating: between chunks the generator is
    suspended at a ``yield``, not inside an ``await``, so it is abandoned and finalised whenever the
    garbage collector next gets to it. Measured before this existed: a client that hung up after two
    chunks still left the worker generating all 400 remaining tokens. On a non-streaming response
    nothing is cancelled at all, so there was never another candidate there.

    The channel is read directly rather than through ``Request.is_disconnected``, which makes its
    read non-blocking by wrapping it in an already-cancelled scope -- and that cancellation escapes
    into this task, which the scope does not own, killing the watcher on its first poll.

    Reading it here does not starve Starlette's own disconnect listener on a streaming response:
    both wait on the same event, both wake when it fires, and a connection that has gone away
    answers every subsequent ``receive()`` with the same message.
    """
    try:
        while True:
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                log.info("request %s: client disconnected, cancelling generation", job.id)
                job.cancel.set()
                return
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - the watcher must never be what fails a request
        log.debug("disconnect watcher for %s stopped", job.id, exc_info=True)


def create_app(engine, queue=None, title="RocketLLM"):
    """Build the ASGI app around an already-loaded engine.

    The engine is passed in rather than constructed here so that the protocol can be tested against
    a mock model on a machine with no accelerator -- which is most of CI.
    """
    import contextlib

    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, StreamingResponse

    owns_queue = queue is None
    queue = queue or RequestQueue()

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        # Only a queue this function created is this function's to shut down. One that was passed in
        # belongs to the caller, who may still be using it -- which is what the tests do.
        if owns_queue:
            queue.close()

    app = FastAPI(title=title, version=_package_version(), lifespan=lifespan)
    app.state.engine = engine
    app.state.queue = queue

    @app.exception_handler(RequestError)
    async def _bad_request(request, exc):
        return JSONResponse(exc.payload(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _schema_error(request, exc):
        # FastAPI's own 422 body is a list of pydantic errors under "detail", which OpenAI clients
        # do not read. Re-shape it into the error envelope they do.
        first = (exc.errors() or [{}])[0]
        location = ".".join(str(part) for part in first.get("loc", ())[1:]) or None
        return JSONResponse(
            error_payload(first.get("msg", "invalid request body"), param=location),
            status_code=400)

    def _headers(job):
        return {REQUEST_ID_HEADER: job.id, QUEUE_POSITION_HEADER: str(job.position)}

    # -- chat ------------------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        body.validate_request()
        job = engine.chat_job(body, asyncio.get_running_loop())
        queue.submit(job)
        response_id = chat_id()
        created = int(time.time())

        if body.stream:
            return StreamingResponse(
                _chat_stream(request, job, body, response_id, created, engine.model_id),
                media_type="text/event-stream", headers=_sse_headers(_headers(job)))

        completed = await _collect(request, job)
        message = _assistant_message(completed)
        return JSONResponse(ChatCompletionResponse(
            id=response_id, created=created, model=engine.model_id,
            choices=[ChatCompletionChoice(index=0, message=message,
                                          finish_reason=completed.finish_reason)],
            usage=Usage.of(completed.prompt_tokens, completed.completion_tokens),
        ).to_payload(), headers=_headers(job))

    async def _chat_stream(request, job, body, response_id, created, model_id):
        def chunk(delta, finish_reason=None, usage=None, choices=True):
            return ChatCompletionChunk(
                id=response_id, created=created, model=model_id, usage=usage,
                choices=[ChatCompletionChunkChoice(index=0, delta=delta,
                                                   finish_reason=finish_reason)] if choices else [])

        watcher = asyncio.ensure_future(_watch_for_disconnect(request, job))
        try:
            # The role chunk goes out before the model has produced anything. That flushes the
            # response headers -- including the queue position -- so a client waiting behind a long
            # queue can see it is queued rather than guessing the server has hung.
            yield sse(chunk(DeltaMessage(role="assistant", content="")))
            usage = None
            async for event in job.events():
                if isinstance(event, Delta):
                    yield sse(chunk(DeltaMessage(content=event.text)))
                elif isinstance(event, ToolCallDelta):
                    # One call per delta, carrying its index. The id and name ride the first delta
                    # for an index and never change, because a client assembling arguments keys on
                    # them -- a regenerated id splits one call into several, each holding a
                    # fragment of the arguments and none of them callable.
                    yield sse(chunk(DeltaMessage(tool_calls=[event.to_payload()])))
                elif isinstance(event, Completed):
                    usage = Usage.of(event.prompt_tokens, event.completion_tokens)
                    yield sse(chunk(DeltaMessage(), finish_reason=event.finish_reason))
                elif isinstance(event, Failed):
                    yield sse(_failure_payload(event.error))
            if usage is not None and body.wants_usage_chunk():
                yield sse(chunk(DeltaMessage(), usage=usage, choices=False))
            yield SSE_DONE
        finally:
            # The ordinary end of a stream. Cancelling an abandoned one is the watcher's job above,
            # because this block is not reliably reached when the client disappears.
            job.cancel.set()
            watcher.cancel()

    # -- legacy completions -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        body.validate_request()
        job = engine.completion_job(body, asyncio.get_running_loop())
        queue.submit(job)
        response_id = completion_id()
        created = int(time.time())

        if body.stream:
            return StreamingResponse(
                _completion_stream(request, job, body, response_id, created, engine.model_id),
                media_type="text/event-stream", headers=_sse_headers(_headers(job)))

        completed = await _collect(request, job)
        text = completed.text
        if body.echo and isinstance(body.prompt, str):
            text = body.prompt + text
        return JSONResponse(CompletionResponse(
            id=response_id, created=created, model=engine.model_id,
            choices=[CompletionChoice(index=0, text=text,
                                      finish_reason=completed.finish_reason)],
            usage=Usage.of(completed.prompt_tokens, completed.completion_tokens),
        ).to_payload(), headers=_headers(job))

    async def _completion_stream(request, job, body, response_id, created, model_id):
        def chunk(text, finish_reason=None, usage=None, choices=True):
            return CompletionChunk(
                id=response_id, created=created, model=model_id, usage=usage,
                choices=[CompletionChoice(index=0, text=text,
                                          finish_reason=finish_reason)] if choices else [])

        watcher = asyncio.ensure_future(_watch_for_disconnect(request, job))
        try:
            if body.echo and isinstance(body.prompt, str):
                yield sse(chunk(body.prompt))
            usage = None
            async for event in job.events():
                if isinstance(event, Delta):
                    yield sse(chunk(event.text))
                elif isinstance(event, Completed):
                    usage = Usage.of(event.prompt_tokens, event.completion_tokens)
                    yield sse(chunk("", finish_reason=event.finish_reason))
                elif isinstance(event, Failed):
                    yield sse(_failure_payload(event.error))
            if usage is not None and body.wants_usage_chunk():
                yield sse(chunk("", usage=usage, choices=False))
            yield SSE_DONE
        finally:
            job.cancel.set()
            watcher.cancel()

    # -- metadata ----------------------------------------------------------------------------------

    @app.get("/v1/models")
    async def models():
        return JSONResponse(ModelList(data=[engine.model_card()]).to_payload())

    @app.get("/v1/models/{model_id:path}")
    async def model_detail(model_id: str):
        if model_id != engine.model_id:
            raise RequestError(f"no such model: {model_id}", param="model", status_code=404,
                               err_type="not_found_error", code="model_not_found")
        return JSONResponse(engine.model_card().model_dump())

    @app.get("/health")
    async def health(verbose: bool = False):
        return JSONResponse(engine.health(queue, verbose=verbose))

    return app


def _sse_headers(extra):
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Nginx buffers proxied responses by default, which turns a token stream into one delivery
        # at the end. This is the documented opt-out and is ignored by everything else.
        "X-Accel-Buffering": "no",
    }
    headers.update(extra)
    return headers


def _failure_payload(error):
    if isinstance(error, RequestError):
        return error.payload()
    return error_payload(f"{type(error).__name__}: {error}", err_type="server_error")


async def _collect(request, job):
    """Wait out a non-streaming generation, giving the worker back if the client goes away.

    A plain JSON endpoint is not cancelled by the ASGI server when the client hangs up -- only
    streaming responses are -- so the disconnect has to be polled for. Without this the worker would
    spend minutes finishing a reply that nothing is left to receive, with every queued request
    waiting behind it.
    """
    watcher = asyncio.ensure_future(_watch_for_disconnect(request, job))
    completed = None
    failure = None
    try:
        async for event in job.events():
            if isinstance(event, Completed):
                completed = event
            elif isinstance(event, Failed):
                failure = event.error
    finally:
        job.cancel.set()
        watcher.cancel()

    if failure is not None:
        raise failure if isinstance(failure, RequestError) else RequestError(
            f"{type(failure).__name__}: {failure}", status_code=500, err_type="server_error")
    if completed is None:
        from starlette.requests import ClientDisconnect

        raise ClientDisconnect("the client disconnected before the reply was finished")
    return completed


def _package_version():
    try:
        from importlib.metadata import version

        return version("rocketllm")
    except Exception:  # noqa: BLE001 - running from a source tree without an install
        return "0"


def serve(model, host="127.0.0.1", port=8000, model_id=None, max_tokens=None, log_level="info",
          tool_parser=None):
    """Load nothing, own nothing: run an app around a model somebody else built."""
    import uvicorn

    engine = GenerationEngine(model, model_id=model_id, max_tokens=max_tokens,
                              tool_parser=tool_parser)
    app = create_app(engine)
    print(f"serving {engine.model_id} on http://{host}:{port}  "
          f"(context {engine.context_length} tokens, one request at a time, "
          f"tool calls parsed as {engine.tool_parser.family})")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
