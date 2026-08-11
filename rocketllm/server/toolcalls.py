"""Reading tool calls out of what a model actually emits, and rendering them back in.

Every model family invented its own syntax for "call this function". Hermes and Qwen wrap a JSON
object in ``<tool_call>`` tags, Mistral prefixes a JSON array with ``[TOOL_CALLS]``, Llama emits bare
JSON with ``parameters`` instead of ``arguments`` after an optional ``<|python_tag|>``, DeepSeek uses
its own full-width delimiters and puts the function name OUTSIDE the JSON. Clients want one shape:
OpenAI's ``tool_calls``, whose ``arguments`` is a JSON *string*.

Two design decisions carry most of the weight here.

**One state machine for both response modes.** :class:`ToolCallStream` is fed the growing raw text
and produces content deltas and tool-call deltas; the streaming endpoint forwards them, and the
non-streaming endpoint accumulates them. They cannot disagree about where a call started or what its
arguments were, because there is only one implementation to disagree with -- the same reason stop
sequences are applied here rather than handed to transformers.

**Arguments stream by span, not by re-parsing.** A client assembles ``arguments`` by concatenating
fragments, so the fragments must be exact substrings of the final JSON, in order, with nothing
repeated or dropped. Re-serialising a partially-parsed object on every token cannot give that. So
the parser locates where the arguments value *begins* in the raw text and tracks where it ends,
emitting the raw slice between what it has already sent and what has arrived. The concatenation is
then the original text by construction. :class:`_ValueSpan` is that tracker.

The id is minted when a call opens and never changes, which is the property clients depend on and
the one most implementations get wrong: an id regenerated per chunk makes a client assemble one call
into several, each with a fragment of the arguments.

Adding a family is a subclass and a registry entry -- see :class:`ToolCallParser`.
"""
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

# ---- what comes out ----------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A finished call, in OpenAI's shape."""

    index: int
    id: str
    name: str
    #: A JSON *string*, not an object. That is the wire format, and clients json.loads it.
    arguments: str = ""

    def to_payload(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ContentDelta:
    text: str


@dataclass
class ToolCallDelta:
    """One increment of one call.

    ``id`` and ``name`` appear on the first delta for an index and never again; ``arguments`` is a
    fragment to be concatenated. This mirrors what OpenAI sends, because clients are written against
    that and merge deltas key by key.
    """

    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = ""

    def to_payload(self):
        function = {}
        if self.name is not None:
            function["name"] = self.name
        if self.id is not None:
            # OpenAI's opening delta carries an empty arguments string, so a client that assembles
            # by concatenation has something to concatenate onto from the start.
            function["arguments"] = self.arguments
        elif self.arguments:
            function["arguments"] = self.arguments
        payload = {"index": self.index}
        if self.id is not None:
            payload["id"] = self.id
            payload["type"] = "function"
        if function:
            payload["function"] = function
        return payload


# ---- incremental JSON ----------------------------------------------------------------------------

_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_ARGUMENTS_RE = re.compile(r'"(?:arguments|parameters)"\s*:\s*')


class _ValueSpan:
    """Where one JSON value starts and ends inside a string that is still growing.

    Scanning resumes from where it stopped rather than restarting, so feeding a token at a time
    costs the same as reading the value once. ``end`` stays None until the value is closed, which is
    exactly the signal for "there may be more arguments coming".
    """

    __slots__ = ("start", "end", "_pos", "_depth", "_in_string", "_escape")

    def __init__(self, start):
        self.start = start
        self.end = None
        self._pos = start
        self._depth = 0
        self._in_string = False
        self._escape = False

    def advance(self, text):
        if self.end is not None:
            return
        index = self._pos
        while index < len(text):
            char = text[index]
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif char == "\\":
                    self._escape = True
                elif char == '"':
                    self._in_string = False
                    if self._depth == 0:  # the whole value was a bare string
                        self.end = index + 1
                        self._pos = self.end
                        return
            elif char == '"':
                self._in_string = True
            elif char in "{[":
                self._depth += 1
            elif char in "}]":
                self._depth -= 1
                if self._depth == 0:
                    self.end = index + 1
                    self._pos = self.end
                    return
                if self._depth < 0:
                    # The container holding this value closed, so the value was a bare scalar and
                    # ended just before it.
                    self.end = index
                    self._pos = index
                    return
            elif char == "," and self._depth == 0:
                self.end = index
                self._pos = index
                return
            index += 1
        self._pos = index

    def slice(self, text):
        return text[self.start:self.end if self.end is not None else len(text)]


def _object_verdict(text, start, budget):
    """Does the JSON object at `start` look like a tool call? True / False / None for "wait".

    Used by the parsers that have no start marker to go on. A bare ``{`` in an answer is far more
    often prose or code than a call, so the decision is made on whether a ``name`` key turns up
    before the object closes -- and until it can be made, the text is held rather than emitted,
    because a client cannot un-print content that turns out to have been a call.
    """
    depth = 0
    in_string = False
    escape = False
    key_start = None
    index = start
    while index < len(text):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
                if depth == 1 and key_start is not None:
                    if text[key_start:index] in ("name", "function"):
                        return True
                key_start = None
        elif char == '"':
            in_string = True
            key_start = index + 1
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return False  # closed without ever naming anything
        index += 1
    # Ran out of text. Hold, unless this has gone on so long that it is plainly not a call and
    # holding it back would swallow a large piece of a real answer.
    return None if index - start <= budget else False


# ---- parsers ---------------------------------------------------------------------------------------

@dataclass
class _Found:
    """A call definitely starts here."""

    content_end: int   # emit content up to this offset; the marker itself is not content
    json_start: int    # where the call's JSON object begins
    name: Optional[str] = None   # for families that put the name outside the JSON


@dataclass
class _Pending:
    """Something that may be a call starts here; not enough text to be sure yet."""

    content_end: int


class ToolCallParser:
    """One family's raw syntax.

    To add a family:

      1. Subclass this. Set ``family`` and ``start_markers`` -- the latter is what content is held
         back by, so a partial marker is never streamed to a client as text.
      2. Implement ``matches`` against the chat template. Detect the SYNTAX the template emits, not
         the model's name: a fine-tune keeps its parent's template and a name check would miss it.
      3. Implement ``find_call``. Return ``_Found`` when a call certainly starts, ``_Pending`` when
         one might, ``None`` when there is none in sight.
      4. Override ``resume`` if the format has a closing marker to step over, and
         ``next_in_region`` if one marker can introduce several calls.
      5. Add it to ``PARSERS``, ahead of ``GenericParser``, which matches everything.

    Nothing else needs to change: reading the name and streaming the arguments out of the JSON is
    shared, and so is everything downstream.
    """

    family = "generic"
    #: Literal strings that announce a call. The longest of these is how much content is withheld.
    start_markers = ()
    #: Closes a call, if the format has one.
    end_marker = ""
    #: True when the JSON object a call points at IS the arguments, rather than a wrapper holding
    #: them under a key. DeepSeek is the family that needs this: it names the function in its
    #: markers and puts nothing but the arguments in the JSON.
    payload_is_arguments = False
    #: OpenAI ids look like call_<hex>. Mistral's own templates reject anything but 9 alphanumeric
    #: characters, and raise rather than ignore it -- so an id minted here has to survive being sent
    #: back in the next request.
    id_prefix = "call_"
    id_length = 24
    #: How far a maybe-a-call region may grow before it is given up on and released as content.
    tentative_budget = 400

    def matches(self, template):
        return False

    def new_id(self):
        return f"{self.id_prefix}{uuid.uuid4().hex[:self.id_length]}"

    @property
    def hold_back(self):
        return max((len(marker) for marker in self.start_markers), default=0)

    def find_call(self, text, cursor, state):
        raise NotImplementedError

    def after_call(self, text, position, final):
        """Where the cursor goes once a call's JSON has closed at `position`.

        None means "not enough text yet, hold". That return is load-bearing: between the closing
        brace of the JSON and the format's end marker sits text that is neither arguments nor
        answer, and emitting it while waiting for the marker to arrive puts a literal
        ``</tool_call>`` in the client's reply.
        """
        if not self.end_marker:
            return position
        at = text.find(self.end_marker, position)
        if at != -1:
            return at + len(self.end_marker)
        return position if final else None

    def next_in_region(self, text, position):
        """Another call inside the region the last one was in, for formats that pack several."""
        return None

    def after_region(self, text, position, final):
        """Where the cursor goes when a multi-call region has run out of calls.

        Its job is to step over the format's own punctuation -- Mistral's closing ``]`` -- so it is
        not handed back to the client as though the model had said it.
        """
        return position


class HermesParser(ToolCallParser):
    """Hermes, Qwen, NousResearch and the many fine-tunes that borrowed the template.

    ``<tool_call>\\n{"name": "x", "arguments": {...}}\\n</tool_call>``, repeated per call.
    """

    family = "hermes"
    start_markers = ("<tool_call>",)
    end_marker = "</tool_call>"

    def matches(self, template):
        return "<tool_call>" in template

    def find_call(self, text, cursor, state):
        at = text.find("<tool_call>", cursor)
        if at == -1:
            return None
        brace = text.find("{", at)
        return _Found(at, brace) if brace != -1 else _Pending(at)


class MistralParser(ToolCallParser):
    """Mistral: ``[TOOL_CALLS] [{"name": "x", "arguments": {...}}, ...]`` -- one marker, an array."""

    family = "mistral"
    start_markers = ("[TOOL_CALLS]",)
    # Mistral's own chat templates raise unless every tool_call id is exactly 9 alphanumeric
    # characters, so an OpenAI-style call_<hex> would come back in the next request and abort the
    # render rather than degrade.
    id_prefix = ""
    id_length = 9

    def matches(self, template):
        return "[TOOL_CALLS]" in template

    def find_call(self, text, cursor, state):
        at = text.find("[TOOL_CALLS]", cursor)
        if at == -1:
            return None
        brace = text.find("{", at)
        return _Found(at, brace) if brace != -1 else _Pending(at)

    def next_in_region(self, text, position):
        index = _skip(text, position, ", \t\r\n")
        if index >= len(text):
            return _Pending(position)
        return _Found(position, index) if text[index] == "{" else None

    def after_region(self, text, position, final):
        index = _skip(text, position, ", \t\r\n")
        return index + 1 if index < len(text) and text[index] == "]" else position


class LlamaParser(ToolCallParser):
    """Llama 3.1 / 3.2: ``{"name": "x", "parameters": {...}}``, optionally after ``<|python_tag|>``
    and optionally several separated by ``;``.

    The bare form has no marker at all, so a ``{`` has to be treated as a maybe until the object
    either names a function or closes without one. Llama's built-in tool syntax
    (``brave_search.call(query="...")``) is a function-call expression rather than JSON and is not
    read here; it comes back as ordinary content.
    """

    family = "llama"
    start_markers = ("<|python_tag|>",)

    def matches(self, template):
        return "<|python_tag|>" in template or "ipython" in template

    def find_call(self, text, cursor, state):
        tagged = text.find("<|python_tag|>", cursor)
        if tagged != -1:
            brace = text.find("{", tagged)
            return _Found(tagged, brace) if brace != -1 else _Pending(tagged)
        return _naked_object(self, text, cursor, state)

    def next_in_region(self, text, position):
        index = _skip(text, position, " \t\r\n")
        if index >= len(text):
            return _Pending(position)
        if text[index] != ";":
            return None
        index = _skip(text, index + 1, " \t\r\n")
        if index >= len(text):
            return _Pending(position)
        return _Found(position, index) if text[index] == "{" else None


class DeepSeekParser(ToolCallParser):
    """DeepSeek V3 / R1, whose delimiters are full-width and whose name sits outside the JSON:

    ``<|tool calls begin|><|tool call begin|>function<|tool sep|>NAME\\n```json\\n{...}\\n```...``
    (with U+FF5C and U+2581 in place of the ASCII shown above).

    This is the family that stops the shared machinery from assuming the name is a JSON key, which
    is why it is worth carrying.
    """

    family = "deepseek"
    CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
    CALL_BEGIN = "<｜tool▁call▁begin｜>"
    SEPARATOR = "<｜tool▁sep｜>"
    CALL_END = "<｜tool▁call▁end｜>"
    CALLS_END = "<｜tool▁calls▁end｜>"
    start_markers = (CALLS_BEGIN, CALL_BEGIN)
    end_marker = CALL_END
    # The fenced JSON after the separator is the arguments themselves; the name came from the
    # markers. Nothing in it is a wrapper to look inside.
    payload_is_arguments = True

    def matches(self, template):
        return self.CALL_BEGIN in template or self.CALLS_BEGIN in template

    def find_call(self, text, cursor, state):
        at = text.find(self.CALL_BEGIN, cursor)
        wrapper = text.find(self.CALLS_BEGIN, cursor)
        if at == -1:
            # The wrapper lands a whole marker before the per-call one does, and the plain hold-back
            # is only as long as a single marker -- so without holding at the wrapper explicitly, it
            # gets streamed out as text while the per-call marker is still arriving.
            return _Pending(wrapper) if wrapper != -1 else None
        content_end = wrapper if 0 <= wrapper <= at else at

        separator = text.find(self.SEPARATOR, at)
        if separator == -1:
            return _Pending(content_end)
        line_end = text.find("\n", separator)
        if line_end == -1:
            return _Pending(content_end)
        name = text[separator + len(self.SEPARATOR):line_end].strip()
        brace = text.find("{", line_end)
        if brace == -1:
            return _Pending(content_end)
        return _Found(content_end, brace, name or None)

    def after_call(self, text, position, final):
        """Step over this call's end marker, and the wrapper too when this was the last call.

        The wrapper cannot be left to ``after_region``: that runs once, at the moment the region is
        found to hold no more calls, which is usually before the closing wrapper has arrived.
        """
        at = text.find(self.CALL_END, position)
        if at == -1:
            return position if final else None
        index = _skip(text, at + len(self.CALL_END), " \t\r\n")
        if text.startswith(self.CALLS_END, index):
            return index + len(self.CALLS_END)
        # Either another call follows, or the closing wrapper is still on its way. Only a partial
        # match means wait; anything else is the answer resuming.
        if not final and self.CALLS_END.startswith(text[index:index + len(self.CALLS_END)]):
            return None
        return index


class GenericParser(ToolCallParser):
    """The fallback: a JSON object with a name, fenced or bare.

    It is deliberately last and deliberately cautious. Because it has no marker to key off, every
    ``{`` is a candidate, and an answer containing code or JSON would be mangled if candidates were
    accepted eagerly -- so a region is held only until the object names a function or closes without
    one. It runs solely on requests that supplied tools, which is the other half of what keeps it
    from touching ordinary prose.
    """

    family = "generic"
    start_markers = ("```json", "```", "{")

    def matches(self, template):
        return True

    def find_call(self, text, cursor, state):
        return _naked_object(self, text, cursor, state)

    def after_call(self, text, position, final):
        index = _skip(text, position, " \t\r\n")
        if text.startswith("```", index):
            return index + 3
        if not final and "```".startswith(text[index:index + 3]):
            return None
        return position


def _skip(text, index, characters):
    while index < len(text) and text[index] in characters:
        index += 1
    return index


#: How much text may follow a code fence before it is accepted as an ordinary code block rather
#: than the wrapper around a call. Long enough for "```json\n" and some indentation, short enough
#: that a real code block is delayed by only a few characters.
_FENCE_PATIENCE = 24


def _open_fence(text, cursor):
    """A code fence with nothing after it yet, which may be about to wrap a call.

    Content stops here while the wait is short. Without it the fence is streamed as text one
    character at a time -- the ordinary hold-back covers a partial "```json" but not the newline and
    whitespace that follow it before the brace arrives.
    """
    at = text.rfind("```", cursor)
    if at == -1 or len(text) - at > _FENCE_PATIENCE:
        return None
    return at


def _fence_language(text, fence):
    """The language tag on a code fence, or None while the rest of the line is still arriving."""
    line_end = text.find("\n", fence + 3)
    if line_end == -1:
        return None
    return text[fence + 3:line_end].strip().lower()


def _track_fences(text, state):
    """Follow fenced code blocks as text arrives. Returns the open block's language, or None.

    Carried as state rather than re-derived from the cursor each time, and that is the whole point:
    the cursor moves past the opening fence as the code streams out as content, so by the time the
    braces inside the block arrive there is nothing left in front of the cursor to say they are
    inside one. Scanning resumes where it stopped, so this costs the same as reading the text once.
    """
    at = state.get("fence_at", 0)
    language = state.get("fence_language")
    # A fence arriving one character at a time must not be half-read, so anything within two
    # characters of the end is left for next time.
    limit = len(text) - 2
    while at < limit:
        found = text.find("```", at)
        if found == -1 or found >= limit:
            at = limit
            break
        if language is None:
            opening = _fence_language(text, found)
            if opening is None:
                break          # the tag is still arriving; read it again next time
            language = opening
        else:
            language = None
        at = found + 3
    state["fence_at"] = max(at, 0)
    state["fence_language"] = language
    return language


def _naked_object(parser, text, cursor, state):
    """Find an unmarked JSON object that turns out to name a function.

    Shared by the two parsers that have to cope with a call arriving with no announcement. A fence
    immediately before the object belongs to the call rather than to the answer, so content is cut
    before it.

    A fenced block in some OTHER language is skipped whole. Code is where a brace with a "name" key
    is most likely to turn up innocently -- observed on a real model asked about JSON in Python,
    which wrote ``data = {"name": ..., "arguments": ...}`` inside a ```python block and had it
    lifted out of the answer and served as a call. A parser that eats an answer is worse than one
    that misses a call, and this is the answer-eating case that actually happens.
    """
    if _track_fences(text, state) not in (None, "", "json"):
        return None    # inside a block of code, which is all content whatever it contains
    search = cursor
    while True:
        brace = text.find("{", search)
        fence = text.find("```", search)
        if fence != -1 and (brace == -1 or fence < brace):
            language = _fence_language(text, fence)
            if language is None:
                return _Pending(fence)          # the tag is still arriving
            if language not in ("", "json"):
                closing = text.find("```", fence + 3)
                if closing == -1:
                    # An unfinished block of code. All of it is content.
                    return None
                search = closing + 3
                continue
        if brace == -1:
            fence = _open_fence(text, search)
            return _Pending(fence) if fence is not None else None
        verdict = _object_verdict(text, brace, parser.tentative_budget)
        if verdict is None:
            return _Pending(_fence_start(text, brace, cursor))
        if verdict:
            return _Found(_fence_start(text, brace, cursor), brace)
        # Not a call. Step over this object and keep looking; the text stays content.
        span = _ValueSpan(brace)
        span.advance(text)
        search = span.end if span.end is not None else brace + 1


def _fence_start(text, brace, floor):
    """Where the content before a call really ends, ignoring a code fence that introduces it."""
    index = brace
    while index > floor and text[index - 1] in " \t\r\n":
        index -= 1
    for opener in ("```json", "```"):
        if text.startswith(opener, max(floor, index - len(opener))) and index - len(opener) >= floor:
            return index - len(opener)
    return brace


#: Checked in order; the generic parser matches everything, so it stays last.
PARSERS = (HermesParser(), MistralParser(), DeepSeekParser(), LlamaParser(), GenericParser())
FAMILIES = tuple(parser.family for parser in PARSERS)


def select_parser(tokenizer=None, config=None, override=None):
    """Pick a parser from what the chat template actually emits.

    Detection reads the template rather than the architecture name, for the same reason expert
    layouts are detected structurally: a fine-tune inherits its parent's template under a new name,
    and an unreleased model has no name anyone has heard of. The template is the ground truth about
    the syntax the model was trained to produce.
    """
    if override:
        for parser in PARSERS:
            if parser.family == override:
                return parser
        raise ValueError(f"unknown tool-call parser {override!r}; known families are "
                         f"{', '.join(FAMILIES)}")

    template = getattr(tokenizer, "chat_template", None) or ""
    if not isinstance(template, str):
        # Some tokenizers ship a dict of named templates; the tool-enabled one is what matters.
        template = " ".join(str(value) for value in getattr(template, "values", lambda: [])())
    for parser in PARSERS:
        if parser.matches(template):
            return parser
    return PARSERS[-1]


# ---- the state machine ------------------------------------------------------------------------------

class _OpenCall:
    """A call being read as its text arrives.

    Two spans, not one, and the distinction is what keeps format punctuation out of the answer. The
    ARGUMENTS span is what gets streamed to the client. The OBJECT span is the whole JSON the call
    lives in, and it closes one brace later -- so resuming from the arguments span would leave the
    wrapper's ``}`` and everything after it to be emitted as though the model had said it. For a
    family whose payload is the arguments outright, the two are the same span.
    """

    __slots__ = ("index", "id", "name", "json_start", "object", "arguments", "sent_open",
                 "sent_arguments", "quoted")

    def __init__(self, index, call_id, json_start, name=None, payload_is_arguments=False):
        self.index = index
        self.id = call_id
        self.name = name
        self.json_start = json_start
        self.object = _ValueSpan(json_start)
        self.arguments = self.object if payload_is_arguments else None
        self.sent_open = False
        self.sent_arguments = 0
        self.quoted = False        # arguments arrived as a JSON-encoded string, not an object


class ToolCallStream:
    """Turns raw model text into content and tool-call deltas, incrementally.

    Fed the whole decoded text each time (which is what the sink already has), and keeps its own
    cursor, so it never re-interprets what it has already emitted.

    ``hold`` is how many trailing characters are not yet safe to emit as content -- the sink's stop
    sequences and this parser's own start markers, whichever is longer. Without it the first
    characters of a ``<tool_call>`` marker would be streamed as text a moment before the parser
    recognised them, and a client cannot take those back.
    """

    def __init__(self, parser, hold=0):
        self.parser = parser
        self.hold = max(hold, parser.hold_back)
        self.content = ""
        self.calls = []
        self._cursor = 0            # raw text interpreted so far
        self._open = None
        self._region = False        # a marker is open that may introduce more calls
        self._trailing = None       # a call's JSON has closed; waiting for the format's end marker
        self._next_index = 0
        #: Scratch the parser may carry between pushes. The parsers are shared singletons, so
        #: anything that has to be remembered across a generation lives here, with the stream that
        #: the generation belongs to -- see _track_fences, which needs to remember that a code block
        #: is open long after the cursor has streamed past its opening fence.
        self._state = {}

    @property
    def saw_tool_call(self):
        return bool(self.calls) or self._open is not None

    def push(self, text):
        """Interpret whatever has arrived. Returns the deltas it produced, in order."""
        return self._advance(text, final=False)

    def finish(self, text):
        """No more text is coming. Flush held content and close anything still open."""
        return self._advance(text, final=True)

    # -- internals -------------------------------------------------------------------------------

    def _advance(self, text, final):
        events = []
        while True:
            if self._open is not None:
                if not self._step_open(text, events, final):
                    return events
                continue

            if self._trailing is not None:
                resumed = self.parser.after_call(text, self._trailing, final)
                if resumed is None:
                    return events   # the end marker has not arrived; emit nothing meanwhile
                self._cursor = max(self._cursor, resumed)
                self._trailing = None
                self._region = True

            found = None
            if self._region:
                found = self.parser.next_in_region(text, self._cursor)
                if found is None:
                    self._cursor = max(self._cursor,
                                       self.parser.after_region(text, self._cursor, final))
                    self._region = False
            if found is None:
                found = self.parser.find_call(text, self._cursor, self._state)

            if found is None:
                self._emit_content(text, self._safe_end(text, final), events)
                return events
            if isinstance(found, _Pending):
                self._emit_content(text, found.content_end, events)
                if not final:
                    return events
                # Out of text with a maybe still open: it was never a call, so it is content.
                self._emit_content(text, len(text), events)
                return events

            self._emit_content(text, found.content_end, events)
            self._open = _OpenCall(self._next_index, self.parser.new_id(), found.json_start,
                                   found.name, self.parser.payload_is_arguments)
            self._next_index += 1
            self._cursor = found.json_start

    def _safe_end(self, text, final):
        return len(text) if final else max(self._cursor, len(text) - self.hold)

    def _emit_content(self, text, end, events):
        if end <= self._cursor:
            return
        chunk = text[self._cursor:end]
        self._cursor = end
        if chunk:
            self.content += chunk
            events.append(ContentDelta(chunk))

    def _step_open(self, text, events, final):
        """Advance the open call. Returns True when it closed and the loop should carry on."""
        call = self._open
        call.object.advance(text)

        if call.arguments is None:
            # Bounded by the call's own object, so a call with no arguments cannot reach forward and
            # adopt the arguments of the next one.
            limit = call.object.end if call.object.end is not None else len(text)
            match = _ARGUMENTS_RE.search(text, call.json_start, limit)
            # The trailing \s* of the pattern is greedy, so a match ending at the end of the text
            # means the value's first character has not arrived. Anchoring the span there would
            # swallow the space that follows the colon into the arguments.
            if match is not None and match.end() < len(text):
                call.arguments = _ValueSpan(match.end())
                # A model that JSON-encodes its arguments into a string cannot have them streamed
                # as they arrive: the fragments would be escaped text, and unescaping is only
                # possible once the string is closed. Rare, and correctness wins over latency.
                call.quoted = text[match.end()] == '"'
        if call.arguments is not None:
            call.arguments.advance(text)

        if call.name is None:
            call.name = self._read_name(text, call)
        if call.name is not None and not call.sent_open:
            call.sent_open = True
            events.append(ToolCallDelta(index=call.index, id=call.id, name=call.name,
                                        arguments=""))

        if call.sent_open and call.arguments is not None and not call.quoted:
            available = call.arguments.slice(text)
            if len(available) > call.sent_arguments:
                events.append(ToolCallDelta(index=call.index,
                                            arguments=available[call.sent_arguments:]))
                call.sent_arguments = len(available)

        if call.object.end is None and not final:
            return False
        return self._close(text, call, events, final)

    def _read_name(self, text, call):
        """The name, once its closing quote has arrived.

        Searched only outside the arguments value, so an argument that happens to be called "name"
        -- ``{"name": "lookup", "arguments": {"name": "Paris"}}`` -- cannot be mistaken for it.
        """
        limit = call.object.end if call.object.end is not None else len(text)
        window = text[call.json_start:call.arguments.start] if call.arguments \
            else text[call.json_start:limit]
        match = _NAME_RE.search(window)
        if match is None and call.arguments is not None and call.arguments.end is not None:
            match = _NAME_RE.search(text, call.arguments.end, limit)
        if match is None:
            return None
        try:
            return json.loads(f'"{match.group(1)}"')
        except ValueError:
            return match.group(1)

    def _close(self, text, call, events, final):
        """Finish the open call, emitting whatever is still owed."""
        arguments = ""
        if call.arguments is not None:
            arguments = call.arguments.slice(text)
            if call.quoted:
                arguments = _unquote(arguments)

        end = call.object.end if call.object.end is not None else len(text)
        self._open = None

        if call.name is None:
            # It never named a function. Nothing has been emitted for it -- the opening delta waits
            # on the name for exactly this reason -- so hand the text back as content rather than
            # inventing a call with nothing to call.
            self._emit_content(text, end, events)
            self._next_index -= 1
            return end > self._cursor or not final
        if not call.sent_open:
            call.sent_open = True
            events.append(ToolCallDelta(index=call.index, id=call.id, name=call.name, arguments=""))
        if len(arguments) > call.sent_arguments:
            events.append(ToolCallDelta(index=call.index,
                                        arguments=arguments[call.sent_arguments:]))
            call.sent_arguments = len(arguments)

        self.calls.append(ToolCall(index=call.index, id=call.id, name=call.name,
                                   arguments=arguments or "{}"))
        self._cursor = max(self._cursor, end)
        self._trailing = end
        return True


def _unquote(raw):
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    return value if isinstance(value, str) else raw


# ---- the request side ---------------------------------------------------------------------------------

TOOL_CHOICE_NONE = "none"
TOOL_CHOICE_AUTO = "auto"
TOOL_CHOICE_REQUIRED = "required"


@dataclass
class ToolSetup:
    """What a request's `tools` and `tool_choice` mean for the prompt and for parsing."""

    #: Tool definitions to render into the prompt, or None to render none.
    tools: Optional[List[dict]] = None
    #: Whether to read tool calls back out of the reply.
    parse: bool = False
    #: Set when the client asked for something this server cannot enforce, so it can say so once.
    unenforced: str = ""


def resolve_tools(request):
    """Turn `tools` and `tool_choice` into a rendering and parsing decision.

    Where a choice cannot be honoured it is approximated and named, never silently ignored. Nothing
    here constrains decoding, so "required" and a named function are requests to the model rather
    than guarantees about it -- the honest thing is to bias the prompt and say that is what happened.
    """
    tools = list(getattr(request, "tools", None) or ())
    choice = getattr(request, "tool_choice", None)
    if not tools:
        return ToolSetup()

    if choice == TOOL_CHOICE_NONE:
        # Rendering the tools while forbidding their use invites a call this server would then have
        # to hand back as raw markers. Withholding them is what reliably produces prose, which is
        # what the caller asked for.
        return ToolSetup(tools=None, parse=False)

    if isinstance(choice, dict):
        wanted = (choice.get("function") or {}).get("name")
        picked = [tool for tool in tools
                  if (tool.get("function") or {}).get("name") == wanted]
        if picked:
            return ToolSetup(tools=picked, parse=True,
                             unenforced=f"tool_choice named {wanted!r}; only that tool is offered "
                                        f"to the model, but nothing constrains decoding so it may "
                                        f"still answer in prose")
        return ToolSetup(tools=tools, parse=True,
                         unenforced=f"tool_choice named {wanted!r}, which is not in tools")

    if choice == TOOL_CHOICE_REQUIRED:
        return ToolSetup(tools=tools, parse=True,
                         unenforced="tool_choice=required is not enforced: this server does not "
                                    "constrain decoding, so the model may still answer in prose")
    return ToolSetup(tools=tools, parse=True)


def render_message(message):
    """One request message in the shape a chat template expects.

    The arguments conversion is the round-trip bug almost every implementation ships. OpenAI's wire
    format makes ``arguments`` a JSON *string*, but every chat template that renders a tool call
    pipes it through Jinja's ``tojson`` -- Qwen, Hermes, Llama and Mistral all do. Handed a string,
    ``tojson`` quotes and escapes it, so the prompt gets
    ``"arguments": "{\\"city\\": \\"Paris\\"}"`` where the model was trained on
    ``"arguments": {"city": "Paris"}``. The model sees a shape it has never seen on the one turn
    that matters, and the second call in a conversation degrades for no visible reason. Decoding it
    back to an object is what makes a multi-turn agentic exchange work.
    """
    content = message.text()
    rendered = {"role": message.role, "content": content}
    if message.name:
        rendered["name"] = message.name
    if message.tool_call_id:
        rendered["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        rendered["tool_calls"] = [
            {"id": call.id, "type": call.type,
             "function": {"name": call.function.name,
                          "arguments": _as_object(call.function.arguments)}}
            for call in message.tool_calls]
    return rendered


def _as_object(arguments):
    if not isinstance(arguments, str):
        return arguments
    try:
        decoded = json.loads(arguments)
    except ValueError:
        # Not JSON at all. Pass it through: a template that concatenates rather than serialising
        # still renders something, and inventing an object here would be a lie about what was sent.
        return arguments
    return decoded if isinstance(decoded, (dict, list)) else arguments
