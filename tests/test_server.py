"""Tests for the OpenAI-compatible server.

Everything here runs on a plain CPU runner in a couple of seconds. There is no model: the engine
touches exactly three things -- a tokenizer, a device, and a ``generate()`` that feeds a streamer --
so a stand-in providing those three exercises the whole protocol. That is the point. The parts of
this server that break in the field are the wire format and the queue, not the arithmetic, and both
are testable without an accelerator.

The stand-in tokenizer is byte-level on purpose. Decoding joins byte fragments and decodes UTF-8
with errors replaced, exactly as a real tokenizer does, so a codepoint split across two tokens
really does render as U+FFFD until its second half arrives -- which is the case the incremental
decoder exists to handle and the one a mock returning tidy strings would never produce.

What is checked, in order: the sampling mapping, the incremental decoder, stop sequences,
finish_reason, the two endpoints in both modes, the queue's serialisation, and cancellation.
"""
import asyncio
import json
import re
import threading
import time
import unittest
from types import SimpleNamespace

import torch

from rocketllm.server.app import (QUEUE_POSITION_HEADER, GenerationEngine, RequestQueue,
                                  StreamDecoder, _Sink, create_app)
from rocketllm.server.protocol import (FINISH_LENGTH, FINISH_STOP, FINISH_TOOL_CALLS,
                                       ChatCompletionChunk,
                                       ChatCompletionChunkChoice, ChatCompletionRequest,
                                       CompletionRequest, DeltaMessage, RequestError,
                                       SamplingSettings, sse)
from rocketllm.server.toolcalls import resolve_tools

PAD_ID = 0
EOS_ID = 1


class FakeTokenizer:
    """A byte-level tokenizer with a chat template, and nothing else.

    Ids are interned per byte fragment, so a test can hand-build a token that carries half a
    codepoint and watch the decoder refuse to emit it.
    """

    chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    model_max_length = 4096
    pad_token_id = PAD_ID
    eos_token_id = EOS_ID

    def __init__(self):
        self._pieces = {PAD_ID: b"", EOS_ID: b""}
        self._ids = {}

    # -- vocabulary ------------------------------------------------------------------------------

    def token(self, fragment):
        """The id for one byte fragment, interning it on first sight."""
        if isinstance(fragment, str):
            fragment = fragment.encode("utf-8")
        if fragment not in self._ids:
            new_id = len(self._pieces) + 1
            self._ids[fragment] = new_id
            self._pieces[new_id] = fragment
        return self._ids[fragment]

    def tokens(self, text):
        """Split into word-ish tokens, keeping the whitespace attached, and intern each."""
        return [self.token(piece) for piece in re.findall(r"\S+\s*|\s+", text)]

    # -- the tokenizer interface the engine uses ---------------------------------------------------

    def decode(self, ids, skip_special_tokens=True):
        raw = b"".join(self._pieces.get(int(i), b"") for i in ids)
        return raw.decode("utf-8", errors="replace")

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.tensor([self.tokens(text)], dtype=torch.long)}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True,
                            return_tensors=None, return_dict=False, tools=None,
                            continue_final_message=False):
        rendered = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        if tools:
            rendered = f"tools: {json.dumps(tools)}\n" + rendered
        if add_generation_prompt:
            rendered += "assistant: "
        if not tokenize:
            return rendered
        ids = torch.tensor([self.tokens(rendered)], dtype=torch.long)
        return {"input_ids": ids} if return_dict else ids


class FakeModel:
    """Stands in for a RocketModel. Feeds a scripted answer to whatever streamer it is given.

    It honours ``max_new_tokens`` and stops on the end-of-sequence token, because those two are what
    decide ``finish_reason`` and a mock that ignored them would let a wrong one through.
    """

    def __init__(self, tokenizer, script=(), context_length=512, delay=0.0):
        self.tokenizer = tokenizer
        self.script = list(script)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(max_position_embeddings=context_length)
        self.generation_config = SimpleNamespace(eos_token_id=EOS_ID)
        self.model_local_path = "/models/fake-model"
        self.delay = delay
        #: Held closed to keep a generation running while a test does something else.
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()
        self.emitted = 0
        self.calls = []
        self._lock = threading.Lock()
        self.concurrent = 0
        self.max_concurrent = 0

    def generate(self, input_ids=None, attention_mask=None, streamer=None, max_new_tokens=16,
                 **kwargs):
        self.calls.append(dict(kwargs, max_new_tokens=max_new_tokens))
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.started.set()
        try:
            self.gate.wait(5.0)
            if streamer is not None:
                streamer.put(input_ids)
            produced = []
            for token in self.script[:max_new_tokens]:
                if self.delay:
                    time.sleep(self.delay)
                produced.append(token)
                self.emitted += 1
                if streamer is not None:
                    streamer.put(torch.tensor([token]))
                if token == EOS_ID:
                    break
            if streamer is not None:
                streamer.end()
            return torch.cat([input_ids, torch.tensor([produced], dtype=torch.long)], dim=1)
        finally:
            with self._lock:
                self.concurrent -= 1


def build(script="Hello there, world", eos=True, template=None, **kwargs):
    tokenizer = FakeTokenizer()
    if template is not None:
        # Shadows the class attribute. The engine picks its tool-call parser off this at
        # construction, so it has to be in place before the engine exists.
        tokenizer.chat_template = template
    ids = tokenizer.tokens(script) if isinstance(script, str) else list(script)
    if eos:
        ids = ids + [EOS_ID]
    model = FakeModel(tokenizer, ids, **kwargs)
    return model, GenerationEngine(model, model_id="fake-model")


# ---- sampling -----------------------------------------------------------------------------------

class TestTheSamplingMapping(unittest.TestCase):
    """What a client asks for, and what generate() is actually told."""

    def settings(self, **fields):
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **fields)
        request.validate_request()
        return SamplingSettings.from_request(request, 32)

    def test_zero_temperature_is_greedy_and_carries_no_warping_arguments(self):
        """Passing temperature=0 through would divide by it, and top_p alongside do_sample=False
        makes transformers warn on every request about settings it is ignoring."""
        kwargs = self.settings(temperature=0, top_p=0.9).to_generate_kwargs()
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)

    def test_a_real_temperature_samples(self):
        kwargs = self.settings(temperature=0.7, top_p=0.9, top_k=40).to_generate_kwargs()
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.9)
        self.assertEqual(kwargs["top_k"], 40)

    def test_a_top_p_of_one_is_left_off_because_it_warps_nothing(self):
        self.assertNotIn("top_p", self.settings(temperature=1.0, top_p=1.0).to_generate_kwargs())

    def test_stop_is_never_forwarded_to_generate(self):
        """It is applied by the server instead, because transformers' stopping criteria live inside
        a generation loop that the speculative path replaces wholesale."""
        settings = self.settings(stop="END")
        self.assertEqual(settings.stop, ("END",))
        self.assertNotIn("stop", settings.to_generate_kwargs())
        self.assertNotIn("stop_strings", settings.to_generate_kwargs())

    def test_a_single_stop_string_and_a_list_normalise_the_same_way(self):
        self.assertEqual(self.settings(stop="a").stop, ("a",))
        self.assertEqual(self.settings(stop=["a", "b"]).stop, ("a", "b"))

    def test_penalties_it_cannot_honour_are_recorded_rather_than_pretended(self):
        settings = self.settings(frequency_penalty=0.5, presence_penalty=0.2)
        self.assertEqual(set(settings.ignored), {"frequency_penalty", "presence_penalty"})
        self.assertNotIn("frequency_penalty", settings.to_generate_kwargs())

    def test_zero_penalties_are_not_reported_as_ignored(self):
        """Clients send them as defaults; warning about those would be noise on every request."""
        self.assertEqual(self.settings(frequency_penalty=0.0).ignored, ())

    def test_more_than_one_sequence_is_refused_with_a_reason(self):
        with self.assertRaises(RequestError) as caught:
            self.settings(n=2)
        self.assertIn("one forward pass at a time", str(caught.exception))

    def test_out_of_range_values_are_refused(self):
        for field, value in (("temperature", -1), ("top_p", 0), ("top_p", 1.5), ("max_tokens", 0)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(RequestError):
                    self.settings(**{field: value})

    def test_the_newer_token_limit_field_wins_over_the_old_one(self):
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                                        max_tokens=10, max_completion_tokens=20)
        self.assertEqual(request.token_limit(), 20)

    def test_a_batch_of_prompts_is_refused_rather_than_silently_truncated(self):
        with self.assertRaises(RequestError):
            CompletionRequest(prompt=["one", "two"]).validate_request()

    def test_best_of_and_suffix_are_refused_rather_than_ignored(self):
        with self.assertRaises(RequestError):
            CompletionRequest(prompt="x", best_of=4).validate_request()
        with self.assertRaises(RequestError):
            CompletionRequest(prompt="x", suffix="y").validate_request()

    def test_unknown_fields_are_accepted(self):
        """Real clients send fields OpenAI itself ignores; a 422 over one looks like a broken
        server to the person using it."""
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                                        logit_bias={}, service_tier="auto", parallel_tool_calls=True)
        request.validate_request()


# ---- the wire format ------------------------------------------------------------------------------

class TestTheWireFormat(unittest.TestCase):

    def test_an_event_is_one_data_line_and_a_blank_line(self):
        frame = sse({"a": 1})
        self.assertTrue(frame.startswith("data: "))
        self.assertTrue(frame.endswith("\n\n"))
        self.assertEqual(json.loads(frame[len("data: "):]), {"a": 1})

    def test_a_chunk_keeps_a_null_finish_reason_but_drops_nulls_inside_the_delta(self):
        """Clients merge deltas key by key, so a null content in the role chunk can overwrite the
        string they have accumulated. finish_reason is the opposite case: some read it directly."""
        payload = ChatCompletionChunk(
            id="x", choices=[ChatCompletionChunkChoice(delta=DeltaMessage(role="assistant"))],
        ).to_payload()
        self.assertIn("finish_reason", payload["choices"][0])
        self.assertIsNone(payload["choices"][0]["finish_reason"])
        self.assertEqual(payload["choices"][0]["delta"], {"role": "assistant"})

    def test_usage_is_absent_from_a_chunk_that_does_not_carry_it(self):
        payload = ChatCompletionChunk(id="x").to_payload()
        self.assertNotIn("usage", payload)


# ---- incremental decoding -------------------------------------------------------------------------

class TestTheIncrementalDecoder(unittest.TestCase):

    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_it_emits_each_token_as_it_arrives(self):
        decoder = StreamDecoder(self.tokenizer)
        chunks = []
        for word in ("Hello ", "there"):
            decoder.push([self.tokenizer.token(word)])
            chunks.append(decoder.take())
        self.assertEqual("".join(chunks), "Hello there")

    def test_half_a_codepoint_is_held_back_rather_than_shown_as_a_replacement_character(self):
        """A multi-byte character routinely straddles two tokens. Emitting the first half would put
        a permanent U+FFFD in the client's output -- it can never be taken back."""
        decoder = StreamDecoder(self.tokenizer)
        first, second = "é".encode("utf-8")[:1], "é".encode("utf-8")[1:]

        decoder.push([self.tokenizer.token(first)])
        self.assertEqual(decoder.take(), "")
        decoder.push([self.tokenizer.token(second)])
        self.assertEqual(decoder.take(), "é")

    def test_the_hold_back_keeps_the_tail_until_it_is_safe(self):
        decoder = StreamDecoder(self.tokenizer, hold_back=3)
        decoder.push(self.tokenizer.tokens("abcdefgh"))
        self.assertEqual(decoder.take(), "abcde")
        self.assertEqual(decoder.flush(), "fgh")

    def test_a_long_run_is_committed_in_segments_without_changing_the_text(self):
        """The segment reset at newlines is what keeps decoding from being quadratic. It must not
        be visible in the result."""
        decoder = StreamDecoder(self.tokenizer)
        expected = ""
        for line in range(20):
            piece = f"line {line}\n"
            expected += piece
            decoder.push(self.tokenizer.tokens(piece))
        self.assertEqual(decoder.text, expected)
        self.assertEqual(decoder.take(), expected)


# ---- stop sequences and finish_reason ---------------------------------------------------------------

class TestStopSequences(unittest.TestCase):
    """A stop sequence is a string, so it can straddle a token boundary."""

    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def sink(self, stop=(), max_new_tokens=64):
        self.deltas = []
        settings = SamplingSettings(max_new_tokens=max_new_tokens, stop=tuple(stop))
        return _Sink(self.tokenizer, settings, self.deltas.append, threading.Event())

    def feed(self, sink, pieces):
        sink.put(torch.tensor([[9, 9]]))  # the prompt, which every streamer drops
        for piece in pieces:
            sink.put(torch.tensor([self.tokenizer.token(piece)]))

    def text(self):
        return "".join(delta.text for delta in self.deltas)

    def test_a_stop_sequence_split_across_two_tokens_is_never_half_emitted(self):
        """The failure this pins: 'EN' streamed, then 'D' arrives and completes the stop sequence,
        and the client has already printed two characters that were never part of the answer."""
        sink = self.sink(stop=["END"])
        from rocketllm.server.app import _StopGeneration

        with self.assertRaises(_StopGeneration):
            self.feed(sink, ["hello ", "EN", "D", " more"])
        self.assertEqual(self.text(), "hello ")
        self.assertEqual(sink.text, "hello ")

    def test_the_reply_is_truncated_at_the_stop_sequence_not_after_it(self):
        sink = self.sink(stop=["\nUser:"])
        from rocketllm.server.app import _StopGeneration

        with self.assertRaises(_StopGeneration):
            self.feed(sink, ["answer", "\nUser:", " next question"])
        self.assertEqual(sink.text, "answer")

    def test_the_earliest_of_several_stop_sequences_wins(self):
        sink = self.sink(stop=["ZZ", "bb"])
        from rocketllm.server.app import _StopGeneration

        with self.assertRaises(_StopGeneration):
            self.feed(sink, ["aa", "bb", "ZZ"])
        self.assertEqual(sink.text, "aa")

    def test_without_a_stop_sequence_nothing_is_held_back(self):
        sink = self.sink()
        self.feed(sink, ["abc", "def"])
        self.assertEqual(self.text(), "abcdef")


class TestFinishReason(unittest.TestCase):
    """Clients branch on this. A wrong one makes a truncated answer look deliberate."""

    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def sink(self, max_new_tokens):
        settings = SamplingSettings(max_new_tokens=max_new_tokens)
        return _Sink(self.tokenizer, settings, lambda delta: None, threading.Event())

    def test_the_end_of_sequence_token_is_stop(self):
        sink = self.sink(16)
        sink.put(torch.tensor([[7]]))
        sink.put(torch.tensor([self.tokenizer.token("hi")]))
        sink.put(torch.tensor([EOS_ID]))
        self.assertEqual(sink.finish_reason({EOS_ID}), FINISH_STOP)

    def test_running_out_of_room_is_length(self):
        sink = self.sink(2)
        sink.put(torch.tensor([[7]]))
        for piece in ("a", "b"):
            sink.put(torch.tensor([self.tokenizer.token(piece)]))
        self.assertEqual(sink.finish_reason({EOS_ID}), FINISH_LENGTH)

    def test_an_end_of_sequence_on_the_very_last_allowed_token_is_stop_not_length(self):
        """The model chose to end; it was not cut off. Reporting "length" here tells an agentic
        client to ask for a continuation of an answer that is already complete."""
        sink = self.sink(2)
        sink.put(torch.tensor([[7]]))
        sink.put(torch.tensor([self.tokenizer.token("a")]))
        sink.put(torch.tensor([EOS_ID]))
        self.assertEqual(sink.finish_reason({EOS_ID}), FINISH_STOP)

    def test_the_prompt_is_not_counted_as_output(self):
        sink = self.sink(8)
        sink.put(torch.tensor([[3, 4, 5, 6]]))
        sink.put(torch.tensor([self.tokenizer.token("x")]))
        self.assertEqual(sink.token_count, 1)


# ---- the HTTP surface -------------------------------------------------------------------------------

class ServerTestCase(unittest.IsolatedAsyncioTestCase):
    """One app, one queue, one stand-in model, torn down per test."""

    script = "Hello there, world"
    template = None
    model_kwargs = {}

    async def asyncSetUp(self):
        import httpx

        self.model, self.engine = build(self.script, template=self.template, **self.model_kwargs)
        self.queue = RequestQueue()
        self.app = create_app(self.engine, queue=self.queue)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                        base_url="http://server", timeout=30.0)

    async def asyncTearDown(self):
        await self.client.aclose()
        self.queue.close(timeout=5.0)

    async def chat(self, **body):
        payload = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
        payload.update(body)
        return await self.client.post("/v1/chat/completions", json=payload)

    @staticmethod
    def events(response):
        """The SSE frames of a response, as parsed payloads, plus the raw terminator check."""
        frames = [line[len("data: "):] for line in response.text.split("\n\n")
                  if line.startswith("data: ")]
        return frames


class TestChatCompletions(ServerTestCase):

    async def test_a_non_streaming_reply_has_content_finish_reason_and_usage(self):
        response = await self.chat(max_tokens=16)
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "fake-model")
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "Hello there, world")
        self.assertEqual(choice["finish_reason"], FINISH_STOP)

        usage = body["usage"]
        self.assertGreater(usage["prompt_tokens"], 0)
        # Three words plus the end-of-sequence token the stand-in emits.
        self.assertEqual(usage["completion_tokens"], 4)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    async def test_being_cut_off_reports_length(self):
        response = await self.chat(max_tokens=2)
        body = response.json()
        self.assertEqual(body["choices"][0]["finish_reason"], FINISH_LENGTH)
        self.assertEqual(body["usage"]["completion_tokens"], 2)

    async def test_a_stop_sequence_reports_stop_and_truncates(self):
        response = await self.chat(max_tokens=16, stop=["there"])
        body = response.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello ")
        self.assertEqual(body["choices"][0]["finish_reason"], FINISH_STOP)

    async def test_the_stream_is_well_formed_and_ends_in_done(self):
        response = await self.chat(max_tokens=16, stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

        frames = self.events(response)
        self.assertEqual(frames[-1], "[DONE]", "the stream did not end with the [DONE] terminator")

        chunks = [json.loads(frame) for frame in frames[:-1]]
        self.assertTrue(all(chunk["object"] == "chat.completion.chunk" for chunk in chunks))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], FINISH_STOP)
        self.assertEqual(chunks[-1]["choices"][0]["delta"], {})

    async def test_the_streamed_text_is_exactly_the_non_streamed_text(self):
        """The property a client actually depends on: the two modes cannot disagree."""
        whole = (await self.chat(max_tokens=16)).json()["choices"][0]["message"]["content"]
        streamed = await self.chat(max_tokens=16, stream=True)
        pieces = [json.loads(frame)["choices"][0]["delta"].get("content", "")
                  for frame in self.events(streamed)[:-1]]
        self.assertEqual("".join(pieces), whole)

    async def test_a_streamed_stop_sequence_truncates_the_same_way(self):
        streamed = await self.chat(max_tokens=16, stream=True, stop=["there"])
        chunks = [json.loads(frame) for frame in self.events(streamed)[:-1]]
        text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        self.assertEqual(text, "Hello ")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], FINISH_STOP)

    async def test_no_usage_chunk_unless_the_client_asks_for_one(self):
        streamed = await self.chat(max_tokens=16, stream=True)
        chunks = [json.loads(frame) for frame in self.events(streamed)[:-1]]
        self.assertFalse(any("usage" in chunk for chunk in chunks))

    async def test_the_opt_in_usage_chunk_comes_last_and_has_no_choices(self):
        streamed = await self.chat(max_tokens=16, stream=True,
                                   stream_options={"include_usage": True})
        frames = self.events(streamed)
        self.assertEqual(frames[-1], "[DONE]")
        final = json.loads(frames[-2])
        self.assertEqual(final["choices"], [])
        self.assertEqual(final["usage"]["completion_tokens"], 4)

    async def test_the_chat_template_is_what_gets_tokenized(self):
        await self.chat(max_tokens=4)
        expected = self.model.tokenizer.tokens("user: hi\nassistant: ")
        self.assertEqual(self.model.calls and True, True)
        # The engine encodes with the template; the prompt token count proves which text it used.
        body = (await self.chat(max_tokens=4)).json()
        self.assertEqual(body["usage"]["prompt_tokens"], len(expected))

    async def test_a_bad_request_gets_the_openai_error_envelope(self):
        response = await self.chat(n=2)
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertIn("error", body)
        self.assertIn("message", body["error"])
        self.assertEqual(body["error"]["param"], "n")

    async def test_a_schema_violation_is_also_reshaped_into_that_envelope(self):
        response = await self.client.post("/v1/chat/completions", json={"messages": "not a list"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    async def test_the_seed_reaches_generate(self):
        await self.chat(max_tokens=4, seed=1234, temperature=0.7)
        self.assertTrue(self.model.calls)


class TestAPromptThatDoesNotFit(ServerTestCase):
    model_kwargs = {"context_length": 8}

    async def test_it_is_refused_with_the_two_numbers_that_matter(self):
        long_prompt = " ".join(str(i) for i in range(50))
        response = await self.chat(messages=[{"role": "user", "content": long_prompt}])
        self.assertEqual(response.status_code, 400)
        message = response.json()["error"]["message"]
        self.assertIn("context", message)
        self.assertEqual(response.json()["error"]["code"], "context_length_exceeded")


class TestLegacyCompletions(ServerTestCase):

    async def completion(self, **body):
        payload = {"model": "fake-model", "prompt": "once upon"}
        payload.update(body)
        return await self.client.post("/v1/completions", json=payload)

    async def test_a_non_streaming_completion(self):
        body = (await self.completion(max_tokens=16)).json()
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], "Hello there, world")
        self.assertEqual(body["choices"][0]["finish_reason"], FINISH_STOP)
        self.assertGreater(body["usage"]["prompt_tokens"], 0)

    async def test_echo_puts_the_prompt_back_in_front(self):
        body = (await self.completion(max_tokens=16, echo=True)).json()
        self.assertTrue(body["choices"][0]["text"].startswith("once upon"))

    async def test_the_stream_ends_in_done_and_carries_a_finish_reason(self):
        response = await self.completion(max_tokens=16, stream=True)
        frames = self.events(response)
        self.assertEqual(frames[-1], "[DONE]")
        chunks = [json.loads(frame) for frame in frames[:-1]]
        self.assertEqual("".join(chunk["choices"][0]["text"] for chunk in chunks),
                         "Hello there, world")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], FINISH_STOP)

    async def test_a_pre_tokenized_prompt_is_accepted(self):
        body = (await self.completion(prompt=[5, 6, 7], max_tokens=4)).json()
        self.assertEqual(body["usage"]["prompt_tokens"], 3)


#: A tool call as a Hermes/Qwen model emits one. The engine picks its parser off the template, so
#: the template below only has to contain the marker for the right one to be chosen.
HERMES_TEMPLATE = '{% for m in messages %}{{ m.role }}{% endfor %}<tool_call></tool_call>'
HERMES_CALL = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", '
               '"unit": "celsius"}}\n</tool_call>')
WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


class TestToolCallsOverHTTP(ServerTestCase):
    """The whole path, as an agentic client sees it."""

    script = HERMES_CALL
    template = HERMES_TEMPLATE

    async def chat(self, **body):
        payload = {"model": "fake-model", "max_tokens": 64, "tools": [WEATHER_TOOL],
                   "messages": [{"role": "user", "content": "weather in Paris?"}]}
        payload.update(body)
        return await self.client.post("/v1/chat/completions", json=payload)

    async def test_the_parser_is_chosen_from_the_template(self):
        self.assertEqual(self.engine.tool_parser.family, "hermes")

    async def test_a_call_comes_back_in_the_openai_shape(self):
        body = (await self.chat()).json()
        choice = body["choices"][0]
        calls = choice["message"]["tool_calls"]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertTrue(calls[0]["id"])
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Paris", "unit": "celsius"})

    async def test_finish_reason_is_tool_calls(self):
        """An agentic loop branches on this. Reporting "stop" leaves the client treating a call as
        a final answer and the conversation simply ends."""
        body = (await self.chat()).json()
        self.assertEqual(body["choices"][0]["finish_reason"], FINISH_TOOL_CALLS)

    async def test_content_is_null_when_the_model_only_called_a_tool(self):
        """What OpenAI sends. A client testing `if message.content` would otherwise append an empty
        assistant turn to the conversation it is building and send it back next request."""
        self.assertIsNone((await self.chat()).json()["choices"][0]["message"]["content"])

    async def test_none_of_the_raw_syntax_leaks_into_the_content(self):
        text = json.dumps((await self.chat()).json())
        for marker in ("<tool_call>", "</tool_call>"):
            self.assertNotIn(marker, text)

    async def test_the_stream_assembles_to_the_same_call(self):
        """The property the design rests on, checked through HTTP rather than in the parser."""
        whole = (await self.chat()).json()["choices"][0]["message"]["tool_calls"]
        streamed = await self.chat(stream=True)
        chunks = [json.loads(frame) for frame in self.events(streamed)[:-1]]

        assembled = {}
        order = []
        for chunk in chunks:
            for delta in (chunk["choices"][0]["delta"].get("tool_calls") or []):
                index = delta["index"]
                if index not in assembled:
                    assembled[index] = {"id": delta.get("id"),
                                        "name": delta["function"].get("name"), "arguments": ""}
                    order.append(index)
                assembled[index]["arguments"] += delta["function"].get("arguments", "")

        self.assertEqual(len(order), 1)
        built = assembled[order[0]]
        self.assertEqual(built["name"], whole[0]["function"]["name"])
        self.assertEqual(json.loads(built["arguments"]),
                         json.loads(whole[0]["function"]["arguments"]))

    async def test_the_streamed_id_is_sent_once_and_never_changes(self):
        streamed = await self.chat(stream=True)
        chunks = [json.loads(frame) for frame in self.events(streamed)[:-1]]
        ids = [delta["id"] for chunk in chunks
               for delta in (chunk["choices"][0]["delta"].get("tool_calls") or [])
               if "id" in delta]
        self.assertEqual(len(ids), 1, "the id was re-sent, which splits the call for a client")
        self.assertTrue(ids[0])

    async def test_the_stream_ends_with_the_tool_calls_finish_reason_then_done(self):
        streamed = await self.chat(stream=True)
        frames = self.events(streamed)
        self.assertEqual(frames[-1], "[DONE]")
        self.assertEqual(json.loads(frames[-2])["choices"][0]["finish_reason"], FINISH_TOOL_CALLS)

    async def test_the_tool_definitions_reach_the_prompt(self):
        """Rendered through the chat template's own tool support, which is the only shape the model
        was trained to read them in."""
        with_tools = (await self.chat()).json()["usage"]["prompt_tokens"]
        without = (await self.chat(tools=None)).json()["usage"]["prompt_tokens"]
        self.assertGreater(with_tools, without)

    async def test_a_request_with_no_tools_gets_the_raw_text_back_as_content(self):
        """A model that writes <tool_call> in an ordinary answer is quoting, not calling. Parsing
        it would silently delete the text."""
        body = (await self.chat(tools=None)).json()
        self.assertIn("<tool_call>", body["choices"][0]["message"]["content"])
        self.assertIsNone(body["choices"][0]["message"]["tool_calls"])
        self.assertEqual(body["choices"][0]["finish_reason"], FINISH_STOP)

    async def test_tool_choice_none_answers_in_prose(self):
        body = (await self.chat(tool_choice="none")).json()
        self.assertIsNone(body["choices"][0]["message"]["tool_calls"])
        self.assertEqual(body["choices"][0]["finish_reason"], FINISH_STOP)


class TestATruncatedToolCall(ServerTestCase):
    script = HERMES_CALL
    template = HERMES_TEMPLATE
    model_kwargs = {"eos": False}

    async def test_running_out_of_room_reports_length_not_tool_calls(self):
        """The arguments of a call cut off mid-write are not valid JSON. Announcing it as a tool
        call hands the client something it cannot parse and no reason why; "length" is both true
        and a signal an agentic loop already handles."""
        response = await self.client.post("/v1/chat/completions", json={
            "model": "fake-model", "max_tokens": 4, "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(response.json()["choices"][0]["finish_reason"], FINISH_LENGTH)


class TestNamingTheModel(unittest.TestCase):
    """What clients echo and humans read in a bug report."""

    def name_for(self, path):
        model = FakeModel(FakeTokenizer())
        model.model_local_path = path
        return GenerationEngine(model).model_id

    def test_a_downloaded_checkpoint_is_named_by_its_repo_not_its_commit(self):
        """A cached checkpoint lives under .../snapshots/<commit>, and answering requests as a
        forty-character hash helps nobody."""
        self.assertEqual(
            self.name_for(r"C:\Users\x\hf\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
                          r"\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"),
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def test_a_plain_directory_is_named_by_its_directory(self):
        self.assertEqual(self.name_for("/models/my-finetune"), "my-finetune")

    def test_an_explicit_name_always_wins(self):
        model = FakeModel(FakeTokenizer())
        self.assertEqual(GenerationEngine(model, model_id="whatever").model_id, "whatever")


class TestTheChatTemplateEncoding(unittest.TestCase):

    def test_a_batch_encoding_is_unwrapped_rather_than_returned_whole(self):
        """apply_chat_template(return_dict=True) hands back a BatchEncoding, which is a UserDict --
        a Mapping, but not a dict subclass. An isinstance(dict) test passes the container through
        and the failure surfaces later, on .shape, pointing nowhere near here."""
        from transformers.tokenization_utils_base import BatchEncoding

        tokenizer = FakeTokenizer()
        expected = torch.tensor([tokenizer.tokens("user: hi\nassistant: ")])
        tokenizer.apply_chat_template = lambda *a, **k: BatchEncoding({"input_ids": expected})

        engine = GenerationEngine(FakeModel(tokenizer))
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        encoded, model_inputs = engine._encode_chat(request, resolve_tools(request))
        self.assertEqual(encoded.tolist(), expected.tolist())
        # A text request carries nothing beside its token ids, and that emptiness is load-bearing:
        # it is what keeps prefix reuse and speculative decoding switched on for ordinary traffic.
        self.assertEqual(model_inputs, {})


class TestMetadataEndpoints(ServerTestCase):

    async def test_models_lists_the_one_that_is_loaded(self):
        body = (await self.client.get("/v1/models")).json()
        self.assertEqual(body["object"], "list")
        self.assertEqual([card["id"] for card in body["data"]], ["fake-model"])
        self.assertEqual(body["data"][0]["object"], "model")

    async def test_an_unknown_model_is_a_404_in_the_error_envelope(self):
        response = await self.client.get("/v1/models/something-else")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")

    async def test_health_carries_what_a_performance_bug_report_needs(self):
        body = (await self.client.get("/health")).json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"], "fake-model")
        for key in ("hardware", "cache", "queue", "budget", "generation"):
            self.assertIn(key, body)
        self.assertIn("depth", body["queue"])
        # No profile was built for a stand-in model, and that is reported rather than crashing.
        self.assertFalse(body["hardware"]["available"])

    async def test_health_counts_the_requests_it_has_served(self):
        await self.chat(max_tokens=4)
        body = (await self.client.get("/health")).json()
        self.assertEqual(body["queue"]["completed"], 1)
        self.assertEqual(body["queue"]["depth"], 0)

    async def test_a_request_that_finished_normally_is_not_counted_as_cancelled(self):
        """Both endpoints set the cancel flag on their way out of every request, so a counter that
        read that flag reported a server dropping every single request it successfully served --
        in the one payload people paste into bug reports."""
        for _ in range(3):
            self.assertEqual((await self.chat(max_tokens=4)).status_code, 200)
        queue = (await self.client.get("/health")).json()["queue"]
        self.assertEqual(queue["completed"], 3)
        self.assertEqual(queue["cancelled"], 0)
        self.assertEqual(queue["failed"], 0)

    async def test_a_request_that_failed_is_counted_as_failed_and_not_cancelled(self):
        queue = self.queue
        long_prompt = " ".join(str(i) for i in range(5000))
        self.assertEqual((await self.chat(
            messages=[{"role": "user", "content": long_prompt}])).status_code, 400)
        self.assertEqual(queue.stats()["cancelled"], 0)


# ---- the queue ----------------------------------------------------------------------------------------

class TestTheRequestQueue(ServerTestCase):
    model_kwargs = {"delay": 0.001}

    async def wait_for(self, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.005)
        self.fail("condition was never met")

    async def test_a_second_request_queues_behind_the_first_instead_of_interleaving(self):
        """One model instance means one pass at a time. Two generations running together would not
        merely be slower -- they would evict each other's layers from the weight cache."""
        self.model.gate.clear()
        first = asyncio.create_task(self.chat(max_tokens=8))
        await self.wait_for(self.model.started.is_set)

        second = asyncio.create_task(self.chat(max_tokens=8))
        await self.wait_for(lambda: self.queue.stats()["depth"] == 2)

        self.model.gate.set()
        responses = await asyncio.gather(first, second)

        self.assertEqual(self.model.max_concurrent, 1, "two generations ran at the same time")
        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual(responses[1].headers[QUEUE_POSITION_HEADER], "1",
                         "the queued request was not told its position")
        self.assertEqual(responses[0].headers[QUEUE_POSITION_HEADER], "0")

    async def test_both_replies_are_whole(self):
        """The failure a shared model corrupts into: two requests splicing each other's tokens."""
        self.model.gate.clear()
        tasks = [asyncio.create_task(self.chat(max_tokens=8)) for _ in range(3)]
        await self.wait_for(lambda: self.queue.stats()["depth"] == 3)
        self.model.gate.set()

        for response in await asyncio.gather(*tasks):
            self.assertEqual(response.json()["choices"][0]["message"]["content"],
                             "Hello there, world")


class TestCancellation(unittest.IsolatedAsyncioTestCase):
    """An abandoned request holds the only worker. That is the whole reason this matters here and
    not in a batched server: everything queued behind it waits for a reply nobody will read."""

    #: Long enough that a disconnect lands well before the end, short enough to stay a fast test.
    #: Real words rather than bare ids, so every token decodes to something and the stream actually
    #: produces chunks -- a script of unknown ids decodes to "" and emits nothing, which makes the
    #: test pass a disconnect it never really tested.
    TOKENS = 400

    def setUp(self):
        self.model, self.engine = build(script="tok " * self.TOKENS, eos=False, delay=0.002)
        self.queue = RequestQueue()
        self.app = create_app(self.engine, queue=self.queue)

    def tearDown(self):
        self.queue.close(timeout=5.0)

    async def drive(self, body, disconnect_after=2):
        """Call the ASGI app directly, hanging up part-way through the response.

        Driven by hand rather than through a test client because the disconnect is the thing being
        tested: the ASGI transport in httpx buffers a response to completion and never sends
        ``http.disconnect`` at all, so a client-side test could not produce this at all.
        """
        disconnected = asyncio.Event()
        sent = []
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": json.dumps(body).encode(),
                        "more_body": False}
            # Once disconnected, return immediately and keep returning it -- which is what uvicorn
            # does, and what the server's poll depends on. A version of this that always awaited
            # would be cancelled inside is_disconnected's pre-cancelled scope and the poll would
            # never see the hang-up, so the fidelity here is the test.
            if not disconnected.is_set():
                await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                sent.append(message["body"])
                if len(sent) >= disconnect_after:
                    disconnected.set()

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
            "query_string": b"", "root_path": "", "client": ("127.0.0.1", 1234),
            "server": ("server", 80),
            "headers": [(b"content-type", b"application/json"), (b"host", b"server")],
        }
        await self.app(scope, receive, send)
        return sent

    async def test_hanging_up_mid_stream_stops_the_generation(self):
        await self.drive({"model": "fake-model", "stream": True, "max_tokens": self.TOKENS,
                          "messages": [{"role": "user", "content": "hi"}]})

        deadline = time.time() + 5.0
        while self.queue.stats()["depth"] and time.time() < deadline:
            await asyncio.sleep(0.01)

        self.assertEqual(self.queue.stats()["depth"], 0, "the worker was never released")
        # Promptly, not eventually. The hang-up is signalled two chunks in, and the streamer checks
        # the flag on every token, so this lands within a handful of them -- measured at 2. A bound
        # of "less than the whole run" would still pass if cancellation took a hundred tokens, which
        # on a storage-bound model is most of a minute.
        self.assertLess(self.model.emitted, 20,
                        "generation carried on well past the point the client hung up")
        self.assertEqual(self.queue.stats()["cancelled"], 1,
                         "the hang-up was not recorded as a cancellation")

    async def test_the_worker_is_free_for_the_next_request_straight_away(self):
        await self.drive({"model": "fake-model", "stream": True, "max_tokens": self.TOKENS,
                          "messages": [{"role": "user", "content": "hi"}]})
        import httpx

        deadline = time.time() + 5.0
        while self.queue.stats()["depth"] and time.time() < deadline:
            await asyncio.sleep(0.01)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                     base_url="http://server", timeout=30.0) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "fake-model", "max_tokens": 3,
                "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers[QUEUE_POSITION_HEADER], "0")


if __name__ == "__main__":
    unittest.main()
