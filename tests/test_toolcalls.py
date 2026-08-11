"""Tests for reading tool calls out of what models actually emit.

No model anywhere in this file, and none is needed: a parser's whole job is a string in and
structure out, so the interesting inputs are captured raw outputs rather than generated ones. RAW
below is that corpus, written exactly as each family emits it -- including DeepSeek's full-width
delimiters, which are not the ASCII characters they resemble.

The property most of these tests are built around is that **the streamed assembly and the
non-streamed result must be identical**. There is one state machine and two views of it, so a
disagreement is a bug by construction rather than a judgement call. Every corpus entry is therefore
run three ways -- all at once, one character at a time, and split at every boundary -- and the three
are required to agree. One character at a time is not a contrived case: it is what a token-by-token
generation looks like to a parser, and it is where a partially-arrived marker or a half-written JSON
value goes wrong.

The second property is that ids never move. A client assembles arguments by concatenating fragments
keyed on index and id, so an id regenerated mid-stream splits one call into several, each holding a
piece of the arguments and none of them callable.
"""
import json
import unittest

from rocketllm.server.protocol import ChatMessage
from rocketllm.server.toolcalls import (FAMILIES, ContentDelta, GenericParser, HermesParser,
                                        PARSERS, ToolCallDelta, ToolCallStream, render_message,
                                        resolve_tools, select_parser)

# ---- captured raw output ---------------------------------------------------------------------

#: What each family actually puts on the wire, and the parser that reads it. Add a row here when
#: adding a family and every test below covers it.
RAW = {
    "hermes": ("hermes",
               'I will look that up.\n<tool_call>\n'
               '{"name": "get_current_weather", "arguments": {"location": "Paris, France", '
               '"unit": "celsius"}}\n</tool_call>',
               "I will look that up.\n",
               [("get_current_weather",
                 {"location": "Paris, France", "unit": "celsius"})]),

    "hermes_two_calls": ("hermes",
                         '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
                         '</tool_call>\n<tool_call>\n'
                         '{"name": "get_time", "arguments": {"tz": "Europe/Paris"}}\n</tool_call>',
                         "\n",
                         [("get_weather", {"city": "Paris"}),
                          ("get_time", {"tz": "Europe/Paris"})]),

    "hermes_no_arguments": ("hermes",
                            '<tool_call>\n{"name": "list_tools", "arguments": {}}\n</tool_call>',
                            "",
                            [("list_tools", {})]),

    "mistral": ("mistral",
                '[TOOL_CALLS] [{"name": "get_current_weather", '
                '"arguments": {"location": "Paris, France", "format": "celsius"}}]',
                "",
                [("get_current_weather", {"location": "Paris, France", "format": "celsius"})]),

    "mistral_two_calls": ("mistral",
                          '[TOOL_CALLS] [{"name": "a", "arguments": {"x": 1}}, '
                          '{"name": "b", "arguments": {"y": 2}}]',
                          "",
                          [("a", {"x": 1}), ("b", {"y": 2})]),

    "llama_tagged": ("llama",
                     '<|python_tag|>{"name": "get_current_weather", '
                     '"parameters": {"location": "Paris, France"}}',
                     "",
                     [("get_current_weather", {"location": "Paris, France"})]),

    "llama_bare": ("llama",
                   '{"name": "get_current_weather", "parameters": {"location": "Paris, France"}}',
                   "",
                   [("get_current_weather", {"location": "Paris, France"})]),

    "llama_two_calls": ("llama",
                        '<|python_tag|>{"name": "a", "parameters": {"x": 1}}; '
                        '{"name": "b", "parameters": {"y": 2}}',
                        "",
                        [("a", {"x": 1}), ("b", {"y": 2})]),

    "deepseek": ("deepseek",
                 'Sure, let me check.\n'
                 '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>'
                 'get_current_weather\n```json\n{"location": "Paris, France"}\n```'
                 '<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
                 "Sure, let me check.\n",
                 [("get_current_weather", {"location": "Paris, France"})]),

    "generic_fenced": ("generic",
                       'Here is the call:\n```json\n'
                       '{"name": "get_weather", "arguments": {"city": "Paris"}}\n```',
                       "Here is the call:\n",
                       [("get_weather", {"city": "Paris"})]),

    "generic_bare": ("generic",
                     '{"name": "get_weather", "arguments": {"city": "Paris"}}',
                     "",
                     [("get_weather", {"city": "Paris"})]),

    "generic_nested_shape": ("generic",
                             '{"type": "function", "function": {"name": "get_weather", '
                             '"arguments": {"city": "Paris"}}}',
                             "",
                             [("get_weather", {"city": "Paris"})]),

    # The JSON scanner's real test: braces and quotes inside string values must not close the
    # arguments early, and an argument that happens to be called "name" must not be mistaken for
    # the function's.
    "braces_in_strings": ("hermes",
                          '<tool_call>\n{"name": "run_code", "arguments": '
                          '{"code": "if (x) { y(); }", "note": "a \\"quoted\\" brace }"}}\n'
                          '</tool_call>',
                          "",
                          [("run_code", {"code": "if (x) { y(); }",
                                         "note": 'a "quoted" brace }'})]),

    "argument_called_name": ("hermes",
                             '<tool_call>\n{"name": "lookup", "arguments": {"name": "Paris"}}\n'
                             '</tool_call>',
                             "",
                             [("lookup", {"name": "Paris"})]),

    # Some models JSON-encode the arguments into a string instead of nesting an object.
    "arguments_as_string": ("hermes",
                            '<tool_call>\n{"name": "f", "arguments": '
                            '"{\\"city\\": \\"Paris\\"}"}\n</tool_call>',
                            "",
                            [("f", {"city": "Paris"})]),
}

#: Text with no call in it at all. Whatever the parser, every character must come back as content.
PROSE = {
    "braces": 'The set {1, 2, 3} has three elements, and {"a": 1} is a dictionary literal.',
    "code_fence": 'Try this:\n```python\nd = {"a": 1}\nprint(d)\n```\nThat prints a dict.',
    "json_without_a_name": 'The response looks like {"status": "ok", "count": 3} when it works.',
    "plain": "There is nothing to call here; the answer is simply forty-two.",

    # Captured from a real model asked about JSON in Python while tools were offered. The object
    # inside the code block has both a name and an arguments key, so nothing about its shape says
    # it is not a call -- only the fence it sits in does. Before the fence was honoured, this
    # answer came back with its code silently lifted out and served as a tool call.
    "json_shaped_object_in_a_code_block":
        'Here is an example:\n\n```python\nimport json\n\n'
        'data = {"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        'print(json.dumps(data))\n```\nThat prints the object.',
}


def parser_for(family):
    return next(parser for parser in PARSERS if parser.family == family)


def assemble(events):
    """Rebuild what a client would, from the delta stream alone.

    Deliberately written the way a client is: merge on index, take id and name from whichever delta
    carries them, and concatenate arguments. If the server's deltas do not support that, this
    breaks in the same way a real client would.
    """
    content = ""
    calls = {}
    order = []
    for event in events:
        if isinstance(event, ContentDelta):
            content += event.text
            continue
        if event.index not in calls:
            calls[event.index] = {"id": event.id, "name": event.name, "arguments": ""}
            order.append(event.index)
        else:
            if event.id is not None:
                calls[event.index]["id"] = event.id
            if event.name is not None:
                calls[event.index]["name"] = event.name
        calls[event.index]["arguments"] += event.arguments
    return content, [calls[index] for index in order]


def run(family, raw, chunks):
    """Feed `raw` to a fresh stream in the given pieces. Returns (stream, events)."""
    stream = ToolCallStream(parser_for(family))
    events = []
    seen = ""
    for piece in chunks:
        seen += piece
        events.extend(stream.push(seen))
    events.extend(stream.finish(raw))
    return stream, events


def one_shot(raw):
    return [raw]


def per_character(raw):
    return list(raw)


def every_split(raw):
    """Every two-way split, run as separate cases by the caller."""
    return [[raw[:at], raw[at:]] for at in range(len(raw) + 1)]


class TestEveryFamilyParses(unittest.TestCase):
    """The corpus, read whole."""

    def test_the_expected_calls_come_out(self):
        for key, (family, raw, content, expected) in RAW.items():
            with self.subTest(case=key):
                stream, _ = run(family, raw, one_shot(raw))
                self.assertEqual([call.name for call in stream.calls],
                                 [name for name, _ in expected])
                self.assertEqual([json.loads(call.arguments) for call in stream.calls],
                                 [arguments for _, arguments in expected])

    def test_the_content_is_what_the_model_said_and_not_its_syntax(self):
        """Markers, fences and delimiters are the format talking, not the model. None of it may
        reach the client as though it were part of the answer."""
        for key, (family, raw, content, _) in RAW.items():
            with self.subTest(case=key):
                stream, _ = run(family, raw, one_shot(raw))
                self.assertEqual(stream.content, content)

    def test_indexes_are_dense_and_in_order(self):
        for key, (family, raw, _, expected) in RAW.items():
            with self.subTest(case=key):
                stream, _ = run(family, raw, one_shot(raw))
                self.assertEqual([call.index for call in stream.calls], list(range(len(expected))))

    def test_arguments_are_always_valid_json(self):
        for key, (family, raw, _, _) in RAW.items():
            with self.subTest(case=key):
                stream, _ = run(family, raw, one_shot(raw))
                for call in stream.calls:
                    json.loads(call.arguments)   # raises if it is not

    def test_every_family_in_the_registry_has_a_captured_sample(self):
        """A family with no corpus entry is a family nothing here actually tests."""
        covered = {family for family, _, _, _ in RAW.values()}
        self.assertEqual(covered, set(FAMILIES))


class TestStreamingMatchesNonStreaming(unittest.TestCase):
    """The property the whole design rests on: the two response modes cannot disagree."""

    def test_one_character_at_a_time_gives_the_same_answer(self):
        """What a token-by-token generation looks like to the parser, and where a half-arrived
        marker or an unfinished JSON value goes wrong."""
        for key, (family, raw, content, expected) in RAW.items():
            with self.subTest(case=key):
                whole, _ = run(family, raw, one_shot(raw))
                streamed_stream, events = run(family, raw, per_character(raw))
                streamed_content, streamed_calls = assemble(events)

                self.assertEqual(streamed_content, whole.content)
                self.assertEqual([call["name"] for call in streamed_calls],
                                 [call.name for call in whole.calls])
                self.assertEqual([call["arguments"] for call in streamed_calls],
                                 [call.arguments for call in whole.calls])

    def test_it_holds_at_every_possible_split(self):
        """A chunk boundary anywhere -- mid-marker, mid-key, mid-escape -- changes nothing."""
        for key, (family, raw, _, _) in RAW.items():
            whole, _ = run(family, raw, one_shot(raw))
            for chunks in every_split(raw):
                with self.subTest(case=key, split=len(chunks[0])):
                    _, events = run(family, raw, chunks)
                    content, calls = assemble(events)
                    self.assertEqual(content, whole.content)
                    self.assertEqual([(c["name"], c["arguments"]) for c in calls],
                                     [(c.name, c.arguments) for c in whole.calls])

    def test_the_streamed_arguments_concatenate_to_valid_json(self):
        for key, (family, raw, _, expected) in RAW.items():
            with self.subTest(case=key):
                _, events = run(family, raw, per_character(raw))
                _, calls = assemble(events)
                self.assertEqual([json.loads(call["arguments"]) for call in calls],
                                 [arguments for _, arguments in expected])


class TestStreamingDeltaDiscipline(unittest.TestCase):
    """What a client needs from the delta sequence itself."""

    def deltas(self, key):
        family, raw, _, _ = RAW[key]
        _, events = run(family, raw, per_character(raw))
        return [event for event in events if isinstance(event, ToolCallDelta)]

    def test_an_id_is_sent_once_and_never_changes(self):
        """The failure this pins: an id regenerated per chunk makes a client assemble one call into
        several, each with a fragment of the arguments and none of them callable."""
        for key in RAW:
            with self.subTest(case=key):
                seen = {}
                for delta in self.deltas(key):
                    if delta.id is None:
                        continue
                    if delta.index in seen:
                        self.fail(f"index {delta.index} was given an id twice")
                    seen[delta.index] = delta.id
                self.assertTrue(all(seen.values()), "a call was opened with an empty id")
                self.assertEqual(len(set(seen.values())), len(seen), "two calls share an id")

    def test_the_opening_delta_carries_the_id_and_the_name_together(self):
        """A delta with an id but no name would have a client create a call it cannot label, and
        nothing later re-sends the id to correct it."""
        for key in RAW:
            with self.subTest(case=key):
                opened = set()
                for delta in self.deltas(key):
                    if delta.index not in opened:
                        opened.add(delta.index)
                        self.assertIsNotNone(delta.id, "first delta for a call has no id")
                        self.assertIsNotNone(delta.name, "first delta for a call has no name")

    def test_the_name_is_never_sent_twice(self):
        for key in RAW:
            with self.subTest(case=key):
                named = [delta.index for delta in self.deltas(key) if delta.name is not None]
                self.assertEqual(len(named), len(set(named)))

    def test_the_opening_delta_of_a_call_is_a_valid_openai_payload(self):
        payload = self.deltas("hermes")[0].to_payload()
        self.assertEqual(payload["type"], "function")
        self.assertEqual(payload["index"], 0)
        self.assertTrue(payload["id"])
        self.assertEqual(payload["function"]["name"], "get_current_weather")
        self.assertEqual(payload["function"]["arguments"], "")

    def test_a_continuation_delta_carries_no_id(self):
        later = [delta.to_payload() for delta in self.deltas("hermes")[1:]]
        self.assertTrue(later, "the arguments arrived in a single delta; nothing was streamed")
        for payload in later:
            self.assertNotIn("id", payload)
            self.assertNotIn("type", payload)
            self.assertIn("arguments", payload["function"])


class TestProseIsLeftAlone(unittest.TestCase):
    """A parser that eats an answer is worse than one that misses a call."""

    def test_text_with_no_call_comes_back_whole(self):
        for family in FAMILIES:
            for key, raw in PROSE.items():
                with self.subTest(family=family, case=key):
                    stream, _ = run(family, raw, one_shot(raw))
                    self.assertEqual(stream.content, raw)
                    self.assertEqual(stream.calls, [])

    def test_that_holds_when_it_arrives_one_character_at_a_time(self):
        """The generic parser has no marker to key off, so every brace and every fence is a
        candidate it has to hold and then release. This is where it would swallow an answer."""
        for family in FAMILIES:
            for key, raw in PROSE.items():
                with self.subTest(family=family, case=key):
                    _, events = run(family, raw, per_character(raw))
                    content, calls = assemble(events)
                    self.assertEqual(content, raw)
                    self.assertEqual(calls, [])


class TestTruncation(unittest.TestCase):
    """A generation cut off mid-call."""

    def test_a_call_with_no_name_yet_is_given_back_as_content(self):
        """Nothing is emitted for a call until its name arrives, so there is no half-announced call
        to retract -- the text simply turns out to have been content."""
        raw = '<tool_call>\n{"na'
        stream, events = run("hermes", raw, per_character(raw))
        self.assertEqual(stream.calls, [])
        _, calls = assemble(events)
        self.assertEqual(calls, [])

    def test_a_call_cut_off_inside_its_arguments_still_reports_what_arrived(self):
        raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Par'
        stream, _ = run("hermes", raw, per_character(raw))
        self.assertEqual([call.name for call in stream.calls], ["get_weather"])
        self.assertEqual(stream.calls[0].arguments, '{"city": "Par')


class TestParserSelection(unittest.TestCase):
    """Detection reads the template, because that is what states the syntax.

    Not the architecture name: a fine-tune inherits its parent's template under a name nobody has
    seen, and an unreleased model has no name to match at all -- the same reason expert layouts are
    detected from checkpoint structure rather than from `architectures`.
    """

    def select(self, template):
        return select_parser(_Tokenizer(template)).family

    def test_each_family_is_recognised_from_the_syntax_its_template_emits(self):
        self.assertEqual(self.select('{{ "<tool_call>" }}'), "hermes")
        self.assertEqual(self.select('{{ "[TOOL_CALLS]" }}'), "mistral")
        self.assertEqual(self.select('{{ "<|python_tag|>" }}'), "llama")
        self.assertEqual(self.select('{{ "<｜tool▁call▁begin｜>" }}'), "deepseek")

    def test_a_template_that_says_nothing_falls_back_to_generic(self):
        self.assertEqual(self.select("{{ messages[0].content }}"), "generic")

    def test_no_template_at_all_falls_back_to_generic(self):
        self.assertEqual(select_parser(_Tokenizer(None)).family, "generic")

    def test_an_override_wins(self):
        self.assertEqual(select_parser(_Tokenizer('{{ "<tool_call>" }}'),
                                       override="mistral").family, "mistral")

    def test_an_unknown_override_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            select_parser(_Tokenizer(""), override="nonesuch")
        self.assertIn("nonesuch", str(caught.exception))
        self.assertIn("hermes", str(caught.exception))

    def test_mistral_ids_are_what_its_own_template_will_accept(self):
        """Mistral's chat templates raise unless every tool_call id is exactly 9 alphanumeric
        characters, so an OpenAI-style call_<hex> comes back in the next request and aborts the
        render rather than degrading."""
        call_id = parser_for("mistral").new_id()
        self.assertEqual(len(call_id), 9)
        self.assertTrue(call_id.isalnum())

    def test_other_families_use_openai_style_ids(self):
        self.assertTrue(parser_for("hermes").new_id().startswith("call_"))


class _Tokenizer:
    def __init__(self, template):
        self.chat_template = template


# ---- the request side --------------------------------------------------------------------------

class TestToolChoice(unittest.TestCase):

    def setup_for(self, tools=None, choice=None):
        from rocketllm.server.protocol import ChatCompletionRequest

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                                        tools=tools, tool_choice=choice)
        return resolve_tools(request)

    WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    CLOCK = {"type": "function", "function": {"name": "get_time", "parameters": {}}}

    def test_no_tools_means_no_parsing(self):
        """A model that writes <tool_call> in an ordinary answer is quoting, not calling."""
        setup = self.setup_for()
        self.assertFalse(setup.parse)
        self.assertIsNone(setup.tools)

    def test_tools_alone_render_and_parse(self):
        setup = self.setup_for(tools=[self.WEATHER])
        self.assertTrue(setup.parse)
        self.assertEqual(setup.tools, [self.WEATHER])

    def test_none_withholds_the_tools_entirely(self):
        """Rendering them while forbidding their use invites a call that would then have to be
        handed back as raw markers."""
        setup = self.setup_for(tools=[self.WEATHER], choice="none")
        self.assertFalse(setup.parse)
        self.assertIsNone(setup.tools)

    def test_a_named_choice_offers_only_that_tool(self):
        setup = self.setup_for(tools=[self.WEATHER, self.CLOCK],
                               choice={"type": "function", "function": {"name": "get_time"}})
        self.assertEqual(setup.tools, [self.CLOCK])
        self.assertTrue(setup.parse)

    def test_what_cannot_be_enforced_is_said_rather_than_pretended(self):
        """Nothing here constrains decoding, so required is a request to the model, not a promise
        to the client."""
        self.assertTrue(self.setup_for(tools=[self.WEATHER], choice="required").unenforced)
        self.assertTrue(self.setup_for(
            tools=[self.WEATHER],
            choice={"type": "function", "function": {"name": "get_weather"}}).unenforced)
        self.assertFalse(self.setup_for(tools=[self.WEATHER], choice="auto").unenforced)


class TestRoundTrip(unittest.TestCase):
    """A tool result has to render back into the next prompt as the model was trained to see it."""

    def assistant_call(self, arguments):
        return ChatMessage(role="assistant", content=None, tool_calls=[
            {"id": "call_abc", "type": "function",
             "function": {"name": "get_weather", "arguments": arguments}}])

    def test_arguments_are_decoded_from_the_wire_string_back_into_an_object(self):
        """The round-trip bug almost every implementation ships. OpenAI's wire format makes
        arguments a JSON *string*, but every chat template that renders a tool call pipes it
        through Jinja's tojson -- so a string arrives quoted and escaped where the model was
        trained on an object."""
        rendered = render_message(self.assistant_call('{"city": "Paris"}'))
        self.assertEqual(rendered["tool_calls"][0]["function"]["arguments"], {"city": "Paris"})

    def test_arguments_that_are_not_json_are_passed_through_unchanged(self):
        rendered = render_message(self.assistant_call("not json at all"))
        self.assertEqual(rendered["tool_calls"][0]["function"]["arguments"], "not json at all")

    def test_a_scalar_is_not_mistaken_for_an_arguments_object(self):
        rendered = render_message(self.assistant_call("42"))
        self.assertEqual(rendered["tool_calls"][0]["function"]["arguments"], "42")

    def test_a_tool_result_keeps_the_id_that_links_it_to_the_call(self):
        rendered = render_message(ChatMessage(role="tool", content="18 degrees",
                                              tool_call_id="call_abc", name="get_weather"))
        self.assertEqual(rendered, {"role": "tool", "content": "18 degrees",
                                    "name": "get_weather", "tool_call_id": "call_abc"})

    def test_it_survives_a_template_that_serialises_the_arguments(self):
        """The end-to-end version, through Jinja rather than by inspection: the same template that
        every one of these families uses, rendering what this server would hand it."""
        import jinja2

        template = jinja2.Template(
            "{%- for message in messages %}{%- for call in message.tool_calls %}"
            '<tool_call>\n{"name": "{{ call.function.name }}", '
            '"arguments": {{ call.function.arguments | tojson }}}\n</tool_call>'
            "{%- endfor %}{%- endfor %}")

        rendered = template.render(messages=[render_message(self.assistant_call('{"city": "Paris"}'))])
        self.assertIn('"arguments": {"city": "Paris"}', rendered)
        self.assertNotIn('\\"', rendered)

        # And the parser reads back exactly what the template just wrote, which is what closes the
        # loop: what this server emits, it can also consume.
        stream, _ = run("hermes", rendered, one_shot(rendered))
        self.assertEqual([call.name for call in stream.calls], ["get_weather"])
        self.assertEqual(json.loads(stream.calls[0].arguments), {"city": "Paris"})

    def test_a_conversation_with_a_tool_turn_renders_in_order(self):
        messages = [
            ChatMessage(role="user", content="weather in Paris?"),
            self.assistant_call('{"city": "Paris"}'),
            ChatMessage(role="tool", content="18 degrees", tool_call_id="call_abc"),
            ChatMessage(role="user", content="and tomorrow?"),
        ]
        rendered = [render_message(message) for message in messages]
        self.assertEqual([message["role"] for message in rendered],
                         ["user", "assistant", "tool", "user"])
        self.assertEqual(rendered[2]["tool_call_id"], "call_abc")
        self.assertNotIn("tool_calls", rendered[0])


class TestTheJsonScanner(unittest.TestCase):
    """The piece that makes argument streaming exact rather than approximate."""

    def span(self, text):
        from rocketllm.server.toolcalls import _ValueSpan

        span = _ValueSpan(0)
        span.advance(text)
        return span

    def test_it_stops_at_the_end_of_an_object(self):
        span = self.span('{"a": 1} trailing')
        self.assertEqual(span.slice('{"a": 1} trailing'), '{"a": 1}')

    def test_braces_inside_strings_do_not_close_it(self):
        text = '{"code": "if (x) { y(); }"} after'
        self.assertEqual(self.span(text).slice(text), '{"code": "if (x) { y(); }"}')

    def test_escaped_quotes_do_not_close_a_string(self):
        text = '{"note": "a \\"quoted\\" }"} after'
        self.assertEqual(self.span(text).slice(text), '{"note": "a \\"quoted\\" }"}')

    def test_an_unfinished_value_has_no_end(self):
        span = self.span('{"a": [1, 2')
        self.assertIsNone(span.end)

    def test_scanning_resumes_rather_than_restarting(self):
        from rocketllm.server.toolcalls import _ValueSpan

        text = '{"a": {"b": [1, 2, 3]}}'
        span = _ValueSpan(0)
        for at in range(1, len(text) + 1):
            span.advance(text[:at])
        self.assertEqual(span.end, len(text))


if __name__ == "__main__":
    unittest.main()
