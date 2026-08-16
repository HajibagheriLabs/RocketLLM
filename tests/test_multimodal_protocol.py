"""Images from the wire: what is accepted, what is refused, and what reaches the model.

The engine's multimodal path has three seams, and each is somewhere a mistake is silent rather than
loud:

  * **decoding.** A ``data:`` URL becomes a picture; an ``http(s)`` URL is refused on purpose, and
    the refusal is a feature -- fetching one would make this server issue outbound requests on a
    client's behalf against whatever the machine it runs on can reach.
  * **rendering.** Image parts must survive into the chat template as parts, because the template
    is what puts the model's placeholder tokens in the right place. Flattened to a string, the
    request still generates -- fluently, about nothing.
  * **routing.** The pixel tensors have to reach ``generate()`` beside the ids, and the request has
    to opt out of prefix reuse, which keys on token ids that two different pictures of the same
    size share exactly.

Everything runs against the same stand-in model the rest of the server tests use, on CPU, in
milliseconds. Pillow is an optional dependency, so the cases that decode a real picture skip
without it -- and the case that checks the refusal names the install hint runs either way.
"""
import base64
import importlib.util
import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketllm.server import multimodal  # noqa: E402
from rocketllm.server.app import GenerationEngine  # noqa: E402
from rocketllm.server.protocol import ChatCompletionRequest, RequestError  # noqa: E402
from rocketllm.server.toolcalls import render_message, resolve_tools  # noqa: E402
from tests.test_server import EOS_ID, FakeModel, FakeTokenizer  # noqa: E402

HAS_PILLOW = importlib.util.find_spec("PIL") is not None
needs_pillow = unittest.skipUnless(HAS_PILLOW, "Pillow is an optional dependency")

TINY_PNG_PIXEL = (2, 3)


def png_bytes(size=TINY_PNG_PIXEL, mode="RGB", colour=(200, 30, 40)):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new(mode, size, colour if mode == "RGB" else 128).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(raw, mime="image/png"):
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def chat(parts):
    return ChatCompletionRequest(messages=[{"role": "user", "content": parts}])


# ---- reading a content part ---------------------------------------------------------------------

class TestPartShapes(unittest.TestCase):
    """Clients disagree about how to spell an image part. All the shapes in the wild are read."""

    def test_the_openai_nested_shape(self):
        self.assertEqual(
            multimodal.part_image_source({"type": "image_url",
                                          "image_url": {"url": "data:image/png;base64,AA"}}),
            "data:image/png;base64,AA")

    def test_the_flat_shape_several_client_libraries_emit(self):
        self.assertEqual(
            multimodal.part_image_source({"type": "image_url", "image_url": "/tmp/a.png"}),
            "/tmp/a.png")

    def test_the_template_shape(self):
        self.assertEqual(multimodal.part_image_source({"type": "image", "image": "/tmp/a.png"}),
                         "/tmp/a.png")

    def test_a_text_part_is_not_an_image(self):
        self.assertIsNone(multimodal.part_image_source({"type": "text", "text": "hello"}))

    def test_a_part_that_is_not_a_mapping_is_not_an_image(self):
        self.assertIsNone(multimodal.part_image_source("hello"))

    def test_images_are_collected_in_the_order_the_model_will_meet_them(self):
        """The processor pairs the Nth image with the Nth placeholder, so order is the contract."""
        request = ChatCompletionRequest(messages=[
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "first"}},
                                         {"type": "text", "text": "and"},
                                         {"type": "image_url", "image_url": {"url": "second"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "third"}}]}])
        sources = [source for message in request.messages
                   for source in multimodal.message_images(message)]
        self.assertEqual(sources, ["first", "second", "third"])


# ---- decoding -----------------------------------------------------------------------------------

class TestDecoding(unittest.TestCase):

    @needs_pillow
    def test_a_base64_data_url_becomes_an_image(self):
        image = multimodal.decode_image(data_url(png_bytes()))
        self.assertEqual(image.size, TINY_PNG_PIXEL)
        self.assertEqual(image.mode, "RGB")

    @needs_pillow
    def test_a_greyscale_image_is_converted_rather_than_left_to_fail_inside_the_processor(self):
        image = multimodal.decode_image(data_url(png_bytes(mode="L")))
        self.assertEqual(image.mode, "RGB")

    @needs_pillow
    def test_wrapped_base64_is_accepted_because_real_clients_send_it(self):
        raw = base64.encodebytes(png_bytes()).decode("ascii")  # newline-wrapped at 76 columns
        self.assertIn("\n", raw)
        self.assertEqual(multimodal.decode_image(f"data:image/png;base64,{raw}").mode, "RGB")

    @needs_pillow
    def test_a_path_on_this_machine_is_read(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picture.png"
            path.write_bytes(png_bytes())
            self.assertEqual(multimodal.decode_image(str(path)).size, TINY_PNG_PIXEL)
            self.assertEqual(multimodal.decode_image(path.as_uri()).size, TINY_PNG_PIXEL)

    def test_an_http_url_is_refused_and_says_why_and_what_to_do_instead(self):
        """Deliberate, not unfinished: this would be an outbound request from wherever the server
        runs, to whatever it can reach, chosen by whoever sent the request."""
        with self.assertRaises(RequestError) as caught:
            multimodal.decode_image("https://example.invalid/cat.png")
        message = str(caught.exception)
        self.assertIn("does not fetch image URLs", message)
        self.assertIn("data:image", message, "a refusal without an alternative is a dead end")
        self.assertEqual(caught.exception.code, "image_url_not_fetched")

    def test_a_data_url_that_is_not_base64_is_a_client_error_not_a_decode_failure(self):
        with self.assertRaises(RequestError) as caught:
            multimodal.decode_image("data:image/png,not-base64-at-all")
        self.assertIn("base64", str(caught.exception))

    def test_an_empty_data_url_is_named_rather_than_decoded(self):
        with self.assertRaises(RequestError) as caught:
            multimodal.decode_image("data:image/png;base64,")
        self.assertIn("no payload", str(caught.exception))

    @needs_pillow
    def test_bytes_that_are_not_a_picture_are_a_400_with_the_decoder_s_reason(self):
        with self.assertRaises(RequestError) as caught:
            multimodal.decode_image(data_url(b"this is not a png"))
        self.assertIn("could not be decoded", str(caught.exception))

    def test_a_missing_file_is_a_client_error(self):
        with self.assertRaises(RequestError) as caught:
            multimodal.decode_image("/definitely/not/here.png")
        self.assertIn("could not read", str(caught.exception))

    def test_without_pillow_the_refusal_names_the_extra_that_fixes_it(self):
        """Run with Pillow genuinely unimportable, because that is the only way to see this path on
        a machine that has it -- and the machine that will meet it in the field does not."""
        from tests.test_optional_imports import run_without

        result = run_without("""
            from rocketllm.server import multimodal
            try:
                multimodal.decode_image("data:image/png;base64,AAAA")
            except Exception as exc:
                print("REFUSED:", type(exc).__name__, exc)
            else:
                raise SystemExit("an image was decoded without an imaging library installed")
        """, blocked=["PIL"])
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, combined)
        self.assertNotIn("Traceback (most recent call last)", combined, combined)
        self.assertIn("REFUSED: RequestError", combined)
        self.assertIn(multimodal.PILLOW_HINT, combined,
                      "a refusal that does not say what to install is a dead end")


# ---- rendering ----------------------------------------------------------------------------------

class TestTemplateRendering(unittest.TestCase):

    def message(self, parts):
        return chat(parts).messages[0]

    def test_an_image_part_survives_as_a_part_the_template_can_branch_on(self):
        content = multimodal.template_content(self.message(
            [{"type": "text", "text": "what is this?"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]))
        self.assertEqual(content, [{"type": "text", "text": "what is this?"},
                                   {"type": "image"}])

    def test_the_payload_does_not_travel_with_the_part(self):
        """The pixels go to the processor separately. Leaving a megabyte of base64 in the template's
        input would render it into the prompt on any template that prints unknown keys."""
        content = multimodal.template_content(self.message(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]))
        self.assertNotIn("image_url", content[0])
        self.assertEqual(set(content[0]), {"type"})

    def test_a_text_only_multi_part_message_still_flattens_to_a_string(self):
        """What this server has always sent, and what every template handles."""
        content = multimodal.template_content(self.message(
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]))
        self.assertEqual(content, "ab")

    def test_render_message_left_alone_is_unchanged(self):
        message = ChatCompletionRequest(
            messages=[{"role": "user", "content": "plain"}]).messages[0]
        self.assertEqual(render_message(message), {"role": "user", "content": "plain"})


# ---- routing ------------------------------------------------------------------------------------

class FakeProcessor:
    """A stand-in for a checkpoint's AutoProcessor: renders a template, then expands placeholders.

    It reproduces the one behaviour the engine depends on and cannot do for itself -- an image
    becomes some number of tokens that only the processor knows -- and records what it was handed,
    which is what the tests below actually assert on.
    """

    #: Tokens one image expands to. Any number but one would do; more than one is the point, since
    #: a naive implementation that appends a single marker per image would pass at one.
    TOKENS_PER_IMAGE = 3

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        rendered = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join("<image>" if part["type"] == "image" else part["text"]
                                  for part in content)
            rendered.append(f"{message['role']}: {content}\n")
        if kwargs.get("add_generation_prompt", True):
            rendered.append("assistant: ")
        return "".join(rendered)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        self.calls.append({"text": list(text or []), "images": list(images or [])})
        prompt = (text or [""])[0]
        expanded = prompt.replace("<image>", "<pad>" * self.TOKENS_PER_IMAGE)
        return {
            "input_ids": torch.tensor([self.tokenizer.tokens(expanded)], dtype=torch.long),
            "attention_mask": torch.ones(1, len(self.tokenizer.tokens(expanded)), dtype=torch.long),
            "pixel_values": torch.zeros(len(images or []), 4, dtype=torch.float32),
            "image_grid_thw": torch.ones(len(images or []), 3, dtype=torch.long),
        }


def multimodal_engine(script="hello there", with_processor=True):
    tokenizer = FakeTokenizer()
    ids = tokenizer.tokens(script) + [EOS_ID]
    model = FakeModel(tokenizer, ids)
    model.processor = FakeProcessor(tokenizer) if with_processor else None
    model.running_dtype = torch.float32
    return model, GenerationEngine(model, model_id="fake-vl")


class TestTheEngineRoutesImages(unittest.TestCase):

    def image_request(self, url="data:image/png;base64,AA"):
        return chat([{"type": "text", "text": "what is this?"},
                     {"type": "image_url", "image_url": {"url": url}}])

    def run_capturing_sessions(self, engine, request):
        """Run one request and hand back the PrefixSessions it built.

        Which store a session was given is the whole of the opt-out, and it is not visible from
        outside afterwards -- so it is watched as it is constructed rather than inferred from what
        the cache did or did not do next.
        """
        from unittest import mock

        from rocketllm.server import app as app_module

        request.validate_request()
        captured = []
        real = app_module.prefixes.PrefixSession

        def spy(cache, *args, **kwargs):
            session = real(cache, *args, **kwargs)
            captured.append(session)
            return session

        with mock.patch.object(app_module.prefixes, "PrefixSession", spy):
            engine._run(_StubJob(), request, chat=True)
        return captured

    @needs_pillow
    def test_the_processor_receives_the_decoded_image_and_the_rendered_prompt(self):
        model, engine = multimodal_engine()
        request = self.image_request(data_url(png_bytes()))
        input_ids, model_inputs = engine._encode_chat(request, resolve_tools(request))

        call = model.processor.calls[0]
        self.assertEqual(len(call["images"]), 1)
        self.assertEqual(call["images"][0].size, TINY_PNG_PIXEL)
        self.assertIn("<image>", call["text"][0])
        self.assertGreater(int(input_ids.shape[-1]), 0)

    @needs_pillow
    def test_the_pixel_tensors_travel_beside_the_ids_and_the_mask_does_not(self):
        """attention_mask is rebuilt from the ids further in; sending both would set it twice."""
        _, engine = multimodal_engine()
        request = self.image_request(data_url(png_bytes()))
        _, model_inputs = engine._encode_chat(request, resolve_tools(request))
        self.assertEqual(set(model_inputs), {"pixel_values", "image_grid_thw"})

    @needs_pillow
    def test_the_extra_inputs_reach_generate(self):
        model, engine = multimodal_engine()
        request = self.image_request(data_url(png_bytes()))
        request.validate_request()
        engine._run(_StubJob(), request, chat=True)
        call = model.calls[-1]
        self.assertIn("pixel_values", call)
        self.assertIn("image_grid_thw", call)

    @needs_pillow
    def test_an_image_request_opts_out_of_prefix_reuse(self):
        """Two different pictures of the same size produce identical placeholder tokens, so a cache
        keyed by ids would answer the second request from the first one's KV and say nothing."""
        _, engine = multimodal_engine()
        sessions = self.run_capturing_sessions(engine, self.image_request(data_url(png_bytes())))
        self.assertEqual(len(sessions), 1)
        self.assertIsNone(sessions[0].cache)

    def test_a_text_request_keeps_prefix_reuse_and_carries_no_extra_inputs(self):
        _, engine = multimodal_engine()
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        request.validate_request()
        input_ids, model_inputs = engine._encode_chat(request, resolve_tools(request))
        self.assertEqual(model_inputs, {})

        sessions = self.run_capturing_sessions(
            engine, ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]))
        self.assertIs(sessions[0].cache, engine.prefixes)

    def test_a_text_only_model_refuses_an_image_and_says_what_it_is(self):
        _, engine = multimodal_engine(with_processor=False)
        request = self.image_request()
        with self.assertRaises(RequestError) as caught:
            engine._encode_chat(request, resolve_tools(request))
        self.assertEqual(caught.exception.code, "model_not_multimodal")

    def test_the_model_card_says_whether_images_are_served(self):
        _, engine = multimodal_engine()
        self.assertTrue(engine.model_card().rocketllm["multimodal"])
        _, text_only = multimodal_engine(with_processor=False)
        self.assertFalse(text_only.model_card().rocketllm["multimodal"])


class _StubJob:
    """The half of a _Job the engine touches on the worker thread."""

    def __init__(self):
        import threading

        self.id = "stub"
        self.cancel = threading.Event()
        self.events = []

    def emit(self, event):
        self.events.append(event)


if __name__ == "__main__":
    unittest.main(verbosity=2)
