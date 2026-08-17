"""The correctness gate for a vision-language checkpoint, on the backend every machine has.

Same question as tests/test_cpu_generation.py and the same non-negotiable answer: generate from a
tiny checkpoint twice -- once through transformers with the whole model loaded, once through
RocketLLM streaming it a module at a time -- and require identical tokens. What this file adds is
the half that only a multimodal checkpoint has:

  * a picture actually goes through it. The image path is exercised with real pixel tensors and a
    real vision tower, because a text-only comparison would pass even with the vision tower left on
    the meta device -- it simply never runs.
  * the checkpoint's names and the model's names disagree. Qwen2.5-VL stores ``model.layers.0...``
    and ``visual...``; recent transformers builds ``model.language_model.layers.0...`` and
    ``model.visual...``. That gap is invisible until a parameter is placed, and then every one of
    them fails at once.
  * ``AutoModelForCausalLM`` refuses the config outright. Three factories are tried; this checks
    the right one is reached rather than that some model was built.

The checkpoint is built here and saved with ``save_pretrained``, which writes the same names the
published Qwen2.5-VL checkpoints carry -- so the mismatch under test is the real one, not a
reconstruction of it. No accelerator, no network.
"""
import os
import tempfile
import unittest
from pathlib import Path

import torch

from rocketllm.base import RocketModel

DEVICE = os.environ.get("ROCKETLLM_TEST_DEVICE", "cpu")

#: Vision geometry. The grid is in patches; spatial_merge_size collapses each 2x2 of them into one
#: embedding, so a 1x2x2 grid is four patches and exactly one placeholder token to substitute.
CHANNELS, PATCH, TEMPORAL, MERGE = 3, 14, 2, 2
GRID = (1, 2, 2)

TEXT_PROMPT = torch.tensor([[1, 5, 9, 14, 3]])
NEW_TOKENS = 6

#: Small ids, because the tiny config's vocabulary is 128 wide and these have to be inside it.
IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END = 10, 11, 12, 13


def build_checkpoint(root):
    """A miniature Qwen2.5-VL, saved the way the real ones are stored."""
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=64, tie_word_embeddings=False,
                         rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           in_chans=CHANNELS, out_hidden_size=32, patch_size=PATCH,
                           spatial_merge_size=MERGE, temporal_patch_size=TEMPORAL,
                           fullatt_block_indexes=[1], window_size=112),
        image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
        vocab_size=128, tie_word_embeddings=False)
    # fp32 throughout: this is a correctness comparison, and a reduced dtype would let a genuine
    # difference hide inside rounding that differs by summation order.
    model = Qwen2_5_VLForConditionalGeneration(config).to(torch.float32).eval()
    model.config.torch_dtype = "float32"
    model.save_pretrained(root, safe_serialization=True)
    return model


def image_request():
    """``(input_ids, pixel_values, image_grid_thw)`` for one image, built without a processor.

    A processor would need a real tokenizer and a real preprocessor config, neither of which a
    synthetic checkpoint has. What it produces is exactly this: placeholder tokens between the
    vision markers, flattened patches, and the grid that says how to fold them back up.
    """
    torch.manual_seed(3)
    patches = GRID[0] * GRID[1] * GRID[2]
    pixel_values = torch.randn(patches, CHANNELS * TEMPORAL * PATCH * PATCH)
    grid = torch.tensor([list(GRID)], dtype=torch.long)
    ids = [5, VISION_START] + [IMAGE_TOKEN] * (patches // (MERGE * MERGE)) + [VISION_END, 7]
    return torch.tensor([ids]), pixel_values, grid


class StreamedVL(RocketModel):
    """RocketModel without the tokenizer or processor, which a synthetic checkpoint has no reason
    to ship. The gate compares tensors and token ids, so both would only be a download."""

    def get_tokenizer(self, hf_token=None):
        return None

    def init_processor(self, hf_token=None):
        return None


class VisionStreamingCase(unittest.TestCase):
    """One checkpoint for the class, and the reference outputs taken from it once."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "model"
        cls.root.mkdir(parents=True)
        reference = build_checkpoint(cls.root).to(DEVICE)
        cls.ids, cls.pixels, cls.grid = image_request()
        with torch.no_grad():
            cls.expected_text = reference.generate(
                TEXT_PROMPT.to(DEVICE), max_new_tokens=NEW_TOKENS, do_sample=False).tolist()
            cls.expected_image_logits = reference(
                input_ids=cls.ids.to(DEVICE), pixel_values=cls.pixels.to(DEVICE),
                image_grid_thw=cls.grid.to(DEVICE)).logits
            cls.expected_image_tokens = reference.generate(
                input_ids=cls.ids.to(DEVICE), pixel_values=cls.pixels.to(DEVICE),
                image_grid_thw=cls.grid.to(DEVICE), max_new_tokens=NEW_TOKENS,
                do_sample=False).tolist()
        del reference

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def stream(self):
        return StreamedVL(str(self.root), device=DEVICE, dtype=torch.float32)

    # -- the gate ---------------------------------------------------------------------------------

    def test_streamed_text_generation_matches_a_full_load(self):
        model = self.stream()
        try:
            with torch.no_grad():
                produced = model.generate(TEXT_PROMPT.to(DEVICE), max_new_tokens=NEW_TOKENS,
                                          do_sample=False).tolist()
            self.assertEqual(produced, self.expected_text)
        finally:
            model.close()

    def test_the_image_logits_are_identical_to_a_full_load(self):
        """The one that would catch a vision tower left on meta, or placed at the wrong path: it
        runs, so a wrong answer here is arithmetic rather than a missing module."""
        model = self.stream()
        try:
            with torch.no_grad():
                got = model(input_ids=self.ids.to(DEVICE), pixel_values=self.pixels.to(DEVICE),
                            image_grid_thw=self.grid.to(DEVICE)).logits
            self.assertTrue(torch.equal(got, self.expected_image_logits),
                            f"streamed logits differ by up to "
                            f"{(got - self.expected_image_logits).abs().max().item():.3e}")
        finally:
            model.close()

    def test_streamed_generation_from_an_image_matches_a_full_load(self):
        model = self.stream()
        try:
            with torch.no_grad():
                produced = model.generate(
                    input_ids=self.ids.to(DEVICE), pixel_values=self.pixels.to(DEVICE),
                    image_grid_thw=self.grid.to(DEVICE), max_new_tokens=NEW_TOKENS,
                    do_sample=False).tolist()
            self.assertEqual(produced, self.expected_image_tokens)
        finally:
            model.close()

    def test_a_second_image_generation_matches_the_first(self):
        """Residency survives a generation on purpose. A resident vision tower must survive it too,
        and must not have been quietly unbound by the weight cache's eviction."""
        model = self.stream()
        try:
            for attempt in ("first", "second"):
                with torch.no_grad():
                    produced = model.generate(
                        input_ids=self.ids.to(DEVICE), pixel_values=self.pixels.to(DEVICE),
                        image_grid_thw=self.grid.to(DEVICE), max_new_tokens=NEW_TOKENS,
                        do_sample=False).tolist()
                self.assertEqual(produced, self.expected_image_tokens, f"the {attempt} generation")
        finally:
            model.close()

    # -- how it got there -------------------------------------------------------------------------

    def test_the_skeleton_is_built_by_the_factory_that_accepts_the_config(self):
        """AutoModelForCausalLM rejects a vision-language config with a ValueError listing every
        config class it does know, which read like an attention problem and was not one."""
        from transformers import AutoModelForCausalLM

        model = self.stream()
        try:
            self.assertEqual(model.model_factory, "AutoModelForImageTextToText")
        finally:
            model.close()
        with self.assertRaises(ValueError):
            AutoModelForCausalLM.from_config(model.config)

    def test_checkpoint_names_are_translated_to_the_paths_the_model_actually_has(self):
        model = self.stream()
        try:
            self.assertEqual(model.module_name("model.layers.0"),
                             "model.language_model.layers.0")
            self.assertEqual(model.module_name("visual"), "model.visual")
            # lm_head did not move, and a rename that reached it would place the output head inside
            # the decoder.
            self.assertEqual(model.module_name("lm_head"), "lm_head")
        finally:
            model.close()

    def test_the_streamed_sequence_is_the_decoder_and_nothing_else(self):
        model = self.stream()
        try:
            self.assertEqual(model.layer_names,
                             ["model.embed_tokens", "model.layers.0", "model.layers.1",
                              "model.norm", "lm_head"])
        finally:
            model.close()

    def test_the_vision_tower_is_found_shard_ed_and_kept_resident(self):
        model = self.stream()
        try:
            self.assertEqual(model.layer_names_dict["resident"], ["visual"])
            self.assertTrue((Path(model.checkpoint_path) / "visual.safetensors").exists(),
                            "the vision tower needs a shard of its own or nothing ever loads it")
            self.assertGreater(model.resident_bytes, 0)
            # Resident means on the device, not on meta, between generations.
            for name, param in model.model.model.visual.named_parameters():
                self.assertNotEqual(param.device.type, "meta",
                                    f"visual.{name} was never materialised")
        finally:
            model.close()

    def test_a_vision_checkpoint_reports_itself_as_one(self):
        model = self.stream()
        try:
            self.assertTrue(model.declares_vision_components())
        finally:
            model.close()


class TestTextOnlyIsUnchanged(unittest.TestCase):
    """The other half of the promise: none of the above may show up on an ordinary text model."""

    @classmethod
    def setUpClass(cls):
        from transformers import LlamaConfig, LlamaForCausalLM

        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "model"
        cls.root.mkdir(parents=True)
        torch.manual_seed(0)
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                             max_position_embeddings=64, tie_word_embeddings=False)
        model = LlamaForCausalLM(config).to(torch.float32).eval()
        model.config.torch_dtype = "float32"
        model.save_pretrained(cls.root, safe_serialization=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def stream(self):
        return StreamedVL(str(self.root), device=DEVICE, dtype=torch.float32)

    def test_a_text_checkpoint_takes_the_causal_factory_and_needs_no_translation(self):
        """Asserted over names rather than over whether a mapping object exists.

        transformers 5 declares a handful of legacy renames (LayerNorm.gamma and friends) for every
        model, so "there is no mapping" stopped being true while "nothing in this checkpoint moves"
        stayed true. The second is the property that matters.
        """
        model = self.stream()
        try:
            self.assertEqual(model.model_factory, "AutoModelForCausalLM")
            for name, _ in model.model.named_parameters():
                self.assertEqual(model.module_name(name), name,
                                 f"{name} was translated on a checkpoint that needs no translation")
            self.assertEqual(model.conversion.fusions, ())
        finally:
            model.close()

    def test_it_finds_no_vision_components_and_no_resident_modules(self):
        model = self.stream()
        try:
            self.assertFalse(model.declares_vision_components())
            self.assertEqual(model.layer_names_dict.get("resident"), [])
            self.assertEqual(model.resident_bytes, 0)
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
