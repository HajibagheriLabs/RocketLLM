"""Reading a checkpoint's shape before anything is loaded from it.

Three decisions are made from nothing but a directory listing and a set of tensor names, and all
three used to be assumptions:

  * **where the checkpoint is.** A Hugging Face cache entry keeps its files under
    ``snapshots/<commit>``, not at the directory whose name has the model in it -- which is the one
    people type.
  * **where the decoder is.** ``model.layers`` is right for a text checkpoint and wrong for every
    multimodal one, which nests the decoder and puts a vision tower beside it.
  * **what is left over.** Anything the streamed embed -> layers -> norm -> lm_head sequence does
    not cover has to be found, or it never gets a shard and never leaves the meta device.

None of this needs a model, a download or an accelerator: every case here is a set of strings and a
temporary directory.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketllm.utils import (  # noqa: E402
    checkpoint_weight_map, load_checkpoint_config, resolve_layer_names, resolve_snapshot_path)

#: What RocketModel declares before it has seen a checkpoint. Everything here starts from this,
#: because the question under test is always "what did the checkpoint change about it".
DEFAULTS = {'embed': 'model.embed_tokens', 'layer_prefix': 'model.layers',
            'norm': 'model.norm', 'lm_head': 'lm_head'}


def llama_keys(layers=3):
    names = ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']
    for i in range(layers):
        names += [f'model.layers.{i}.self_attn.q_proj.weight',
                  f'model.layers.{i}.mlp.up_proj.weight']
    return names


def qwen_vl_keys(layers=3, vision_blocks=2):
    """Qwen2.5-VL as it is actually stored: an unnested decoder and a top-level ``visual``."""
    names = llama_keys(layers) + ['visual.patch_embed.proj.weight',
                                  'visual.merger.mlp.0.weight']
    for i in range(vision_blocks):
        names.append(f'visual.blocks.{i}.attn.qkv.weight')
    return names


def nested_vl_keys(layers=3, vision_layers=4):
    """The other spelling: the decoder under ``model.language_model``, vision under ``model.visual``.

    The vision tower here is a stack of ``layers`` too, and a longer one than the decoder, which is
    the case that separates counting blocks from finding the text model.
    """
    names = ['model.language_model.embed_tokens.weight', 'model.language_model.norm.weight',
             'lm_head.weight']
    for i in range(layers):
        names.append(f'model.language_model.layers.{i}.self_attn.q_proj.weight')
    for i in range(vision_layers):
        names.append(f'model.visual.encoder.layers.{i}.attn.qkv.weight')
    return names


# ---- where the checkpoint is ------------------------------------------------------------------

class TestSnapshotResolution(unittest.TestCase):
    """A cache root has to resolve to the snapshot inside it, and nothing else may move."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cache_entry(self, commits=("aaa111",), ref=None, config_in=None):
        entry = self.root / "models--Qwen--Qwen2.5-VL-7B-Instruct"
        for commit in commits:
            snapshot = entry / "snapshots" / commit
            snapshot.mkdir(parents=True)
            if config_in is None or commit in config_in:
                (snapshot / "config.json").write_text("{}", encoding="utf-8")
        if ref is not None:
            (entry / "refs").mkdir(parents=True, exist_ok=True)
            (entry / "refs" / "main").write_text(ref, encoding="utf-8")
        return entry

    def test_a_cache_root_resolves_to_the_snapshot_holding_the_files(self):
        entry = self.cache_entry()
        self.assertEqual(resolve_snapshot_path(entry), entry / "snapshots" / "aaa111")

    def test_the_revision_in_refs_wins_over_whichever_snapshot_is_newest(self):
        """A cache may hold several commits, and the newest directory is not the checked-out one."""
        entry = self.cache_entry(commits=("older", "newer"), ref="older")
        # Make the ref'd snapshot look stale, so age alone would pick the other.
        os.utime(entry / "snapshots" / "newer", (2 ** 31, 2 ** 31))
        self.assertEqual(resolve_snapshot_path(entry), entry / "snapshots" / "older")

    def test_a_snapshot_without_a_config_is_skipped_for_one_that_has_it(self):
        """Half-downloaded snapshots are ordinary; a directory with no config.json is not a model."""
        entry = self.cache_entry(commits=("empty", "complete"), config_in=("complete",))
        self.assertEqual(resolve_snapshot_path(entry), entry / "snapshots" / "complete")

    def test_a_snapshot_directory_passed_directly_is_returned_unchanged(self):
        snapshot = self.cache_entry() / "snapshots" / "aaa111"
        self.assertEqual(resolve_snapshot_path(snapshot), snapshot)

    def test_an_ordinary_model_directory_is_returned_unchanged(self):
        plain = self.root / "my-model"
        plain.mkdir()
        self.assertEqual(resolve_snapshot_path(plain), plain)


# ---- where the decoder is ---------------------------------------------------------------------

class TestLayerNameResolution(unittest.TestCase):

    def test_a_plain_text_checkpoint_changes_nothing_at_all(self):
        """The regression that matters most: every model that loads today has to keep loading."""
        resolved = resolve_layer_names(llama_keys(), DEFAULTS)
        self.assertEqual({k: resolved[k] for k in DEFAULTS}, DEFAULTS)
        self.assertEqual(resolved['resident'], [])

    def test_a_vision_tower_beside_an_ordinary_decoder_becomes_a_resident_module(self):
        """Qwen2.5-VL's own layout: the decoder names are right, `visual` is simply not covered."""
        resolved = resolve_layer_names(qwen_vl_keys(), DEFAULTS)
        self.assertEqual({k: resolved[k] for k in DEFAULTS}, DEFAULTS)
        self.assertEqual(resolved['resident'], ['visual'])

    def test_a_nested_decoder_is_found_and_the_vision_tower_with_it(self):
        resolved = resolve_layer_names(nested_vl_keys(), DEFAULTS)
        self.assertEqual(resolved['layer_prefix'], 'model.language_model.layers')
        self.assertEqual(resolved['embed'], 'model.language_model.embed_tokens')
        self.assertEqual(resolved['norm'], 'model.language_model.norm')
        self.assertEqual(resolved['lm_head'], 'lm_head')
        self.assertEqual(resolved['resident'], ['model.visual'])

    def test_the_longer_stack_of_layers_does_not_win_over_the_text_decoder(self):
        """A big encoder on a small language model. Counting blocks would stream the wrong half."""
        resolved = resolve_layer_names(nested_vl_keys(layers=2, vision_layers=32), DEFAULTS)
        self.assertEqual(resolved['layer_prefix'], 'model.language_model.layers')

    def test_the_resident_group_stops_at_the_shallowest_path_clear_of_the_decoder(self):
        """`model.visual`, not `model` -- which would swallow the decoder it sits beside."""
        resolved = resolve_layer_names(nested_vl_keys(), DEFAULTS)
        self.assertNotIn('model', resolved['resident'])

    def test_a_declared_layout_is_kept_and_its_resident_list_is_extended(self):
        """Kimi K3 declares all four names and four resident modules; detection must not fight it."""
        declared = {'embed': 'language_model.model.embed_tokens',
                    'layer_prefix': 'language_model.model.layers',
                    'norm': 'language_model.model.norm',
                    'lm_head': 'language_model.lm_head',
                    'resident': ['mm_projector', 'vision_tower']}
        names = ['language_model.model.embed_tokens.weight',
                 'language_model.model.layers.0.self_attn.q_proj.weight',
                 'language_model.model.layers.1.self_attn.q_proj.weight',
                 'language_model.model.norm.weight', 'language_model.lm_head.weight',
                 'mm_projector.proj.0.weight', 'vision_tower.encoder.blocks.0.mlp.fc0.weight']
        resolved = resolve_layer_names(names, declared)
        self.assertEqual(resolved['layer_prefix'], 'language_model.model.layers')
        self.assertEqual(resolved['resident'], ['mm_projector', 'vision_tower'])

    def test_tied_embeddings_leave_lm_head_alone_rather_than_inventing_a_path(self):
        """A tied model stores no lm_head. The splitter drops the name; nothing here may guess one."""
        names = [n for n in nested_vl_keys() if not n.startswith('lm_head')]
        resolved = resolve_layer_names(names, DEFAULTS)
        self.assertEqual(resolved['lm_head'], 'lm_head')
        self.assertNotIn('lm_head', resolved['resident'])

    def test_a_checkpoint_with_no_layer_stack_is_left_completely_alone(self):
        """Nothing recognisable is a reason to say nothing, not a reason to guess."""
        resolved = resolve_layer_names(['transformer.h.0.attn.weight'], DEFAULTS)
        self.assertEqual({k: resolved[k] for k in DEFAULTS}, DEFAULTS)

    def test_the_caller_s_dict_is_not_modified(self):
        original = dict(DEFAULTS)
        resolve_layer_names(nested_vl_keys(), DEFAULTS)
        self.assertEqual(DEFAULTS, original)


# ---- reading the index ------------------------------------------------------------------------

class TestWeightMap(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_sharded_checkpoint_is_read_from_its_index(self):
        weight_map = {name: "model-00001-of-00001.safetensors" for name in llama_keys(1)}
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
        index, is_safetensors = checkpoint_weight_map(self.root)
        self.assertEqual(set(index), set(weight_map))
        self.assertTrue(is_safetensors)

    def test_a_single_file_checkpoint_has_its_index_synthesised_from_the_header(self):
        save_file({name: torch.zeros(2) for name in llama_keys(1)},
                  str(self.root / "model.safetensors"))
        index, is_safetensors = checkpoint_weight_map(self.root)
        self.assertEqual(set(index), set(llama_keys(1)))
        self.assertEqual(set(index.values()), {"model.safetensors"})
        self.assertTrue(is_safetensors)

    def test_a_directory_with_no_weights_says_what_it_expected_to_find(self):
        with self.assertRaises(FileNotFoundError) as caught:
            checkpoint_weight_map(self.root)
        self.assertIn("model.safetensors", str(caught.exception))


# ---- an architecture transformers does not have -------------------------------------------------

class TestUnknownArchitecture(unittest.TestCase):
    """The `KeyError: 'qwen3_5'` case, which used to surface as a bare traceback.

    RocketLLM defers every architecture to transformers on purpose, so "transformers does not have
    it" genuinely means "nothing here can supply it". That is worth saying, along with the version
    bound that may be what is standing in the way.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, config):
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def message_for(self, config):
        self.write(config)
        with self.assertRaises(ValueError) as caught:
            load_checkpoint_config(self.root)
        return str(caught.exception)

    def test_it_names_the_model_type_the_installed_transformers_and_what_to_type(self):
        import transformers

        message = self.message_for({"model_type": "a_model_type_nobody_has_published",
                                    "architectures": ["NoSuchForCausalLM"]})
        self.assertIn("a_model_type_nobody_has_published", message)
        self.assertIn(transformers.__version__, message)
        self.assertIn("pip install --upgrade transformers", message)

    def test_it_states_the_supported_range_so_the_upgrade_advice_is_actionable(self):
        """"Upgrade transformers" is only useful next to what this package will accept.

        It also has to name the interpreter floor: transformers 5 needs Python 3.10, so on 3.9 the
        upgrade silently resolves back to a 4.x that still does not have the architecture, and
        nothing would explain why the advice did not work.
        """
        message = self.message_for({"model_type": "a_model_type_nobody_has_published"})
        self.assertIn("4.49", message)
        self.assertIn("3.10", message)

    def test_an_architecture_added_in_transformers_5_loads_when_transformers_has_it(self):
        """The `KeyError: 'qwen3_5'` this whole path was built around, from the other side.

        Skipped rather than dropped on an older transformers: the message under test is only
        reachable where the architecture is genuinely absent, and which of the two cases the
        running environment is in is exactly what a reader needs to see recorded.
        """
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        if "qwen3_5" not in CONFIG_MAPPING_NAMES:
            # Refused, and the refusal explains itself -- which is the whole of the behaviour on a
            # transformers that does not have it.
            self.assertIn("qwen3_5", self.message_for({"model_type": "qwen3_5"}))
            self.skipTest("this transformers predates qwen3_5, so it is correctly refused")
        self.write({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]})
        config = load_checkpoint_config(self.root)
        self.assertEqual(config.model_type, "qwen3_5")

    def test_it_says_whether_the_checkpoint_carries_modeling_code_of_its_own(self):
        """Whether trust_remote_code had anything to work with changes what to do next."""
        without = self.message_for({"model_type": "not_a_real_type_at_all"})
        self.assertIn("no auto_map", without)

        # An auto_map that names a model class but no AutoConfig: transformers still has to resolve
        # the config type itself, and still cannot. This is the real shape of the case, because an
        # auto_map WITH an AutoConfig entry resolves through the remote code and never gets here.
        with_code = self.message_for({
            "model_type": "not_a_real_type_at_all",
            "auto_map": {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}})
        self.assertIn("auto_map", with_code)
        self.assertNotIn("no auto_map", with_code)

    def test_a_config_transformers_does_know_still_loads(self):
        self.write({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
        config = load_checkpoint_config(self.root)
        self.assertEqual(config.model_type, "llama")

    def test_a_cache_root_is_resolved_before_the_config_is_read(self):
        entry = self.root / "models--org--name" / "snapshots" / "abc123"
        entry.mkdir(parents=True)
        (entry / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
            encoding="utf-8")
        config = load_checkpoint_config(self.root / "models--org--name")
        self.assertEqual(config.model_type, "llama")


if __name__ == "__main__":
    unittest.main(verbosity=2)
