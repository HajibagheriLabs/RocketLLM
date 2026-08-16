import gc
import json
import os
import ctypes
import logging
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob

from collections import defaultdict
from sys import platform

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .persist import ModelPersister

import huggingface_hub

log = logging.getLogger(__name__)

is_on_mac_os = (platform == "darwin")


class NotEnoughSpaceException(Exception):
    pass


# RocketLLM used to quantize a checkpoint's shards itself as it split them. That is gone: the engine
# imports checkpoints someone else quantized deliberately, with a toolchain built for it, rather
# than squeezing weights on the way past. Ignoring the argument silently would leave a user
# believing they were streaming 4-bit weights while the engine moved 16-bit ones -- the one failure
# this project can least afford, since bytes moved per token is the whole performance model. So it
# raises, and the message says where to get a checkpoint that does what was asked.
_COMPRESSION_REMOVED = """\
compression={value!r} is no longer supported: RocketLLM loads pre-quantized checkpoints and does \
not quantize models itself.

Point it at a checkpoint that is already quantized. Supported formats:
  AWQ, GPTQ, compressed-tensors W4A16, MXFP4, bitsandbytes-prequantized.
Most widely-used models already have such a repository on the Hugging Face hub; searching the model \
name together with "AWQ" or "GPTQ" usually finds one.

The compressed-tensors and bitsandbytes formats need their reader packages, which are optional: \
`pip install "rocketllm[quant]"` installs both. AWQ and GPTQ need nothing extra.

To load this checkpoint exactly as it is stored, drop the compression= argument."""


def reject_compression_argument(compression):
    """Refuse an on-the-fly quantization request, naming what to do instead."""
    if compression is None:
        return
    raise ValueError(_COMPRESSION_REMOVED.format(value=compression))


# Function to clean RAM & vRAM
def clean_memory(device=None):
    """Give freed memory back to the OS and the driver.

    This is expensive: releasing device blocks makes the next allocation a fresh, synchronizing
    driver call. It belongs between generations, not between layers -- see RocketModel.reset().
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        # maybe platform
        pass
    # Routed through the device abstraction so this works on every backend, not just CUDA.
    from .hw.caps import get_caps
    get_caps(device, announce=False).empty_cache()


def layer_tensor_names(local_path, layer_name):
    """List the tensors in a layer shard without reading any tensor data."""
    with safe_open(str(Path(local_path) / (layer_name + ".safetensors")), framework="pt") as f:
        return list(f.keys())


def load_layer_subset(local_path, layer_name, keys):
    """Read only `keys` from a layer shard.

    safetensors can seek to individual tensors, so a single MoE expert costs its own few MB rather
    than the whole ~16GB layer file. That is what makes per-expert streaming worthwhile.
    """
    out = {}
    with safe_open(str(Path(local_path) / (layer_name + ".safetensors")), framework="pt") as f:
        for k in keys:
            out[k] = f.get_tensor(k)
    return out


def load_layer_rows(local_path, layer_name, rows):
    """Read only the given rows of each named tensor out of a layer shard.

    This is the fused-expert read. Where a mixture stores its experts as one batched tensor there is
    no per-expert tensor to ask for, but safetensors can still seek inside one: ``get_slice(key)[e]``
    reads that row's bytes and nothing else. So a token that routes to 8 of 128 experts pays for 8,
    exactly as it would under the per-expert layout.

    `rows` maps a tensor name to the row indices wanted. The result is one compacted tensor per name,
    its rows in the order they were asked for -- the caller knows where each belongs and scatters
    them into the full-width parameter on the device.
    """
    out = {}
    with safe_open(str(Path(local_path) / (layer_name + ".safetensors")), framework="pt") as f:
        for k, indices in rows.items():
            entry = f.get_slice(k)
            parts = [entry[i:i + 1] for i in indices]
            out[k] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return out


def load_layer(local_path, layer_name):
    """Read one module's shard off disk, exactly as it is stored.

    Nothing is decoded here. A pre-quantized checkpoint's packed payloads stay packed all the way
    to the device, which is the whole point: expanding them on the host would multiply what crosses
    the link by four and undo the reason for reading a quantized checkpoint at all.
    """
    return ModelPersister.get_model_persister().load_model(layer_name, local_path)


def check_space(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model'):
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    # Shards are copied byte for byte, so the split costs exactly what the checkpoint costs.
    split_target = checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
    total, used, free = shutil.disk_usage(split_target)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        gb = 1024 ** 3
        raise NotEnoughSpaceException(
            f"Not enough space. Free space under {split_target}: {free / gb:.02f}GB. "
            f"Model total size: {total_shard_files_size_bytes / gb:.02f}GB. "
            f"existing space under {split_target} assuming can reuse: "
            f"{total_saved_split_files_size_bytes / gb:.02f}GB. ")

def remove_real_and_linked_file(to_delete):
    if (os.path.realpath(to_delete) != to_delete):
        targetpath = os.path.realpath(to_delete)

    os.remove(to_delete)
    if (targetpath):
         os.remove(targetpath)



def link_or_copy_file(src, dst):
    """Point dst at src's data without duplicating it, falling back to a real copy.

    A hard link is preferred over a symlink because it keeps the data alive even if the original
    checkpoint file is later deleted (``delete_original``), and because it costs no extra disk.
    Hard links need both paths on one filesystem, so we degrade to a symlink and finally to a copy.
    Hugging Face caches store files as symlinks into a blob dir, so we always link the real file.
    """
    src = Path(os.path.realpath(str(src)))
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return 'symlink'
    except OSError:
        pass
    shutil.copyfile(src, dst)
    return 'copy'


#: Files that mark a directory as holding a checkpoint rather than pointing at one.
WEIGHT_INDEX_FILES = ('pytorch_model.bin.index.json', 'model.safetensors.index.json',
                      'model.safetensors', 'pytorch_model.bin')


def resolve_snapshot_path(path):
    """The directory that actually holds a checkpoint, given one that may only point at it.

    A Hugging Face cache entry is ``models--org--name/{blobs,refs,snapshots/<commit>}``: config.json
    and the weight files live under a commit hash inside ``snapshots``, not at the top. The
    directory a user can actually read the model's name off is the one they type, and it used to
    fail twice over -- once as "found a local directory but no downloaded model", then again when
    the same string was retried as a repo id and rejected as an invalid repo name.

    Anything that is not a cache root comes back unchanged, so no caller has to know the difference.
    """
    path = Path(path)
    snapshots = path / 'snapshots'
    if not snapshots.is_dir():
        return path

    # refs/<revision> holds the commit that revision resolves to. Prefer main, then any other
    # revision the cache knows, and only then fall back to whichever snapshot was written last --
    # a cache may hold several commits and the newest directory is not always the checked-out one.
    candidates = []
    refs = path / 'refs'
    if refs.is_dir():
        for ref in sorted(refs.iterdir(), key=lambda p: (p.name != 'main', p.name)):
            try:
                commit = ref.read_text(encoding='utf-8').strip()
            except OSError:
                continue
            if commit and (snapshots / commit).is_dir():
                candidates.append(snapshots / commit)
    try:
        by_age = sorted((d for d in snapshots.iterdir() if d.is_dir()),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        by_age = []
    candidates.extend(d for d in by_age if d not in candidates)

    for candidate in candidates:
        if (candidate / 'config.json').exists():
            return candidate
    return candidates[0] if candidates else path


# A checkpoint whose model_type this transformers does not have produces a message about updating
# transformers and nothing about where that leaves RocketLLM, which is the half a user standing in
# front of it actually needs. The engine defers every architecture to transformers on purpose -- it
# is what makes a model released this morning stream tonight -- so "transformers does not have it"
# really does mean "nothing here can supply it", and the honest thing is to say so and name the
# bound that may be in the way.
_UNKNOWN_MODEL_TYPE = """\
transformers {version} does not recognise the model type {model_type!r}, so this checkpoint cannot \
be loaded.

RocketLLM does not implement architectures itself: transformers owns the forward pass, which is \
what lets a newly released model stream here without a code change. An architecture transformers \
does not have is therefore one nothing in this package can supply.
{remote}
Upgrade transformers to a release that has it:

  pip install --upgrade transformers

If the checkpoint is newer than any release:

  pip install "transformers @ git+https://github.com/huggingface/transformers.git"

Note that RocketLLM currently requires transformers<5.0, and that bound is not arbitrary: \
transformers 5 replaced the per-expert module lists this engine streams with one fused expert \
module, so per-expert streaming reads the wrong tensors against it. If {model_type!r} exists only \
in a 5.x release, it cannot be served here until that port is done.

Original error: {error}"""

_SHIPS_NO_REMOTE_CODE = """
The checkpoint ships no modeling code of its own either -- there is no auto_map in its config.json \
-- so trust_remote_code, which RocketLLM already passes, has nothing to load.
"""

_SHIPS_REMOTE_CODE = """
The checkpoint does declare modeling code of its own under auto_map in config.json, and RocketLLM \
already loads that with trust_remote_code=True, so this is not a permission problem. Either that \
auto_map has no AutoConfig entry -- which leaves the config type for transformers to resolve on its \
own, and it cannot -- or the code it names would not import against the transformers installed here.
"""


def _declares_remote_code(path):
    try:
        with open(Path(path) / 'config.json', 'rb') as handle:
            return bool(json.load(handle).get('auto_map'))
    except Exception:  # noqa: BLE001 - a diagnostic must not be what fails the load
        return False


def _unrecognised_model_type(path, error):
    """The model_type transformers refused, or None when this was some other failure."""
    text = str(error)
    if 'does not recognize this architecture' not in text and 'Unrecognized model' not in text:
        return None
    try:
        with open(Path(path) / 'config.json', 'rb') as handle:
            return json.load(handle).get('model_type')
    except Exception:  # noqa: BLE001
        return None


def load_checkpoint_config(path, trust_remote_code=True, hf_token=None):
    """``AutoConfig.from_pretrained``, with a readable failure for an architecture it lacks."""
    from transformers import AutoConfig

    path = resolve_snapshot_path(path) if os.path.exists(str(path)) else path
    kwargs = {'trust_remote_code': trust_remote_code}
    if hf_token is not None:
        kwargs['token'] = hf_token
    try:
        return AutoConfig.from_pretrained(path, **kwargs)
    except Exception as exc:
        model_type = _unrecognised_model_type(path, exc)
        if model_type is None:
            raise
        import transformers

        remote = (_SHIPS_REMOTE_CODE if _declares_remote_code(path) else _SHIPS_NO_REMOTE_CODE)
        raise ValueError(_UNKNOWN_MODEL_TYPE.format(
            version=transformers.__version__, model_type=model_type, remote=remote,
            error=exc)) from exc


def checkpoint_weight_map(checkpoint_path):
    """``(tensor name -> file that stores it, is_safetensors)`` for a checkpoint directory.

    Multi-shard checkpoints ship an index.json; small and modern ones often ship a single file with
    no index at all, so one is synthesized from the file's own header. Nothing here reads tensor
    data for the safetensors case -- the header answers it.
    """
    checkpoint_path = Path(checkpoint_path)
    if os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
            return json.load(f)['weight_map'], False
    if os.path.exists(checkpoint_path / 'model.safetensors.index.json'):
        with open(checkpoint_path / 'model.safetensors.index.json', 'rb') as f:
            return json.load(f)['weight_map'], True
    if os.path.exists(checkpoint_path / 'model.safetensors'):
        with safe_open(str(checkpoint_path / 'model.safetensors'), framework='pt') as f:
            return {k: 'model.safetensors' for k in f.keys()}, True
    if os.path.exists(checkpoint_path / 'pytorch_model.bin'):
        single_sd = torch.load(checkpoint_path / 'pytorch_model.bin', map_location='cpu')
        index = {k: 'pytorch_model.bin' for k in single_sd.keys()}
        del single_sd
        return index, False
    raise FileNotFoundError(
        f"No model weights found under {checkpoint_path}. Expected one of: "
        f"model.safetensors(.index.json) or pytorch_model.bin(.index.json).")


def _decoder_layer_prefix(names):
    """Where the repeated decoder blocks live, read off the checkpoint's own tensor names.

    The engine streams one thing per token: the stack of identical decoder layers. Finding it is
    structural -- a path segment ``layers`` followed by an integer -- and never a lookup by
    architecture, so a model released next month is found the same way.

    A vision tower is frequently a stack of ``layers`` too, and on a small language model paired
    with a big encoder it can be the *longer* stack, so counting blocks alone would stream the
    wrong half of the checkpoint. What separates them is that only the text decoder has a token
    embedding beside it, so that is the first thing sorted on.
    """
    stacks = {}
    for name in names:
        parts = name.split('.')
        for index, part in enumerate(parts[:-1]):
            if part == 'layers' and parts[index + 1].isdigit():
                stacks.setdefault('.'.join(parts[:index + 1]), set()).add(int(parts[index + 1]))
    if not stacks:
        return None

    known = set(names)

    def score(prefix):
        stem = prefix[:-len('.layers')].rstrip('.')
        embed = f'{stem}.embed_tokens' if stem else 'embed_tokens'
        has_embed = any(name == embed or name.startswith(embed + '.') for name in known)
        # Shortest prefix last in the key, negated, so a tie breaks toward the outer stack.
        return (has_embed, len(stacks[prefix]), -len(prefix))

    return max(stacks, key=score)


def _first_present(names, candidates):
    """The first candidate module path the checkpoint actually stores tensors for."""
    for candidate in candidates:
        if candidate and any(name.startswith(candidate + '.') for name in names):
            return candidate
    return None


def _resident_groups(names, streamed):
    """Modules the streamed sequence does not cover, as the shallowest paths that stay clear of it.

    A multimodal checkpoint carries a vision tower and a projector beside the decoder; some
    checkpoints add extra top-level norms or a materialised rotary table. None of them are part of
    embed -> layers -> norm -> lm_head, so nothing would ever load them, and the model would run
    with those modules still on the meta device.

    Each leftover tensor is walked outward from its root until the path stops being an ancestor of
    something streamed. That is what keeps ``model.visual`` from collapsing into ``model`` on a
    checkpoint whose decoder is ``model.language_model.layers``, without needing to know that
    either name means anything.
    """
    groups = set()
    for name in names:
        if any(name == path or name.startswith(path + '.') for path in streamed):
            continue
        parts = name.split('.')
        for cut in range(1, len(parts)):
            candidate = '.'.join(parts[:cut])
            if any(path == candidate or path.startswith(candidate + '.') for path in streamed):
                continue
            groups.add(candidate)
            break
    return sorted(groups)


def resolve_layer_names(tensor_names, layer_names):
    """Fit a layer-name plan to the checkpoint in front of us.

    ``layer_names`` is what the model class declares -- the ordinary
    ``model.embed_tokens / model.layers / model.norm / lm_head``, or a subclass's own. It is right
    for nearly every text checkpoint and is left completely alone when it matches. What it cannot
    cover is a checkpoint that nests its decoder somewhere else, which is what every multimodal one
    does, and modules sitting outside the streamed sequence, which is what a vision tower is.

    Returns a new dict; the input is not modified. Detection only overrides a name the checkpoint
    disagrees with, so a model that already loads keeps loading exactly as it did.
    """
    names = list(tensor_names)
    resolved = dict(layer_names)
    known = set(names)

    prefix = layer_names.get('layer_prefix')
    if not prefix or not any(name.startswith(prefix + '.') for name in known):
        found = _decoder_layer_prefix(names)
        if found is None:
            # Nothing that looks like a stack of decoder blocks. Say nothing and change nothing:
            # the declared names are still the best information available, and the splitter's own
            # error is clearer than a guess would be.
            return resolved
        prefix = found
        resolved['layer_prefix'] = prefix
        stem = prefix[:-len('.layers')].rstrip('.')
        parent = stem.rpartition('.')[0]
        for key, candidates in (
                ('embed', (f'{stem}.embed_tokens' if stem else 'embed_tokens',)),
                ('norm', (f'{stem}.norm' if stem else 'norm',)),
                # lm_head sits at the top for a plain causal model, beside the decoder for one
                # wrapped in a multimodal container, and under it for a couple of others.
                ('lm_head', ('lm_head', f'{parent}.lm_head' if parent else None,
                             f'{stem}.lm_head' if stem else None))):
            found = _first_present(known, candidates)
            if found is not None:
                resolved[key] = found
        log.info("decoder layers found at %r; streaming %s -> %s -> %s -> %s",
                 prefix, resolved.get('embed'), prefix, resolved.get('norm'),
                 resolved.get('lm_head'))

    streamed = {resolved.get(key) for key in ('embed', 'layer_prefix', 'norm', 'lm_head')}
    streamed.discard(None)
    if 'rotary_pos_emb' in resolved:
        streamed.add(resolved['rotary_pos_emb'])

    declared = list(resolved.get('resident', ()))
    discovered = [group for group in _resident_groups(known, streamed) if group not in declared]
    if discovered:
        log.info("modules outside the streamed sequence, loaded once and kept resident: %s",
                 ", ".join(discovered))
    resolved['resident'] = declared + discovered
    return resolved


def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          layer_names=None, delete_original=False, repo_id=None, hf_token=None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.

    ``layer_names`` is updated IN PLACE with whatever the checkpoint turns out to need -- a nested
    decoder prefix, a vision tower to keep resident. The caller owns that dict and has to see the
    same plan the split was written under, and this is the first point at which every tensor name
    in the checkpoint is known.
    """

    checkpoint_path = Path(checkpoint_path)


    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name


    index, safetensors_format = checkpoint_weight_map(checkpoint_path)

    if layer_names is not None:
        layer_names.update(resolve_layer_names(index.keys(), layer_names))

    if layer_names is None:
        n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    else:
        prefix = layer_names['layer_prefix']
        # Anchored, not a substring test. `k[len(prefix):]` already assumes the key starts with the
        # prefix, and a multimodal checkpoint is exactly where the two diverge: a vision tower's
        # `...vision_model.encoder.layers.3....` contains `model.layers` without being one of them,
        # and slicing it by length lands mid-name and either counts a layer that does not exist or
        # raises on int().
        n_layers = len(set(int(k[len(prefix) + 1:].split('.')[0])
                           for k in index.keys() if k.startswith(prefix + '.')
                           and k[len(prefix) + 1:].split('.')[0].isdigit()))

    if layer_names is None:
        layers = (['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)]
                  + ['model.norm.', 'lm_head.'])
    else:
        layers = ([layer_names['embed']]
                  + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)]
                  + [layer_names['norm'], layer_names['lm_head']])

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        # Modules that are not part of the streamed sequence but still need their weights on disk,
        # e.g. a multimodal model's vision tower / projector, or extra top-level norms. They get
        # their own shard and are loaded once and kept resident.
        layers = layers + list(layer_names.get('resident', []))
        layers = [name + "." for name in layers]

    # Drop layers that have no weights in the checkpoint. This happens for tied embeddings,
    # where lm_head shares storage with embed_tokens and has no entry of its own. Without this we
    # would try to save an empty shard (which fails) and never detect the split as complete.
    layers = [name for name in layers if any(k.startswith(name) for k in index.keys())]

    # Split in ascending shard order. The loop below only ever walks the shard counter forward, so
    # a module whose weights sit in an earlier shard than its predecessor's would silently be saved
    # incomplete. That ordering isn't guaranteed once non-sequential modules (a vision tower, extra
    # norms) are in the list, so sort by the last shard each module touches. This is a stable sort,
    # so plain embed -> layers -> norm -> lm_head checkpoints keep their existing order.
    def _last_shard_of(layer):
        nums = [int(v.split('-')[1]) for k, v in index.items()
                if k.startswith(layer) and '-' in v and len(v.split('-')) > 1]
        return max(nums) if nums else -1

    layers.sort(key=_last_shard_of)


    # check if splitting exists and all files are there
    found_layers = None
    #print(f"checking exists: {saving_path}")
    if os.path.exists(saving_path):
        # dir already exists, check if all layer files are there

        found_layers = {}
        for layer in layers:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)

        print(f"found_layers:{found_layers}")
        if all(found_layers.values()):
            # already downloaded, return saving path...
            print(f"saved layers already found in {saving_path}")
            return str(saving_path)
        else:
            print("some layer splits found, some are not, re-save all layers in case there's some corruptions.")

    # Some checkpoints are already sharded exactly one module per file (Kimi K3, for instance, ships
    # one ~17GB shard per decoder layer). Re-writing those into per-layer files would duplicate the
    # entire checkpoint on disk -- 1.5TB+ for a 2.8T-parameter model -- and take hours, to produce
    # byte-identical content. When a shard holds nothing but one module's tensors we link to it
    # instead of copying.
    passthrough = {}
    # Linking only produces a file the loader can read when shards are stored in the same format
    # the persister writes; the MLX persister, for instance, writes .mlx.npz.
    persister_is_safetensors = type(ModelPersister.get_model_persister()).__name__ == 'SafetensorModelPersister'
    if safetensors_format and persister_is_safetensors:
        shard_contents = defaultdict(list)
        for k, v in index.items():
            shard_contents[v].append(k)
        for layer in layers:
            files = {v for k, v in index.items() if k.startswith(layer)}
            if len(files) != 1:
                continue
            only_file = next(iter(files))
            if all(k.startswith(layer) for k in shard_contents[only_file]):
                passthrough[layer] = only_file

    if passthrough:
        print(f"{len(passthrough)}/{len(layers)} modules are already one-per-shard; "
              f"linking to the original files instead of copying them.")

    # Must exist before check_space, which stats the filesystem it lives on.
    saving_path.mkdir(parents=True, exist_ok=True)

    # A copy is only made for the layers we cannot link, so only those need free space.
    if not delete_original and len(passthrough) < len(layers):
        check_space(checkpoint_path, layer_shards_saving_path, splitted_model_dir_name=splitted_model_dir_name)


    shard = 0
    n_shards = len(set(index.values()))
    state_dict = {}

    # Map shard ordinal -> actual checkpoint filename, taken straight from the index. We must NOT
    # reconstruct names like f"model-000{n:02d}-of-000{n_shards:02d}.safetensors": repos differ in
    # zero-padding width (e.g. DeepSeek uses model-00001-of-000004.safetensors) and in extension.
    shard_num_to_file = {}
    for v in set(index.values()):
        parts = v.split('-')
        if len(parts) > 1:
            try:
                shard_num_to_file[int(parts[1])] = v
            except ValueError:
                pass

    single_modelfile = None

    for layer in tqdm(layers):

        if layer in passthrough:
            src = checkpoint_path / passthrough[layer]
            if not os.path.exists(src):
                assert repo_id is not None
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(src),
                                                  token=hf_token)
            if not ModelPersister.get_model_persister().model_persist_exist(layer, saving_path):
                link_or_copy_file(src, saving_path / (layer + 'safetensors'))
                (saving_path / (layer + 'safetensors.done')).touch()
            # Keep the shard cursor in step with what we skipped, so a later layer that does need
            # loading doesn't walk back through (and read) every shard we just linked past.
            src_parts = passthrough[layer].split('-')
            if len(src_parts) > 1:
                try:
                    shard = max(shard, int(src_parts[1]))
                except ValueError:
                    pass
            continue

        # Optionnally load next shard.
        # Checking whether after splitting from '-' a second element exists; otherwise this throws
        # for single 'model.safetensor' files.
        shards = [int(v.split('-')[1]) for k, v in index.items()
                  if k.startswith(layer) and '-' in v and len(v.split('-')) > 1]
        if len(shards) > 0:
            # A layer can span several shards (especially fp8 checkpoints, where each weight has a
            # companion weight_scale_inv tensor). Load *every* shard up to the highest one this layer
            # references, not just the next one -- otherwise the layer is saved missing some tensors
            # (e.g. the block scales), which silently corrupts fp8 weights.
            while max(shards) > shard:
                # optionally delete the original file we're done with (its tensors are already in RAM)
                if delete_original and shard != 0:
                    to_delete = checkpoint_path / shard_num_to_file[shard]

                    print(f"deleting original file: {to_delete}")
                    remove_real_and_linked_file(to_delete)
                shard += 1
                print(f'Loading shard {shard}/{n_shards}')

                to_load = checkpoint_path / shard_num_to_file[shard]

                # check if to_load exist, if not downloaad it...
                if not os.path.exists(to_load):
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                    token=hf_token)

                if not safetensors_format:
                    state_dict.update(torch.load(to_load, map_location='cpu'))
                else:
                    state_dict.update(load_file(to_load, device='cpu'))

        else:
            shards = [v for k, v in index.items() if k.startswith(layer)]
            single_modelfile = shards[0]
            to_load = checkpoint_path / single_modelfile
            # check if to_load exist, if not downloaad it...
            if not os.path.exists(to_load):
                assert repo_id is not None
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                token=hf_token)
            if not safetensors_format:
                state_dict.update(torch.load(to_load, map_location='cpu'))
            else:
                state_dict.update(load_file(to_load, device='cpu'))

        # Get layer state dict
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

        # Save layer state dict as using safetensors

        marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
        if not marker_exists:
            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

        # Free memory
        for k in layer_state_dict.keys():
            if k in state_dict:
                del state_dict[k]
        del layer_state_dict
        clean_memory()

    # deleting single modelfile if only a single modelfile was existing in hf repo
    # and deletion of single modelfile should happen in the end if delete_original=True
    if delete_original and single_modelfile is not None:
        to_delete = checkpoint_path / single_modelfile
        print(f"deleting original file: {to_delete}")
        remove_real_and_linked_file(to_delete)

    return str(saving_path)

def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None,
                                       layer_names=None, hf_token=None, delete_original=False):
    """
    find the model's local cache path, download the cache if not exists, then split and save the model.

    Parameters
    ----------
    model_local_path_or_repo_id : str
        model local path or hf repo id
    layer_shards_saving_path : str, optional
        optional path to save the splitted model, by default directly under the model local path

    Returns
    -------
    model_local_path : str
        local model path
    saved_layer_shards_path : str
        the path saved layer shards
    hf_token: str, optional
        huggingface api token could be provided, by default None
    """

    # try local model path, if the model exist split and save there
    if os.path.exists(model_local_path_or_repo_id):
        # A Hugging Face cache root keeps the files one level down under snapshots/<commit>, and
        # that directory is the one people actually type -- it is the only one with the model's
        # name in it. Resolving it here means both spellings work.
        local_path = resolve_snapshot_path(model_local_path_or_repo_id)
        # Accept single-file checkpoints too, not just sharded ones with an index: the splitter
        # handles both, so requiring an index needlessly sent local single-file models down the
        # "treat it as a repo id" path, where they fail as an invalid repo name.
        if any(os.path.exists(local_path / f) for f in WEIGHT_INDEX_FILES):
            print("found local checkpoint...")
            return local_path, split_and_save_layers(
                local_path, layer_shards_saving_path,
                layer_names=layer_names, delete_original=delete_original)
        else:
            print(f"Found local directory in {model_local_path_or_repo_id}, but didn't find a "
                  f"downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...")

    # it should be a repo id at this point...
    # First grab everything except the (potentially huge) weight files. For multi-shard models the
    # index.json tells us the structure and we stream each shard on demand during splitting.
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        #allow_patterns= ["model.safetensors.index.json", 'pytorch_model.bin.index.json'],
        ignore_patterns=['*.safetensors', '*.bin'])

    # Single-file checkpoints have no index.json, so there's nothing to stream on demand and we
    # can't infer the structure without the file itself. Download the single weight file now.
    has_index = os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json') or \
                os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json')
    if not has_index:
        hf_cache_path = huggingface_hub.snapshot_download(
            model_local_path_or_repo_id, token=hf_token,
            allow_patterns=['model.safetensors', 'pytorch_model.bin'])

    # if splitted_model subdir exists under cache use it, otherwise split and save
    return Path(hf_cache_path), split_and_save_layers(hf_cache_path, layer_shards_saving_path,
                                                      layer_names=layer_names,
                                                      delete_original=delete_original,
                                                      repo_id=model_local_path_or_repo_id,
                                                      hf_token=hf_token)
