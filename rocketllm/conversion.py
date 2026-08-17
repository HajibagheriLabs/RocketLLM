"""What transformers says about the gap between a checkpoint and the model it builds.

A checkpoint's tensor names and the parameter names of the model transformers builds from it used to
be the same string. They are increasingly not, and for two different reasons:

* **Renames.** A multimodal wrapper's module tree moved after its weights shipped. Qwen2.5-VL stores
  ``model.layers.0…`` and ``visual…``; the class transformers builds holds
  ``model.language_model.layers.0…`` and ``model.visual…``.
* **Fusions.** transformers 5 stopped building a mixture's experts as an ``nn.ModuleList`` and
  builds one batched module instead, so ``…experts.0.w1.weight`` and ``…experts.0.w3.weight`` are
  now row 0 of ``…experts.gate_up_proj``. That is a shape change, not a rename, and no amount of
  string rewriting expresses it.

transformers declares both, per model class, and applies them while loading. RocketLLM does its own
placement, so it has to apply the same declarations or every parameter lands somewhere the model
does not have. This module reads them -- it does not restate them. Nothing here knows the name of a
single architecture, and nothing here should ever learn one: the declarations are what change when a
new model is released, and reading them is what keeps this engine working on the day it is.

Two transformers generations spell the declarations differently and both are read:

* 4.x exposes ``_checkpoint_conversion_mapping``, a dict of regex renames applied first-match-wins.
* 5.x exposes ``conversion_mapping.get_model_conversion_mapping(model)``, an ordered list of
  ``WeightRenaming`` and ``WeightConverter`` objects. Renamings chain; at most one converter fires.

Where transformers offers a function to apply its own declarations, it is called rather than
reimplemented. What cannot be borrowed is the *partial* application this engine needs: transformers
assembles a fused parameter from every expert at once, and the whole point here is to assemble one
row from one expert. That arithmetic is derived from the declared operations rather than assumed --
see :class:`ExpertFusion`.
"""
import logging
import re

log = logging.getLogger(__name__)

#: Operation class names, matched structurally rather than imported. The classes live at a path that
#: has already moved once, and an engine that fails to import them would lose expert streaming
#: silently; a name that stops appearing costs the same fallback but says so.
_STACK_OP = "MergeModulelist"
_CONCAT_OP = "Concatenate"

#: What a source pattern puts where the expert index goes.
_EXPERT_WILDCARD = ".*."


class ExpertFusion:
    """One fused expert parameter, and how one row of it is built from a checkpoint.

    transformers declares this as, for example::

        source_patterns = ['mlp.experts.*.gate_proj.weight', 'mlp.experts.*.up_proj.weight']
        target_patterns = ['mlp.experts.gate_up_proj']
        operations      = [MergeModulelist(dim=0), Concatenate(dim=1)]

    read as: stack each source over the experts to get ``[E, …]``, then join the stacks along dim 1.
    A row of the result is therefore the same join of the same sources for one expert, one dimension
    lower -- ``cat([gate_proj[e], up_proj[e]], dim=0)``. That shift by one is the only arithmetic
    here, and it is what lets a token pay for its own experts instead of the layer's.

    ``concat_dim`` is None when a single source is merely stacked, in which case a row is that
    source's tensor unchanged.
    """

    __slots__ = ("sources", "target", "concat_dim", "_matchers")

    def __init__(self, sources, target, concat_dim):
        self.sources = tuple(sources)
        self.target = target
        self.concat_dim = concat_dim
        # Anchored at the end so a pattern cannot match a longer name that merely contains it, and
        # capturing the expert ordinal, which is the one thing transformers' own compiled form
        # throws away -- it never needs to know which expert a tensor was, only that it was one.
        self._matchers = tuple(re.compile(_expert_regex(pattern)) for pattern in self.sources)

    def match(self, key):
        """``(source index, expert ordinal)`` for a checkpoint key this fusion consumes, else None."""
        for index, matcher in enumerate(self._matchers):
            found = matcher.search(key)
            if found is not None:
                return index, int(found.group("expert"))
        return None

    def source_key(self, index, expert, sample):
        """The checkpoint key holding source `index` of expert `expert`, patterned after `sample`.

        `sample` is any key this fusion already matched, which is what supplies the parts of the
        name the pattern does not describe -- the layer prefix, and whatever a regex wildcard in the
        pattern stood for. Rebuilding those from the pattern alone is not possible; substituting
        the ordinal into a name that already exists is.
        """
        mine = self.match(sample)
        if mine is None:
            return None
        found = self._matchers[mine[0]].search(sample)
        head, tail = sample[:found.start("expert")], sample[found.end("expert"):]
        if index == mine[0]:
            return f"{head}{expert}{tail}"
        # A different source of the same expert: swap the pattern's fixed part as well. Both
        # patterns describe the same module path up to the leaf, so the tail is what differs.
        own_tail = _pattern_tail(self.sources[mine[0]])
        other_tail = _pattern_tail(self.sources[index])
        if own_tail is None or other_tail is None or not tail.endswith(own_tail):
            return None
        return f"{head}{expert}{tail[:len(tail) - len(own_tail)]}{other_tail}"

    def __repr__(self):
        return (f"<ExpertFusion {' + '.join(self.sources)} -> {self.target} "
                f"concat_dim={self.concat_dim}>")


def _expert_regex(pattern):
    """A regex over a checkpoint key that captures the expert ordinal.

    The pattern is transformers' own and is already a regex; only the wildcard standing for the
    expert index is replaced, with a capturing group for a number rather than the ``.*`` transformers
    compiles it to. Anchored at the end, because a leaf name is the end of a key.
    """
    head, _, tail = pattern.partition(_EXPERT_WILDCARD)
    return rf"{head}\.(?P<expert>\d+)\.{tail}$"


def _pattern_tail(pattern):
    head, sep, tail = pattern.partition(_EXPERT_WILDCARD)
    return tail if sep else None


def _row_arithmetic(operations):
    """How to build one row of a fused parameter, from the declared operations.

    Returns ``(is_fusion, concat_dim)``. A stack over the experts is what makes this an expert
    fusion at all; anything else -- a chunk, a transpose, a permutation for rope -- is a
    transformation this engine has no partial form of, and is declined rather than guessed at, so
    the layer streams whole and stays correct.
    """
    names = [type(op).__name__ for op in operations]
    if not names or names[0] != _STACK_OP:
        return False, None
    if len(names) == 1:
        return True, None
    if len(names) == 2 and names[1] == _CONCAT_OP:
        dim = getattr(operations[1], "dim", None)
        if dim is None:
            return False, None
        # The concat dim is stated over the stacked `[E, …]` tensors. One row is that tensor without
        # its leading axis, so the same join happens one dimension lower.
        return (True, dim - 1) if dim >= 1 else (False, None)
    return False, None


class CheckpointConversion:
    """The renames and expert fusions one model class declares. Read once, at load."""

    def __init__(self, rename=None, fusions=(), source="none"):
        #: Callable mapping a checkpoint key to a model parameter name, or None when they agree.
        self._rename = rename
        self.fusions = tuple(fusions)
        #: Which generation of transformers declared this, for the load report.
        self.source = source

    @property
    def renames(self):
        return self._rename is not None

    def rename(self, name):
        return name if self._rename is None else self._rename(name)

    def fusion_for(self, key):
        """The fusion that consumes this checkpoint key, or None."""
        for fusion in self.fusions:
            if fusion.match(key) is not None:
                return fusion
        return None

    def __repr__(self):
        return (f"<CheckpointConversion {self.source} renames={self.renames} "
                f"fusions={len(self.fusions)}>")


def _legacy_renamer(model_class):
    """transformers 4.x: a dict of regex renames, first match winning and the rest skipped."""
    mapping = getattr(model_class, "_checkpoint_conversion_mapping", None) or {}
    rules = [(re.compile(pattern), replacement) for pattern, replacement in mapping.items()]
    if not rules:
        return None

    def rename(name):
        for pattern, replacement in rules:
            renamed, count = pattern.subn(replacement, name)
            if count:
                return renamed
        return name

    return rename


def _modern_transforms(model):
    """transformers 5.x: the ordered transform list, or None where the API does not exist."""
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
    except ImportError:
        return None
    try:
        return list(get_model_conversion_mapping(model))
    except Exception as exc:  # noqa: BLE001 - a model we cannot ask about simply has no mapping
        log.info("could not read this model's weight conversions (%s); checkpoint names will be "
                 "used as parameter names, which is right for a checkpoint that needs no "
                 "conversion and will fail loudly for one that does", exc)
        return None


def _modern_renamer(transforms):
    """Apply only the renames, never the conversions.

    A converter's rename maps every one of an expert's tensors onto the single fused parameter they
    are rows of, which is the right answer to a question this is not asking: these names are used to
    place one tensor, and eight experts' worth of them would all be placed at the same name, each
    overwriting the last. The fusions are handled where they belong -- as fusions, in the mixture's
    own streaming path -- so here they are deliberately left out.
    """
    from transformers.core_model_loading import WeightConverter, rename_source_key

    renamings = [t for t in transforms if not isinstance(t, WeightConverter)]
    if not renamings:
        return None

    def rename(name):
        renamed, _ = rename_source_key(name, renamings, [])
        return renamed

    return rename


def _modern_fusions(transforms):
    """Every declared conversion that is a per-expert stack, as ExpertFusions."""
    from transformers.core_model_loading import WeightConverter

    fusions = []
    for transform in transforms:
        if not isinstance(transform, WeightConverter):
            continue
        sources = list(getattr(transform, "source_patterns", ()) or ())
        targets = list(getattr(transform, "target_patterns", ()) or ())
        if len(targets) != 1 or not sources:
            continue
        if not all(_EXPERT_WILDCARD in pattern for pattern in sources):
            continue
        is_fusion, concat_dim = _row_arithmetic(getattr(transform, "operations", ()))
        if not is_fusion:
            log.info("conversion %s -> %s is not a per-expert stack, so its layers stream whole",
                     sources, targets)
            continue
        if len(sources) > 1 and concat_dim is None:
            continue
        fusions.append(ExpertFusion(sources, targets[0], concat_dim))
    return fusions


def describe(model):
    """Everything this model class says about reading its own checkpoints.

    Empty for the overwhelming majority of checkpoints, whose names already match, which is why
    every caller has a fast path for that case.
    """
    transforms = _modern_transforms(model)
    if transforms is not None:
        return CheckpointConversion(rename=_modern_renamer(transforms),
                                    fusions=_modern_fusions(transforms),
                                    source="transformers>=5")
    return CheckpointConversion(rename=_legacy_renamer(type(model)), source="transformers<5")
