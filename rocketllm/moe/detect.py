"""Finding a decoder layer's experts without being told where they are.

A sparse mixture routes each token to a handful of its experts, so materialising a whole MoE layer
moves one or two orders of magnitude more bytes than the token actually reads. Streaming experts
individually is the largest single saving available on such a model -- but only if the engine can
*find* them, and until now it could not: it had to be handed an ``expert_prefix`` by a subclass, so
exactly one architecture got the fast path and every other mixture materialised whole layers.

This module removes the configuration. It looks at the model transformers built and at the shapes
the checkpoint stores, and works out where the experts are from the structure alone. Nothing here
may key off an architecture name, a class name or a module name: a mixture released next month has
to work in this build without a code change, and its names are exactly what will be different about
it. What is stable across every mixture ever shipped is the shape of the thing -- many
interchangeable sub-modules, or one batched tensor with an expert per row, sitting beside something
that emits one score per expert.

Two layouts exist in the wild and both are handled.

``module_list``
    The classic layout: a list of per-expert modules, addressed as ``...experts.7.gate_proj.weight``.
    A forward hook per expert module streams exactly the experts that run, so neither the router nor
    the top-k has to be understood at all. Note that "the experts that run" is the model's decision:
    some implementations call only the routed ones, others walk the whole list and mask the rest, and
    only the first kind reduces bytes read. Both stop the layer's experts being resident at once.
    Mixtral, Qwen2/Qwen3-MoE, DeepSeek, OLMoE, Phi-MoE, Granite-MoE and Kimi K3 are all this.

``fused``
    Recent transformers batches several families' experts into one tensor per projection --
    ``experts.gate_up_proj`` with shape ``[num_experts, ...]`` -- and runs them with a single
    ``bmm``. There are no per-expert modules to hook, so the router's selection is read as it is
    produced and only the routed rows are read out of the shard. Llama 4 and Aria are this.

``fused_merge``
    The module tree is fused and the CHECKPOINT is not. transformers 5 rebuilt Mixtral, Qwen2-MoE
    and their relatives around a batched expert module while their published weights stayed
    per-expert, so the two no longer describe the same shape and neither of the layouts above fits:
    there is no expert module to hook, and the row the fused parameter wants does not exist in the
    shard as a tensor. It has to be built -- ``gate_up_proj[e]`` is ``cat(w1[e], w3[e])`` -- and
    which tensors compose which row is not guessed here. transformers declares it, and
    :mod:`rocketllm.conversion` reads that declaration; see :class:`~rocketllm.conversion.ExpertFusion`.

    The saving survives the arrangement. A per-expert checkpoint already stores each expert as its
    own tensors, so reading the routed few costs their own bytes exactly as slicing a fused tensor
    does, and the assembly is a concatenation of tensors already in hand.

The result is per-container rather than per-layer, and that matters. A layer holding something this
module recognises as experts but cannot stream *safely* does not poison the rest of the layer: that
container's tensors come back in ``other_keys`` and stream with the layer exactly as they did
before, the reason is recorded, and the layer's other containers still get the fast path. Guessing
wrong about a mixture means silently wrong output, which is worth giving up every byte of savings to
avoid -- so each uncertain case resolves to whole-layer streaming and says why.
"""
import dataclasses
import logging
import re

log = logging.getLogger(__name__)

#: The layouts this module can recognise.
LAYOUT_MODULE_LIST = "module_list"
LAYOUT_FUSED = "fused"
#: A fused module over a per-expert checkpoint. The rows are assembled as they are read.
LAYOUT_FUSED_MERGE = "fused_merge"

#: Config fields a checkpoint may declare its routing width under. This is the one number that
#: cannot be recovered from structure -- the expert count is a tensor dimension, but how many of
#: them a token visits is recorded only in the config. These are config field names, not
#: architecture names: a new mixture spelling it any of these ways works untouched, and one that
#: spells it otherwise is reported as ambiguous rather than guessed at.
#:
#: A bare ``top_k`` is deliberately absent. It is also the name of a sampling parameter, and reading
#: a generation setting as a routing width would quietly stream the wrong experts.
TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_topk", "moe_top_k",
              "num_selected_experts", "topk_experts", "moe_k")

#: How many entries a container needs before it is called a mixture. Two is the smallest number that
#: means anything, and low enough to admit the toy mixtures people test with.
MIN_EXPERTS = 2


@dataclasses.dataclass(frozen=True)
class MergedTarget:
    """One fused parameter that has to be built from a checkpoint's per-expert tensors.

    ``sources`` is ``{expert index: (checkpoint key per source, in the declared order)}``, and the
    order is load-bearing: it is the order the concatenation happens in, so swapping two of them
    swaps halves of every expert's weight and produces fluent, wrong output rather than an error.
    """

    #: Where the parameter lives relative to the decoder layer, e.g. ``mlp.experts.gate_up_proj``.
    path: str
    #: The shape the module builds, which is what the destination tensor is allocated at.
    shape: tuple
    #: Which dimension of ONE ROW the sources join on, or None when there is a single source.
    concat_dim: object
    sources: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ExpertContainer:
    """One streamable group of experts inside a decoder layer."""

    #: LAYOUT_MODULE_LIST, LAYOUT_FUSED or LAYOUT_FUSED_MERGE.
    layout: str
    #: Module path relative to the decoder layer, e.g. ``"mlp.experts"``.
    path: str
    num_experts: int
    #: Experts a token routes to, where the config declares it. Never required for the module-list
    #: layout -- the model itself decides which expert modules to call -- and always required for
    #: the fused one, which has to reproduce the selection to know which rows to read.
    top_k: int = None
    #: Module path, relative to the decoder layer, of the thing that emits one score per expert.
    #: Only the fused layout needs it.
    router_path: str = None
    #: ``{expert_index: (checkpoint keys,)}``, for the module-list layout.
    expert_keys: dict = dataclasses.field(default_factory=dict)
    #: ``{checkpoint key: full shape}`` for the fused layout. The shape is the checkpoint's, which
    #: is what the destination tensor has to be allocated at.
    fused_shapes: dict = dataclasses.field(default_factory=dict)
    #: ``{module path: (checkpoint keys,)}`` for the always-on feed-forward paths beside the routed
    #: experts -- what architectures variously call a shared expert. Every token takes these, so
    #: they are worth keeping resident rather than streaming, and they are found by exclusion rather
    #: than by name; see :func:`_find_shared_modules`.
    shared_keys: dict = dataclasses.field(default_factory=dict)
    #: ``{layer-relative parameter path: MergedTarget}`` for LAYOUT_FUSED_MERGE.
    merged_targets: dict = dataclasses.field(default_factory=dict)

    @property
    def is_fused(self):
        """Whether the module is one batched tensor rather than a list of expert modules.

        True for both fused layouts, because what follows from it is the same either way: there is
        no per-expert module to hook, so the container is hooked instead and the router's selection
        decides what is read. Where the two differ is only in how a row is obtained -- sliced out of
        the shard, or assembled from it -- which is :attr:`is_merged`'s business.
        """
        return self.layout in (LAYOUT_FUSED, LAYOUT_FUSED_MERGE)

    @property
    def is_merged(self):
        """Whether a row has to be built from several checkpoint tensors rather than read."""
        return self.layout == LAYOUT_FUSED_MERGE

    @property
    def keys(self):
        """The checkpoint tensors of the ROUTED experts, and nothing else.

        Deliberately excludes the shared modules: the fused path reads rows out of exactly these
        tensors, so anything that is not expert-major must stay out of it.
        """
        if self.layout == LAYOUT_FUSED:
            return tuple(self.fused_shapes)
        return tuple(key for keys in self.expert_keys.values() for key in keys)

    @property
    def owned_keys(self):
        """Everything this container takes off the layer's own stream, shared modules included."""
        shared = tuple(key for keys in self.shared_keys.values() for key in keys)
        return self.keys + shared

    def target_shapes(self, layer_name):
        """``{full parameter name: full shape}`` for the destinations rows are scattered into.

        Named in the checkpoint's namespace for both fused layouts, so the caller translates them
        the same way it translates everything else it places. For the merged layout the leaf comes
        from the module tree, because it is the module -- not the checkpoint -- that has a fused
        parameter at all.
        """
        if self.layout == LAYOUT_FUSED:
            return dict(self.fused_shapes)
        prefix = f"{layer_name}." if layer_name else ""
        return {f"{prefix}{target.path}": target.shape
                for target in self.merged_targets.values()}

    def describe(self):
        top_k = "unknown" if self.top_k is None else self.top_k
        shared = f", {len(self.shared_keys)} shared" if self.shared_keys else ""
        return (f"{self.path or '<layer>'}: {self.layout} layout, {self.num_experts} experts, "
                f"top-k {top_k}{shared}")


@dataclasses.dataclass(frozen=True)
class ExpertLayout:
    """What one decoder layer turned out to hold."""

    containers: tuple = ()
    #: The layer's tensors that no streamable container owns -- attention, norms, the router, shared
    #: experts, and anything belonging to a container that was recognised but rejected. This is what
    #: the layer's ordinary streaming hook has to load.
    other_keys: tuple = ()
    #: ``(path, reason)`` for containers that look like experts but stream with the layer.
    skipped: tuple = ()

    def __bool__(self):
        return bool(self.containers)

    @property
    def num_experts(self):
        return sum(c.num_experts for c in self.containers)


# -- structural predicates ------------------------------------------------------------------------

def _indexed_children(module):
    """Child modules named by integers, or ``None``.

    This is the structural signature of an ``nn.ModuleList``, expressed over the thing the module
    tree and the checkpoint agree on -- a child's name -- rather than over the container's Python
    type. A model building its experts in some other sequence container, or a remote-code class
    subclassing ModuleList, is the same thing to a streaming engine and is recognised here.
    """
    children = dict(module.named_children())
    if len(children) < MIN_EXPERTS or not all(name.isdigit() for name in children):
        return None
    return {int(name): child for name, child in children.items()}


def _parameter_signature(module):
    """A module's parameters as (name, shape) pairs, for comparing one expert against another."""
    return tuple(sorted((name, tuple(int(d) for d in param.shape))
                        for name, param in module.named_parameters()))


def _interchangeable(children):
    """Whether every entry has the same parameters with the same shapes.

    Experts are alternatives to one another, so they are built identically; the stages of a
    ``Sequential`` are a pipeline and are not. That difference separates a real expert list from any
    other integer-named container, and it is free to check -- the model is on the meta device, so
    its shapes are known without reading a weight.
    """
    signatures = {_parameter_signature(child) for child in children.values()}
    return len(signatures) == 1 and bool(next(iter(signatures)))


def _find_router(parent, container_name, num_experts, widths=None):
    """The sibling of an expert container that emits one score per expert.

    Structural again: a router turns a hidden state into ``num_experts`` numbers, so it holds a
    parameter with an expert-sized leading dimension, and beside a set of experts nothing else does.
    The search covers a sibling's whole subtree, because some routers wrap their linear layer in a
    gating module rather than being one.

    ``widths`` is the set of dimensions the candidate experts actually consume, and passing it is
    what keeps a leading dimension from being read as an expert count by coincidence. A depthwise
    convolution beside a linear projection is the case that made this necessary: Qwen3.5's
    ``linear_attn.conv1d`` is ``[8192, 1, 4]`` and its sibling ``in_proj_qkv`` is ``[8192, 2048]``,
    so both lead with 8192 and the pair looks exactly like 8192 experts with a router in front of
    them. It is not one, and streaming 8 rows of 8192 would have zeroed the rest of a convolution
    that every token needs -- fluent, wrong output, and nothing to see in a log.

    What separates the two is that a router reads the same width its experts do. Checked against
    every non-expert dimension of the fused tensor rather than a chosen one, because families
    disagree about the order: ``[E, 2I, H]`` for Mixtral, ``[E, H, 2I]`` for Llama 4.

    ``None`` unless exactly one sibling qualifies. Two candidates means the structure is not saying
    which one drives the experts, and the honest response to that is to stop.
    """
    candidates = []
    for name, child in parent.named_children():
        if name == container_name:
            continue
        for _, param in child.named_parameters():
            if param.ndim < 1 or int(param.shape[0]) != num_experts:
                continue
            if widths is not None and not (param.ndim == 2 and int(param.shape[1]) in widths):
                continue
            candidates.append(name)
            break
    return candidates[0] if len(candidates) == 1 else None


def _find_shared_modules(parent, container_name, router_name):
    """Siblings of the expert container that every token goes through anyway.

    Inside a mixture block the children are the routed experts, the thing that routes to them, and
    -- where the architecture has one -- a feed-forward path taken regardless of routing. This
    returns that third group, defined by exclusion: whatever is neither the experts nor the router.

    Exclusion rather than a name test, because the names are the part that varies (`shared_expert`,
    `shared_experts`, `shared_mlp`, and a gate beside it) while the structure does not. Being
    unrouted is exactly what makes them worth pinning: a routed expert earns residency by being
    popular, and these are read on every token by construction, so they never have to earn it.
    """
    shared = []
    for name, child in parent.named_children():
        if name in (container_name, router_name):
            continue
        if any(True for _ in child.parameters()):
            shared.append(name)
    return tuple(shared)


def _find_int(node, keys, depth=0):
    """First integer stored under any of `keys`, at any depth of a config.

    Multimodal checkpoints keep the decoder's settings in a sub-config, and mixtures differ in where
    they record the routing width, so the search cannot be a fixed path.
    """
    if depth > 6:
        return None
    if isinstance(node, dict):
        for key in keys:
            value = node.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
        children = list(node.values())
    elif isinstance(node, (list, tuple)):
        children = list(node)
    elif hasattr(node, "__dict__"):
        for key in keys:
            value = getattr(node, key, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
        children = [v for k, v in vars(node).items() if not k.startswith("_")]
    else:
        return None
    for child in children:
        found = _find_int(child, keys, depth + 1)
        if found is not None:
            return found
    return None


def resolve_top_k(config, num_experts):
    """How many experts a token visits, or ``None`` when the config does not say.

    Bounded by the expert count on the way out: a value outside ``1..num_experts`` is not a routing
    width but some other field that happened to share a name, and acting on it would stream the
    wrong rows.
    """
    if config is None:
        return None
    found = _find_int(config, TOP_K_KEYS)
    if found is None or not 1 <= found <= num_experts:
        return None
    return found


# -- per-container recognition ----------------------------------------------------------------------

def _module_list_container(path, children, parent, parent_path, container_name, keys_by_prefix,
                           config):
    """Recognise an expert list, or return ``(None, reason)``."""
    if not _interchangeable(children):
        # A pipeline of differently-shaped stages. Not experts at all, and not worth reporting.
        return None, None

    expert_keys = {}
    for index in sorted(children):
        found = keys_by_prefix.get(f"{path}.{index}.")
        if found:
            expert_keys[index] = tuple(found)

    if len(expert_keys) < MIN_EXPERTS:
        # The module tree says experts, the shard does not store them per expert. That is a
        # checkpoint this code does not understand; stream the layer whole rather than invent a
        # mapping between the two.
        return None, (f"the module tree holds {len(children)} interchangeable entries but the shard "
                      f"stores per-expert tensors for {len(expert_keys)} of them")

    # Unlike the fused layout this one does not *need* the router -- the model calls the expert
    # modules itself -- but knowing it is what allows the top-k to be fetched in parallel the moment
    # it fires, and what separates an always-on sibling from the thing that routes. Missing it costs
    # those two, not correctness.
    router_name = _find_router(parent, container_name, len(children)) if parent is not None else None
    shared = _shared_key_map(parent, parent_path, container_name, router_name, keys_by_prefix)

    return ExpertContainer(layout=LAYOUT_MODULE_LIST, path=path, num_experts=len(children),
                           top_k=resolve_top_k(config, len(children)),
                           router_path=_qualify(parent_path, router_name),
                           expert_keys=expert_keys, shared_keys=shared), None


def _qualify(parent_path, name):
    if name is None:
        return None
    return f"{parent_path}.{name}" if parent_path else name


def _shared_key_map(parent, parent_path, container_name, router_name, keys_by_prefix):
    """``{module path: checkpoint keys}`` for the always-on siblings that the shard actually holds."""
    if parent is None:
        return {}
    shared = {}
    for name in _find_shared_modules(parent, container_name, router_name):
        path = _qualify(parent_path, name)
        found = keys_by_prefix.get(f"{path}.")
        if found:
            shared[path] = tuple(found)
    return shared


def _merged_targets(path, direct, shapes, prefix, conversion):
    """Build the per-parameter assembly plan for a fused module over a per-expert checkpoint.

    Returns ``(targets, expert_keys, reason)``. ``targets`` is empty when this is not that case --
    which is nearly always, since a checkpoint whose shape matches the module it loads into needs
    none of this.

    Every requirement below is a way for the plan to be wrong rather than merely absent, so each one
    refuses instead of improvising: a missing source tensor, experts that disagree about how many of
    them there are, or a count that does not match the dimension the module built.
    """
    if conversion is None or not getattr(conversion, "fusions", ()):
        return {}, {}, None

    targets = {}
    expert_keys = {}
    for name, param in direct:
        relative_path = f"{path}.{name}"
        fusion = _fusion_for_target(conversion, relative_path)
        if fusion is None:
            return {}, {}, (f"the module holds a batched {name} that the shard does not store, and "
                            f"nothing this model declares says what it is built from")

        # Which checkpoint tensors this fusion consumes, bucketed by expert. Read off the shard's
        # own key list rather than generated from the pattern, so a checkpoint that stores an
        # unexpected subset is discovered here rather than by a read that returns nothing.
        found = {}
        for relative_key in shapes:
            hit = fusion.match(relative_key)
            if hit is None:
                continue
            source_index, expert = hit
            found.setdefault(expert, {})[source_index] = prefix + relative_key
        if not found:
            return {}, {}, (f"{relative_path} is declared to be built from {fusion.sources}, none "
                            f"of which this shard holds")

        complete = {expert: tuple(parts[i] for i in range(len(fusion.sources)))
                    for expert, parts in found.items()
                    if len(parts) == len(fusion.sources)}
        if len(complete) != len(found):
            return {}, {}, (f"{relative_path} needs {len(fusion.sources)} tensors per expert and "
                            f"the shard is missing some of them")

        declared = int(param.shape[0]) if param.ndim else 0
        if len(complete) != declared:
            return {}, {}, (f"the module builds {relative_path} for {declared} experts and the "
                            f"shard stores {len(complete)}")

        targets[relative_path] = MergedTarget(
            path=relative_path, shape=tuple(int(d) for d in param.shape),
            concat_dim=fusion.concat_dim, sources=complete)
        for expert, keys in complete.items():
            expert_keys.setdefault(expert, []).extend(keys)

    counts = {len(target.sources) for target in targets.values()}
    if len(counts) > 1:
        return {}, {}, (f"the parameters of {path} disagree about how many experts there are "
                        f"({sorted(counts)})")
    return targets, {expert: tuple(keys) for expert, keys in expert_keys.items()}, None


def _fusion_for_target(conversion, relative_path):
    """The declared fusion that produces this module parameter, or None.

    The declared target is a suffix pattern -- ``.experts.gate_up_proj`` -- because it has to match
    at whatever depth the mixture sits, so it is searched for rather than compared.
    """
    for fusion in conversion.fusions:
        try:
            if re.search(fusion.target, relative_path):
                return fusion
        except re.error:
            continue
    return None


def _fused_container(path, module, parent, parent_path, container_name, shapes, prefix, config,
                     keys_by_prefix, conversion=None):
    """Recognise a fused ``[num_experts, ...]`` expert tensor, or return ``(None, reason)``.

    The signature is a module whose own parameters -- not its children's -- all lead with the same
    dimension, at least one of them batched into three. That is what a fused expert tensor is, and
    it is hard to produce by accident; but "hard" is not "impossible", so the expert count is
    confirmed against a router before any row is streamed off the back of it.
    """
    direct = list(module.named_parameters(recurse=False))
    if not direct:
        return None, None

    batched = any(param.ndim >= 3 for _, param in direct)
    fused_shapes = {}
    merged = {}
    merged_expert_keys = {}
    for name, _ in direct:
        relative_key = f"{path}.{name}"
        shape = shapes.get(relative_key)
        if shape is None:
            if not batched:
                return None, None
            # The shard does not store this parameter. Either the checkpoint predates the module
            # being fused -- in which case what it does store is the per-expert tensors the rows are
            # built from, and the model says so -- or there is nothing here to stream.
            merged, merged_expert_keys, reason = _merged_targets(path, direct, shapes, prefix,
                                                                 conversion)
            if not merged:
                return None, reason or (f"the module holds batched expert tensors but the shard "
                                        f"has no {prefix}{relative_key}")
            break
        # Keyed the way the checkpoint names it, because that is what will be read and what the
        # layer's own stream has to be told not to load.
        fused_shapes[prefix + relative_key] = shape

    if merged:
        return _merged_container(path, parent, parent_path, container_name, merged,
                                 merged_expert_keys, config, keys_by_prefix)

    leading = {shape[0] for shape in fused_shapes.values() if shape}
    if len(leading) != 1:
        return None, None
    num_experts = int(leading.pop())
    if num_experts < MIN_EXPERTS:
        return None, None
    if not any(len(shape) >= 3 for shape in fused_shapes.values()):
        # Every fused expert projection is a batch of matrices. A stack of vectors is a bias table
        # or an embedding, and slicing it per expert would be reading something else entirely.
        return None, None

    widths = {int(dim) for shape in fused_shapes.values() for dim in shape[1:]}
    router_name = (_find_router(parent, container_name, num_experts, widths)
                   if parent is not None else None)
    if router_name is None:
        return None, (f"tensors are batched {num_experts} ways but no single sibling both emits "
                      f"{num_experts} scores and reads a width these tensors use, so the expert "
                      f"axis is unconfirmed")

    top_k = resolve_top_k(config, num_experts)
    if top_k is None:
        return None, (f"{num_experts} fused experts, but the config does not declare how many a "
                      f"token routes to under any of: {', '.join(TOP_K_KEYS)}")

    shared = _shared_key_map(parent, parent_path, container_name, router_name, keys_by_prefix)
    return ExpertContainer(layout=LAYOUT_FUSED, path=path, num_experts=num_experts, top_k=top_k,
                           router_path=_qualify(parent_path, router_name),
                           fused_shapes=fused_shapes, shared_keys=shared), None


def _merged_container(path, parent, parent_path, container_name, targets, expert_keys, config,
                      keys_by_prefix):
    """Wrap an assembly plan as a container, once the router confirms the expert axis.

    The same two confirmations the fused layout insists on, for the same reason. A batched leading
    dimension is easy to produce by accident; a sibling that emits exactly that many scores is not.
    And the routing width has to be declared, because a layout with no per-expert module to hook
    learns which experts ran only by reproducing the router's own choice.
    """
    num_experts = len(next(iter(targets.values())).sources)
    widths = {int(dim) for target in targets.values() for dim in target.shape[1:]}
    router_name = (_find_router(parent, container_name, num_experts, widths)
                   if parent is not None else None)
    if router_name is None:
        return None, (f"the shard stores {num_experts} experts for a batched module, but no single "
                      f"sibling both emits {num_experts} scores and reads a width they use, so the "
                      f"expert axis is unconfirmed")

    top_k = resolve_top_k(config, num_experts)
    if top_k is None:
        return None, (f"{num_experts} experts assembled from per-expert tensors, but the config "
                      f"does not declare how many a token routes to under any of: "
                      f"{', '.join(TOP_K_KEYS)}")

    shared = _shared_key_map(parent, parent_path, container_name, router_name, keys_by_prefix)
    return ExpertContainer(layout=LAYOUT_FUSED_MERGE, path=path, num_experts=num_experts,
                           top_k=top_k, router_path=_qualify(parent_path, router_name),
                           expert_keys=expert_keys, shared_keys=shared,
                           merged_targets=targets), None


def _outermost(containers):
    """Drop containers nested inside another container's experts.

    An expert that is itself built from an indexed list would otherwise be detected twice, once as
    part of its mixture and once on its own. The outer match is the one that streams a whole expert.
    """
    paths = [c.path for c in containers]
    return [c for c in containers
            if not any(c.path.startswith(other + ".") for other in paths if other != c.path)]


# -- the entry point ----------------------------------------------------------------------------------

def detect_expert_layout(layer_module, tensor_shapes, layer_name="", config=None,
                         conversion=None):
    """Work out where one decoder layer keeps its experts.

    Parameters
    ----------
    layer_module : torch.nn.Module
        The decoder layer as transformers built it. It is on the meta device -- no weights, but
        every shape is known, which is all this needs.
    tensor_shapes : Mapping[str, tuple]
        The layer shard's tensor names and shapes, straight from the safetensors header. Detection
        reads no tensor data.
    layer_name : str
        The layer's fully-qualified prefix, so checkpoint keys can be matched to module paths.
    config : optional
        The model config, consulted for the routing width and nothing else.
    conversion : rocketllm.conversion.CheckpointConversion, optional
        What the model class declares about reading its own checkpoints. Needed only where the
        module tree and the checkpoint disagree about the SHAPE of the experts, which is what
        ``fused_merge`` is; without it that case is reported as unstreamable rather than guessed at.

    Returns
    -------
    ExpertLayout
        Empty when the layer is dense, which is the common case and stays cheap.
    """
    prefix = f"{layer_name}." if layer_name else ""
    relative = {}
    for key, shape in tensor_shapes.items():
        if prefix and not key.startswith(prefix):
            continue
        relative[key[len(prefix):]] = tuple(int(d) for d in shape)

    # Checkpoint keys bucketed under every module path that prefixes them, so a candidate container
    # can ask what it owns without rescanning the shard.
    keys_by_prefix = {}
    for rel in relative:
        parts = rel.split(".")
        for cut in range(1, len(parts)):
            keys_by_prefix.setdefault(".".join(parts[:cut]) + ".", []).append(prefix + rel)

    parents = {"": layer_module}
    for path, module in layer_module.named_modules():
        if path:
            parents[path] = module

    containers = []
    skipped = []
    for path in sorted(parents):
        if not path:
            continue
        module = parents[path]
        parent_path, _, container_name = path.rpartition(".")
        parent = parents.get(parent_path)

        children = _indexed_children(module)
        if children is not None:
            container, reason = _module_list_container(path, children, parent, parent_path,
                                                       container_name, keys_by_prefix, config)
        else:
            container, reason = _fused_container(path, module, parent, parent_path, container_name,
                                                 relative, prefix, config, keys_by_prefix,
                                                 conversion)
        if container is not None:
            containers.append(container)
        elif reason:
            skipped.append((path, reason))

    containers = _outermost(containers)

    owned = set()
    for container in containers:
        owned.update(container.owned_keys)
    other_keys = tuple(key for key in tensor_shapes if key not in owned)

    for path, reason in skipped:
        log.info("%s%s looks like an expert container but streams with its layer: %s",
                 prefix, path, reason)

    return ExpertLayout(containers=tuple(containers), other_keys=other_keys,
                        skipped=tuple(skipped))


def summarize(layouts):
    """One line per distinct container shape across the model, for the load-time report.

    A mixture's layers are structurally identical, so reporting each of ninety-four of them says
    nothing the first one did not. Collapsing to distinct shapes keeps the load log readable while
    still surfacing a model whose layers genuinely differ -- an every-other-layer mixture, which is
    a thing that exists.
    """
    seen = {}
    for idx, layout in sorted(layouts.items()):
        for container in layout.containers:
            key = (container.layout, container.path, container.num_experts,
                   -1 if container.top_k is None else container.top_k)
            seen.setdefault(key, []).append(idx)

    lines = []
    for key in sorted(seen):
        layout, path, num_experts, top_k = key
        shown = "unknown" if top_k < 0 else top_k
        lines.append(f"{len(seen[key])} layers x {path}: {layout} layout, {num_experts} experts, "
                     f"top-k {shown}")
    return lines
