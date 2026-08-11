"""HardwareProfile: probe the machine once, derive every tuning knob from what was measured.

There is no reference machine. A constant that was right on the box it was tuned on is a bug on
every other one, so nothing in the engine may pick a number -- it asks here, and what comes back
was measured on the hardware actually running.

Two kinds of number live in this file, and the difference matters:

* **Measured or queried** -- device memory, bandwidths, core counts, dtype support. Never assumed,
  never defaulted. If a backend will not report one, it stays ``None`` and every consumer treats
  that as "unknown", not as zero.
* **Policy factors** -- the dimensionless fractions in ``Policy`` below, such as how much of usable
  device memory a prefetch window may claim. These are design choices, not properties of the
  machine, so they are stated in one place, carried in the profile, printed with the value they
  produced, and overridable. Calling them "measured" would be a lie; burying them in the code
  would be worse.

Every derived knob records the formula that produced it and the inputs that went in, so
``rocketllm profile`` prints something a user can paste into a bug report and an engineer can check
without reading this module.
"""
import concurrent.futures
import dataclasses
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

from . import caps

log = logging.getLogger(__name__)

#: Bumped whenever the derived set changes shape, because a cached profile is replayed verbatim and
#: a stale one would keep supplying a knob that no longer means what it did. Version 2 replaced the
#: absolute budget_hysteresis_bytes with budget_hysteresis_ratio; a cache still carrying the old key
#: would pin the band to a byte count measured against the whole card, which is the bug that change
#: exists to fix.
SCHEMA_VERSION = 2

# Dimensionless policy factors. Not hardware facts -- design choices, gathered here so they are
# visible and overridable rather than sprinkled through the engine as literals.
@dataclasses.dataclass(frozen=True)
class Policy:
    #: Hard ceiling on `reserve`, as a share of total device memory. Stops a pathological
    #: fragmentation measurement from reserving the whole card.
    reserve_ceiling_fraction: float = 0.25
    #: Share of total host RAM left to the OS and everything else on the box.
    os_headroom_fraction: float = 0.10
    #: Share of what remains after that headroom which the host weight cache may claim.
    host_cache_fraction: float = 0.50
    #: Share of the same remainder that reusable host staging buffers may hold. Kept well below
    #: the cache's share: staging only needs the layers in flight, and page-locked memory cannot
    #: be paged out, so over-claiming it hurts the whole machine rather than just this process.
    staging_pool_fraction: float = 0.10
    #: Share of usable device memory a prefetch window may hold. The rest is for the KV cache,
    #: resident weights and activations.
    window_fraction: float = 0.50
    #: How much slower than device memory the slowest weight-serving tier must be before
    #: speculative decoding is worth it: below this, a streaming pass is not the dominant cost.
    speculative_amortization_threshold: float = 10.0
    #: A concurrency level must beat the best so far by this much to be judged a real improvement,
    #: rather than run-to-run noise.
    io_concurrency_improvement: float = 0.10
    #: Read bandwidth under which storage is called out as slow enough to dominate everything.
    #: Expressed in bytes/s; roughly where spinning rust and degraded links live.
    slow_storage_bytes_per_s: float = 150e6
    #: How far a measured device budget must sit from the published one before the move is treated
    #: as real rather than as allocator churn -- as a share of the budget in play, so it means the
    #: same thing on any card. Only a floor: the measured fragmentation ratio is used instead
    #: wherever it is larger, because that is the actual size of the noise this exists to reject.
    budget_hysteresis_fraction: float = 0.02
    #: Consecutive deviating samples required before the published budget moves. Sampling happens
    #: at every layer boundary, so this is cheap to satisfy for a real shift and hard to satisfy
    #: for a single allocation spike.
    budget_hysteresis_samples: int = 3
    #: How far the pin budget must move, as a share of the budget the current plan was built for,
    #: before the plan is worth rebuilding. Damping on top of the budget's own hysteresis: acting on
    #: a small shift means evicting pinned weights and refetching them, which costs more than the
    #: residency it buys back.
    pin_replan_fraction: float = 0.10
    #: Accesses between halvings of the expert popularity counts. Aging is what keeps LFU from
    #: freezing around whatever was hot at the start of a long generation.
    expert_aging_interval: int = 4096
    #: Values sharing one scale in the KV cache. 64 is the KIVI recipe's group.
    kv_group_size: int = 64
    #: Most recent tokens kept unquantized. They are the ones attention weights most sharply, and
    #: quantizing K per channel cannot compute a scale until a whole group of tokens exists, so the
    #: window is what makes the layout possible as well as what protects quality.
    kv_residual_tokens: int = 128
    #: How much room beyond the weights must be free before an automatic KV choice keeps the context
    #: in full precision. Below this the device is the binding constraint and int4 buys context.
    kv_fit_headroom_fraction: float = 0.15
    #: Router firings between rebuilds of the expert pin plan, as a share of the aging interval.
    #: The two describe one timescale: re-ranking experts far more often than the counts they are
    #: ranked by can move only pays for evictions and refetches that the next rebuild undoes.
    expert_replan_fraction: float = 0.125


DEFAULT_POLICY = Policy()

_ENV_PREFIX = "ROCKETLLM_"

#: Knobs a user may pin by environment variable. Value is the parser for the string.
_OVERRIDABLE = {
    "reserve_bytes": int,
    "host_cache_bytes": int,
    "staging_pool_bytes": int,
    "io_workers": int,
    "window_fraction": float,
    # An absolute band is the escape hatch for reproducing a suspected bad measurement; the ratio is
    # what is normally derived, because a band has to scale with the budget it governs.
    "budget_hysteresis_bytes": int,
    "budget_hysteresis_ratio": float,
    "budget_hysteresis_samples": int,
    "pin_replan_bytes": int,
    "expert_aging_interval": int,
    "expert_replan_interval": int,
    "kv_group_size": int,
    "kv_residual_tokens": int,
    "kv_fit_headroom_percent": int,
    "compute_dtype": str,
    "kv_dtype": str,
    "quant_compute_path": str,
    "speculative_recommended": lambda v: v.strip().lower() in ("1", "true", "yes", "on"),
}


def _env_overrides():
    found = {}
    for knob, parse in _OVERRIDABLE.items():
        raw = os.environ.get(_ENV_PREFIX + knob.upper())
        if raw is None:
            continue
        try:
            found[knob] = parse(raw)
        except (TypeError, ValueError):
            log.warning("ignoring %s%s=%r: cannot parse", _ENV_PREFIX, knob.upper(), raw)
    return found


def user_cache_dir():
    """Per-user cache directory, resolved the way each platform expects."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "rocketllm"


# ---------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------

def _timed_copy_bandwidth(make_src, make_dst, nbytes, device, iterations=3):
    """Bytes/s for a copy, taking the best pass so we measure the machine and not the noise."""
    try:
        src, dst = make_src(), make_dst()
    except Exception:
        return None
    best = None
    try:
        for _ in range(iterations):
            caps.synchronize(device)
            start = time.perf_counter()
            dst.copy_(src)
            caps.synchronize(device)
            elapsed = time.perf_counter() - start
            if elapsed > 0:
                best = max(best or 0.0, nbytes / elapsed)
    except Exception:
        return None
    finally:
        del src, dst
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return best


def measure_device_memory_bandwidth(device, budget_bytes=None):
    """On-device copy bandwidth. Counts read+write, which is what a copy actually moves."""
    total, free = caps.device_memory(device)
    if device.type == "cpu":
        # Still meaningful: on CPU the "device" tier is host RAM, and the engine's cost model
        # needs a number for it either way.
        free = free if free else (caps.host_memory()[1] or 0)
        total = total or free
    if not free:
        return None
    # Small enough to leave the machine usable, large enough to leave launch overhead behind.
    nbytes = int(min(budget_bytes or 256 << 20, free * 0.10))
    nbytes -= nbytes % 4
    if nbytes < (1 << 20):
        return None
    count = nbytes // 4
    moved = nbytes * 2  # a copy reads one buffer and writes another
    return _timed_copy_bandwidth(
        lambda: torch.empty(count, dtype=torch.float32, device=device),
        lambda: torch.empty(count, dtype=torch.float32, device=device),
        moved, device)


def measure_host_to_device_bandwidth(device, pinned, budget_bytes=None):
    """Host->device bandwidth over the link, for pinned and for pageable host memory."""
    if device.type == "cpu":
        return None
    if pinned and not caps.supports_pinned_memory(device):
        return None
    _, free = caps.device_memory(device)
    _, host_available = caps.host_memory()
    ceiling = min(free or 0, host_available or 0)
    if not ceiling:
        return None
    nbytes = int(min(budget_bytes or 128 << 20, ceiling * 0.05))
    nbytes -= nbytes % 4
    if nbytes < (1 << 20):
        return None
    count = nbytes // 4

    def make_src():
        host = torch.empty(count, dtype=torch.float32)
        return host.pin_memory() if pinned else host

    return _timed_copy_bandwidth(
        make_src,
        lambda: torch.empty(count, dtype=torch.float32, device=device),
        nbytes, device)


def _drop_page_cache(fd, nbytes):
    """Ask the OS to forget what it just cached, so the next read reaches the device.

    Only Linux exposes this portably. Elsewhere the storage figure includes page-cache hits and
    the profile says so rather than pretending the number is a device measurement.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return False
    try:
        fadvise(fd, 0, nbytes, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


def _shard_files(path, limit=64):
    """Real checkpoint shards under the weights path, largest first."""
    root = Path(path)
    found = []
    try:
        if root.is_file():
            return [(root, root.stat().st_size)]
        for pattern in ("*.safetensors", "*.bin", "*.gguf"):
            for candidate in root.rglob(pattern):
                try:
                    size = candidate.stat().st_size
                except OSError:
                    continue
                if size > (1 << 20):
                    found.append((candidate, size))
                if len(found) >= limit:
                    break
            if len(found) >= limit:
                break
    except OSError:
        return []
    found.sort(key=lambda pair: pair[1], reverse=True)
    return found


def _read_chunk(path, offset, nbytes, drop_cache):
    """Read one chunk and report the bytes actually delivered."""
    try:
        with open(path, "rb") as handle:
            fd = handle.fileno()
            if drop_cache:
                _drop_page_cache(fd, 0)
            handle.seek(offset)
            return len(handle.read(nbytes))
    except OSError:
        return 0


def measure_storage(weights_path, cpu_count, budget_seconds=3.0, chunk_bytes=8 << 20):
    """Read bandwidth of the filesystem the weights are actually on.

    Probes the real shards where there are any, because a different file on a different mount
    answers a different question. Sweeps concurrency to find where the device stops getting
    faster, which is what sets the number of io workers.
    """
    result = {
        "path": str(weights_path) if weights_path else None,
        "probed_real_shards": False,
        "synthetic_probe": False,
        "queue_depth_1_bytes_per_s": None,
        "by_concurrency": {},
        "best_bytes_per_s": None,
        "saturating_concurrency": None,
        "page_cache_dropped": False,
        "page_cache_influence": "unknown",
        "rotational": None,
        "bytes_read": 0,
        "seconds": 0.0,
        "error": None,
    }
    if not weights_path:
        result["error"] = "no weights path given; storage was not probed"
        return result

    root = Path(weights_path)
    if not root.exists():
        result["error"] = f"weights path does not exist: {root}"
        return result

    started = time.perf_counter()
    files = _shard_files(root)
    temp_file = None

    if files:
        result["probed_real_shards"] = True
    else:
        # Nothing to read yet (a fresh cache). Write one file on the *same* filesystem so at
        # least the number describes the right device, and record that it was synthetic.
        try:
            target_dir = root if root.is_dir() else root.parent
            temp_file = target_dir / f".rocketllm-probe-{os.getpid()}.bin"
            payload = os.urandom(1 << 20)
            with open(temp_file, "wb") as handle:
                for _ in range(64):
                    handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            files = [(temp_file, temp_file.stat().st_size)]
            result["synthetic_probe"] = True
        except OSError as exc:
            result["error"] = f"could not probe storage at {root}: {exc}"
            return result

    result["rotational"] = _rotational_hint(files[0][0])

    def sample(concurrency, deadline):
        """Read `concurrency` chunks at once from spread-out offsets; return bytes/s."""
        jobs = []
        for index in range(concurrency):
            path, size = files[index % len(files)]
            span = max(1, size - chunk_bytes)
            jobs.append((path, random.randrange(0, span) if span > 1 else 0))
        drop = result["page_cache_dropped"]
        start = time.perf_counter()
        if concurrency == 1:
            delivered = _read_chunk(jobs[0][0], jobs[0][1], chunk_bytes, drop)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_read_chunk, path, offset, chunk_bytes, drop)
                           for path, offset in jobs]
                delivered = sum(f.result() for f in futures)
        elapsed = time.perf_counter() - start
        result["bytes_read"] += delivered
        if elapsed <= 0 or delivered <= 0 or time.perf_counter() > deadline:
            return None
        return delivered / elapsed

    # Try to defeat the page cache once; if the platform allows it, every read below is a real one.
    try:
        with open(files[0][0], "rb") as handle:
            result["page_cache_dropped"] = _drop_page_cache(handle.fileno(), 0)
    except OSError:
        pass

    deadline = started + budget_seconds
    levels = [level for level in (1, 2, 4, 8, 16) if level <= max(1, cpu_count or 1) * 2]
    for level in levels:
        if time.perf_counter() > deadline:
            break
        rate = sample(level, deadline)
        if rate is None:
            continue
        result["by_concurrency"][str(level)] = rate
        if level == 1:
            result["queue_depth_1_bytes_per_s"] = rate

    if temp_file is not None:
        try:
            temp_file.unlink()
        except OSError:
            pass

    result["seconds"] = time.perf_counter() - started

    if result["by_concurrency"]:
        # The saturating level is the smallest one within noise of the best: paying for more
        # threads than the device rewards just burns CPU the engine needs elsewhere.
        best_rate = max(result["by_concurrency"].values())
        result["best_bytes_per_s"] = best_rate
        for level in sorted(int(k) for k in result["by_concurrency"]):
            if result["by_concurrency"][str(level)] >= best_rate * (1 - DEFAULT_POLICY.io_concurrency_improvement):
                result["saturating_concurrency"] = level
                break
    if not result["page_cache_dropped"]:
        result["page_cache_influence"] = ("likely: this platform cannot drop the page cache, so "
                                          "some reads may have been served from RAM")
    else:
        result["page_cache_influence"] = "reduced: page cache dropped before probing"
    return result


def _rotational_hint(sample_path):
    """Whether the OS calls the backing device rotational. Linux only; elsewhere unknown."""
    try:
        st = os.stat(sample_path)
        major, minor = os.major(st.st_dev), os.minor(st.st_dev)
        queue = Path(f"/sys/dev/block/{major}:{minor}/queue/rotational")
        if not queue.exists():
            # Partitions defer to their parent device.
            queue = Path(f"/sys/dev/block/{major}:{minor}/../queue/rotational")
        if queue.exists():
            return queue.read_text().strip() == "1"
    except (OSError, AttributeError, ValueError):
        pass
    return None


def measure_allocator_behaviour(device):
    """How much the caching allocator holds beyond what is live, and its workspace high-water.

    This is what `reserve` is built from. Asking the allocator beats guessing: fragmentation and
    workspace differ by backend, driver and allocator configuration, and a literal chosen on one
    machine is wrong on the next.
    """
    out = {"peak_allocated_bytes": None, "peak_reserved_bytes": None,
           "fragmentation_ratio": None, "workspace_bytes": None,
           "expandable_segments": "PYTORCH_CUDA_ALLOC_CONF" in os.environ
                                  and "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}
    if device.type != "cuda":
        return out
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base_reserved = torch.cuda.memory_reserved(device)

        # A matmul plus a few odd-sized buffers: enough to make the allocator carve blocks and
        # pull in whatever workspace the backend wants, without taking over the card.
        _, free = caps.device_memory(device)
        side = 1024
        if free and free > (512 << 20):
            side = 2048
        a = torch.empty((side, side), dtype=torch.float32, device=device)
        b = torch.empty((side, side), dtype=torch.float32, device=device)
        scratch = [torch.empty(int(3 << 20) + i * 4099, dtype=torch.uint8, device=device)
                   for i in range(4)]
        c = a @ b
        torch.cuda.synchronize(device)

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        del a, b, c, scratch
        torch.cuda.synchronize(device)

        out["peak_allocated_bytes"] = int(peak_allocated)
        out["peak_reserved_bytes"] = int(peak_reserved)
        if peak_reserved > 0:
            out["fragmentation_ratio"] = max(0.0, (peak_reserved - peak_allocated) / peak_reserved)
        out["workspace_bytes"] = int(max(0, peak_reserved - base_reserved))
    except Exception as exc:  # a probe must never take the process down
        log.debug("allocator probe failed: %s", exc)
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Derivation:
    """One tuning knob: what it came out as, how, and from what."""
    value: object
    formula: str
    inputs: dict
    source: str = "derived"  # or "override"

    def to_dict(self):
        return {"value": self.value, "formula": self.formula,
                "inputs": self.inputs, "source": self.source}


@dataclasses.dataclass
class HardwareProfile:
    schema_version: int
    fingerprint: str
    probed_at: str
    probe_seconds: float

    backend: str
    device_type: str
    device_name: str
    device_count: object
    compute_capability: object
    architecture: object
    driver_version: object
    runtime_version: object

    device_total_bytes: object
    device_free_bytes: object
    host_total_bytes: object
    host_available_bytes: object
    cpu_count: object

    device_memory_bandwidth: object
    host_to_device_pinned_bandwidth: object
    host_to_device_pageable_bandwidth: object
    storage: dict

    dtypes: dict
    pinned_memory: bool
    async_copy_streams: bool
    triton: bool
    fused_4bit: dict
    allocator: dict
    versions: dict

    derived: dict
    warnings: list
    policy: dict

    # -- probing -------------------------------------------------------------------------------

    @classmethod
    def probe(cls, weights_path=None, device=None, policy=DEFAULT_POLICY,
              storage_budget_seconds=3.0, overrides=None):
        started = time.perf_counter()
        dev = caps.resolve_device(device)
        warnings = []

        total_device, free_device = caps.device_memory(dev)
        total_host, available_host = caps.host_memory()
        cpu_count = os.cpu_count()

        dtypes = caps.dtype_support(dev)
        pinned = caps.supports_pinned_memory(dev)
        versions = caps.python_and_torch_versions()
        drv = caps.driver_and_runtime_versions(dev)
        allocator = measure_allocator_behaviour(dev)

        device_bw = measure_device_memory_bandwidth(dev)
        pinned_bw = measure_host_to_device_bandwidth(dev, pinned=True)
        pageable_bw = measure_host_to_device_bandwidth(dev, pinned=False)
        storage = measure_storage(weights_path, cpu_count, budget_seconds=storage_budget_seconds)

        profile = cls(
            schema_version=SCHEMA_VERSION,
            fingerprint="",
            probed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            probe_seconds=0.0,
            backend=caps.backend_of(dev),
            device_type=dev.type,
            device_name=caps.device_name(dev),
            device_count=caps.device_count(dev),
            compute_capability=(list(caps.compute_capability(dev))
                                if caps.compute_capability(dev) else None),
            architecture=caps.architecture_string(dev),
            driver_version=drv["driver"],
            runtime_version=drv["runtime"],
            device_total_bytes=total_device,
            device_free_bytes=free_device,
            host_total_bytes=total_host,
            host_available_bytes=available_host,
            cpu_count=cpu_count,
            device_memory_bandwidth=device_bw,
            host_to_device_pinned_bandwidth=pinned_bw,
            host_to_device_pageable_bandwidth=pageable_bw,
            storage=storage,
            dtypes=dtypes,
            pinned_memory=pinned,
            async_copy_streams=caps.supports_async_copy_streams(dev),
            triton=caps.has_triton(),
            fused_4bit=caps.fused_4bit_kernels(dev),
            allocator=allocator,
            versions=versions,
            derived={},
            warnings=warnings,
            policy=dataclasses.asdict(policy),
        )
        profile.fingerprint = profile._compute_fingerprint()
        profile.derive(policy=policy, overrides=overrides)
        profile.probe_seconds = time.perf_counter() - started
        return profile

    def _compute_fingerprint(self):
        """Identity of the machine, so a cached profile is invalidated when the hardware moves.

        Software versions are deliberately in here too: a torch upgrade can change allocator
        behaviour and kernel availability, which changes what we would derive.
        """
        identity = {
            "schema": SCHEMA_VERSION,
            "backend": self.backend,
            "device": self.device_name,
            "count": self.device_count,
            "cc": self.compute_capability,
            "arch": self.architecture,
            "device_total": self.device_total_bytes,
            "host_total": self.host_total_bytes,
            "cpu": self.cpu_count,
            "torch": self.versions.get("torch"),
            "machine": self.versions.get("machine"),
            "storage_path": self.storage.get("path"),
        }
        blob = json.dumps(identity, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- derivation ----------------------------------------------------------------------------

    def derive(self, policy=DEFAULT_POLICY, overrides=None):
        """Turn measurements into tuning knobs. Overrides win, and say so in the record."""
        chosen = dict(_env_overrides())
        chosen.update(overrides or {})
        self.derived = {}
        self.warnings = list(self.warnings)

        self._derive_reserve(policy, chosen)
        self._derive_host_cache(policy, chosen)
        self._derive_staging_pool(policy, chosen)
        self._derive_io_workers(policy, chosen)
        self._derive_window(policy, chosen)
        self._derive_budget_hysteresis(policy, chosen)
        self._derive_dtypes(policy, chosen)
        self._derive_quant_path(policy, chosen)
        self._derive_speculative(policy, chosen)
        self._collect_warnings(policy)
        return self.derived

    def _set(self, name, value, formula, inputs, overrides):
        if name in overrides:
            self.derived[name] = Derivation(overrides[name], f"manual override (was: {formula})",
                                            inputs, source="override")
        else:
            self.derived[name] = Derivation(value, formula, inputs)

    def _derive_reserve(self, policy, overrides):
        """Device memory held back for activations, workspace and fragmentation.

        Built from what the allocator was measured doing, not from a round number. The
        fragmentation ratio scales with the card; the workspace figure is a floor no smaller than
        what a single matmul was seen to need.
        """
        total = self.device_total_bytes or 0
        frag = self.allocator.get("fragmentation_ratio")
        workspace = self.allocator.get("workspace_bytes") or 0
        ceiling = int(total * policy.reserve_ceiling_fraction)

        if not total:
            value = 0
            formula = "0 (no device memory to reserve on this backend)"
        else:
            scaled = int(total * (frag or 0.0))
            value = min(max(workspace, scaled), ceiling)
            formula = ("min(max(measured_workspace, total_device * measured_fragmentation_ratio), "
                       "total_device * reserve_ceiling_fraction)")
        self._set("reserve_bytes", int(value), formula, {
            "total_device_bytes": total,
            "measured_workspace_bytes": workspace,
            "measured_fragmentation_ratio": frag,
            "reserve_ceiling_fraction": policy.reserve_ceiling_fraction,
            "ceiling_bytes": ceiling,
        }, overrides)

    def _derive_host_cache(self, policy, overrides):
        """How much host RAM the weight cache may hold, leaving the OS real room.

        Driven by *available* RAM, not total: what another process is already using is not ours to
        spend. This can legitimately come out at zero on a loaded machine, and zero must work --
        it means pure streaming with no host tier.
        """
        total = self.host_total_bytes or 0
        available = self.host_available_bytes or 0
        headroom = int(total * policy.os_headroom_fraction)
        value = max(0, int((available - headroom) * policy.host_cache_fraction))
        self._set("host_cache_bytes", value,
                  "max(0, (host_available - host_total * os_headroom_fraction) "
                  "* host_cache_fraction)", {
                      "host_total_bytes": total,
                      "host_available_bytes": available,
                      "os_headroom_fraction": policy.os_headroom_fraction,
                      "os_headroom_bytes": headroom,
                      "host_cache_fraction": policy.host_cache_fraction,
                  }, overrides)

    def _derive_staging_pool(self, policy, overrides):
        """How much host memory reusable staging buffers may hold.

        Staging buffers are where a layer is packed before its single transfer. Page-locking one
        is a synchronizing driver call expensive enough to cost more than the transfer it speeds
        up, so they are pooled rather than allocated per layer -- and a pool needs a budget.

        Taken from the same measured base as the host cache, and as its own share rather than a
        slice of it, so the two cannot silently compete. Zero is a valid answer on a loaded
        machine: it means no pooling and no pinning, which must still work.
        """
        total = self.host_total_bytes or 0
        available = self.host_available_bytes or 0
        headroom = int(total * policy.os_headroom_fraction)
        value = max(0, int((available - headroom) * policy.staging_pool_fraction))
        self._set("staging_pool_bytes", value,
                  "max(0, (host_available - host_total * os_headroom_fraction) "
                  "* staging_pool_fraction)", {
                      "host_total_bytes": total,
                      "host_available_bytes": available,
                      "os_headroom_fraction": policy.os_headroom_fraction,
                      "os_headroom_bytes": headroom,
                      "staging_pool_fraction": policy.staging_pool_fraction,
                  }, overrides)

    def _derive_io_workers(self, policy, overrides):
        """Reader threads: the concurrency the storage was measured to saturate at, capped by CPUs.

        More threads than the device rewards costs CPU the forward pass needs; fewer leaves the
        device idle. The sweep in `measure_storage` answers this directly.
        """
        saturating = self.storage.get("saturating_concurrency")
        cpu = self.cpu_count or 1
        if saturating:
            value = max(1, min(int(saturating), cpu))
            formula = "clamp(measured_saturating_concurrency, 1, cpu_count)"
        else:
            # Storage could not be probed. One worker is the only honest choice: it is the level
            # every device supports, and it never oversubscribes something we did not measure.
            value = 1
            formula = "1 (storage was not probed, so no concurrency was measured)"
        self._set("io_workers", value, formula, {
            "measured_saturating_concurrency": saturating,
            "cpu_count": cpu,
            "measured_by_concurrency": self.storage.get("by_concurrency"),
        }, overrides)

    def _derive_window(self, policy, overrides):
        """Byte budget for the prefetch window of decoder layers."""
        total = self.device_total_bytes or 0
        reserve = self.derived["reserve_bytes"].value
        usable = max(0, total - reserve)
        fraction = overrides.get("window_fraction", policy.window_fraction)
        value = int(usable * fraction)
        self._set("window_budget_bytes", value,
                  "(total_device - reserve) * window_fraction", {
                      "total_device_bytes": total,
                      "reserve_bytes": reserve,
                      "usable_device_bytes": usable,
                      "window_fraction": fraction,
                  }, overrides)
        self.derived["usable_device_bytes"] = Derivation(
            usable, "total_device - reserve",
            {"total_device_bytes": total, "reserve_bytes": reserve})

    def _derive_budget_hysteresis(self, policy, overrides):
        """How much the live device budget must move, and for how long, before anyone acts on it.

        The budget is measured rather than modelled, so it carries the allocator's own noise: blocks
        get carved and released constantly, and free memory jogs up and down by that much without
        anything having really changed. Reacting to that would make the cache evict and refetch on a
        reading it will take back a sample later, which costs a whole streaming pass to learn
        nothing. So the threshold is floored at the fragmentation actually measured on this machine
        -- the size of the noise itself -- and never at a round number chosen elsewhere.

        What is derived is a SHARE, not a byte count, and that distinction is the whole of this
        function. A byte count has to be sized against something, and the only thing a probe knows
        about is the whole card -- but the budget it governs is whatever is left once a model is
        resident, which can be a small fraction of it. Sized off the card, the band was measured at
        1693MB while the live budget was 507MB: three times the quantity it was damping, so no
        change of any size could ever be published and the pin plan never moved. As a share it means
        the same thing on a 4GB card and a 192GB one, which is the property this actually needs.

        The share is the larger of a policy floor and the measured fragmentation ratio, which is a
        ratio already -- applying it to the memory in play is what it was always describing.
        """
        usable = self.derived["usable_device_bytes"].value
        frag = self.allocator.get("fragmentation_ratio")
        ratio = max(float(policy.budget_hysteresis_fraction), float(frag or 0.0))
        self._set("budget_hysteresis_ratio", ratio,
                  "max(budget_hysteresis_fraction, measured_fragmentation_ratio), applied to the "
                  "live budget rather than to the card", {
                      "usable_device_bytes": usable,
                      "budget_hysteresis_fraction": policy.budget_hysteresis_fraction,
                      "measured_fragmentation_ratio": frag,
                      "band_at_full_usable_bytes": int(usable * ratio),
                  }, overrides)
        self._set("budget_hysteresis_samples", int(policy.budget_hysteresis_samples),
                  "policy: consecutive deviating samples before the published budget moves",
                  {"budget_hysteresis_samples": policy.budget_hysteresis_samples}, overrides)

        # Damping for the pin plan, on top of the budget's own hysteresis. Rebuilding a plan is not
        # free -- it evicts weights that were resident and refetches them from storage -- so the
        # move has to be worth a streaming pass before it is acted on.
        self._set("pin_replan_bytes", int(usable * policy.pin_replan_fraction),
                  "usable_device * pin_replan_fraction", {
                      "usable_device_bytes": usable,
                      "pin_replan_fraction": policy.pin_replan_fraction,
                  }, overrides)
        self._set("expert_aging_interval", int(policy.expert_aging_interval),
                  "policy: expert accesses between halvings of the popularity counts",
                  {"expert_aging_interval": policy.expert_aging_interval}, overrides)
        self._set("expert_replan_interval",
                  max(1, int(policy.expert_aging_interval * policy.expert_replan_fraction)),
                  "expert_aging_interval * expert_replan_fraction", {
                      "expert_aging_interval": policy.expert_aging_interval,
                      "expert_replan_fraction": policy.expert_replan_fraction,
                  }, overrides)

    def window_max(self, largest_layer_bytes):
        """How many layers of the largest size fit in the window budget. Never below 1.

        Kept a method rather than a field because it needs the model, which the probe has no
        business knowing. One layer must always be allowed: if even that does not fit, the caller
        raises with the smallest configuration that would work.
        """
        if not largest_layer_bytes or largest_layer_bytes <= 0:
            return 1
        budget = self.derived["window_budget_bytes"].value
        return max(1, int(budget // largest_layer_bytes))

    def _derive_dtypes(self, policy, overrides):
        """Compute dtype, then KV cache dtype, which is an independent decision."""
        if self.dtypes.get("bf16"):
            compute, formula = "bfloat16", "bf16 where the backend supports it"
        elif self.dtypes.get("fp16"):
            compute, formula = "float16", "fp16 fallback: bf16 unsupported here"
        else:
            compute, formula = "float32", "fp32 fallback: neither bf16 nor fp16 available"
        self._set("compute_dtype", compute, formula, {"dtype_support": self.dtypes}, overrides)

        # Independent of the weight format, on purpose. Nothing in this decision may consult it:
        # the checkpoint's format is a property of what someone else produced, and this is a
        # property of how much room the running machine has for a context.
        #
        # The probe cannot settle it either, and says so rather than guessing. Whether memory is the
        # binding constraint is a question about the machine AND the model together -- the same card
        # is roomy for one checkpoint and hopeless for the next -- and the model is not known until
        # something is loaded. So what is derived here is the *inputs* to that choice, and the
        # engine resolves it once it can measure the weights against the budget.
        #
        # An earlier revision hardcoded int4 on the reasoning that a bigger card is bought for a
        # longer context rather than a more precise one. That holds when the model does not fit,
        # which is the case this engine exists for, but it also spent quality on a model that fits
        # resident with room to spare -- where nothing is gained by compressing the context, because
        # no weight was going to be evicted for it.
        usable = self.derived["usable_device_bytes"].value
        self._set("kv_dtype", "auto",
                  "deferred: decided at load from measured weight bytes against the device budget",
                  {
                      "usable_device_bytes": usable,
                      "compute_dtype": compute,
                      "rule": "weights fitting resident with kv_fit_headroom left over -> the "
                              "compute dtype; otherwise int4",
                      "quantization": "K per-channel, V per-token, group size "
                                      f"{policy.kv_group_size}",
                      "note": f"the most recent ~{policy.kv_residual_tokens} tokens stay in an fp16 "
                              f"residual window either way, so the tokens most sensitive to "
                              f"quantization are never quantized",
                  }, overrides)
        self._set("kv_group_size", int(policy.kv_group_size),
                  "policy: values sharing one KV scale",
                  {"kv_group_size": policy.kv_group_size}, overrides)
        self._set("kv_residual_tokens", int(policy.kv_residual_tokens),
                  "policy: most recent tokens left unquantized",
                  {"kv_residual_tokens": policy.kv_residual_tokens}, overrides)
        self._set("kv_fit_headroom_percent",
                  int(round(policy.kv_fit_headroom_fraction * 100)),
                  "policy: free room beyond the weights required before the context is kept exact",
                  {"kv_fit_headroom_fraction": policy.kv_fit_headroom_fraction}, overrides)

    def _derive_quant_path(self, policy, overrides):
        """Compute on packed weights, or dequantize into scratch first."""
        usable = bool(self.fused_4bit.get("any_usable"))
        available = sorted(k for k, v in self.fused_4bit.items()
                           if v and k != "any_usable")
        if usable:
            value = "fused_packed"
            formula = "fused_packed: a fused 4-bit kernel is present and the device can run it"
        else:
            value = "dequant_to_scratch"
            formula = ("dequant_to_scratch: no usable fused 4-bit kernel, so weights are expanded "
                       "into a reusable scratch buffer and computed in the compute dtype")
        self._set("quant_compute_path", value, formula, {
            "kernels_found": available,
            "device_can_run_them": self.device_type in ("cuda", "xpu"),
            "native_fp4": self.dtypes.get("fp4"),
        }, overrides)

    def _derive_speculative(self, policy, overrides):
        """Whether speculative decoding is likely to pay off here.

        It wins by amortizing one streaming pass over several tokens, so it pays exactly when
        moving the weights dominates computing with them. The ratio of device memory bandwidth to
        the slowest tier that has to serve weights is that comparison, measured.
        """
        tiers = [b for b in (self.host_to_device_pinned_bandwidth
                             or self.host_to_device_pageable_bandwidth,
                             self.storage.get("best_bytes_per_s")) if b]
        slowest = min(tiers) if tiers else None
        device_bw = self.device_memory_bandwidth
        ratio = None
        if slowest and device_bw and slowest > 0:
            ratio = device_bw / slowest
        if ratio is None:
            value = None
            formula = ("unavailable: needs both device memory bandwidth and a weight-serving tier "
                       "bandwidth, and at least one was not measured")
        else:
            value = ratio >= policy.speculative_amortization_threshold
            formula = ("device_memory_bandwidth / slowest_weight_tier_bandwidth "
                       ">= speculative_amortization_threshold")
        self._set("speculative_recommended", value, formula, {
            "device_memory_bandwidth": device_bw,
            "slowest_weight_tier_bandwidth": slowest,
            "amortization_ratio": ratio,
            "threshold": policy.speculative_amortization_threshold,
        }, overrides)

    def _collect_warnings(self, policy):
        """Everything the user should be told once, loudly, before a long run."""
        found = []
        if self.dtypes.get("bf16") is False and self.dtypes.get("fp16"):
            found.append("bf16 is not available on this device; falling back to fp16. fp16's "
                         "range overflows on very deep models and can silently corrupt output.")
        if not self.pinned_memory:
            found.append("pinned host memory is unavailable; the host tier will use pageable "
                         "buffers and transfers will be slower.")
        if not self.async_copy_streams:
            found.append("async copy streams are unavailable; transfers use the synchronous path.")
        if not self.fused_4bit.get("any_usable"):
            found.append("no usable fused 4-bit kernel was found; packed weights will be "
                         "dequantized into scratch before the matmul.")
        if self.derived.get("host_cache_bytes") and self.derived["host_cache_bytes"].value == 0:
            found.append("host cache budget computed to 0: not enough free RAM for a host tier, "
                         "so every layer will stream from storage on every pass.")

        best = self.storage.get("best_bytes_per_s")
        if self.storage.get("rotational"):
            found.append("the weights are on a device the OS reports as rotational; expect "
                         "streaming to dominate run time by a wide margin.")
        if best and best < policy.slow_storage_bytes_per_s:
            found.append(
                f"storage read bandwidth measured at {best / 1e6:.0f} MB/s, which is slow enough "
                f"to dominate everything else: a model of S bytes costs about S/{best / 1e9:.2f}GB/s "
                f"per streaming pass.")
        if self.storage.get("synthetic_probe"):
            found.append("no checkpoint shards were found at the weights path, so storage was "
                         "measured with a temporary file on the same filesystem.")
        if self.storage.get("error"):
            found.append(f"storage was not probed: {self.storage['error']}")
        if self.device_type == "cuda" and not self.allocator.get("expandable_segments"):
            found.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not set; the caching "
                         "allocator will fragment more under a streaming workload. Set it before "
                         "the first CUDA allocation.")
        self.warnings = found

    # -- the caching rule ----------------------------------------------------------------------

    def device_memory_budget(self, device=None):
        """Bytes the weight cache may currently hold on the device.

        mem_get_info alone under-reports, because the caching allocator sits on blocks it has
        already freed and the driver counts those as in use. Adding them back is the difference
        between a cache that fills the card and one that gives up early.
        """
        dev = caps.resolve_device(device) if device is None else torch.device(device)
        _, free = caps.device_memory(dev)
        if free is None:
            return 0
        held = 0
        if dev.type == "cuda":
            try:
                held = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
            except Exception:
                held = 0
        return max(0, free + held - self.derived["reserve_bytes"].value)

    # -- serialisation -------------------------------------------------------------------------

    def to_dict(self):
        out = dataclasses.asdict(self)
        out["derived"] = {name: d.to_dict() for name, d in self.derived.items()}
        return out

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        derived = {name: Derivation(**value) for name, value in (data.get("derived") or {}).items()}
        data["derived"] = derived
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    # -- caching -------------------------------------------------------------------------------

    @staticmethod
    def cache_path(fingerprint):
        return user_cache_dir() / f"profile-{fingerprint}.json"

    def save(self):
        path = self.cache_path(self.fingerprint)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n",
                            encoding="utf-8")
        except OSError as exc:
            log.warning("could not cache the hardware profile at %s: %s", path, exc)
        return path

    @classmethod
    def load_or_probe(cls, weights_path=None, device=None, reprofile=False,
                      policy=DEFAULT_POLICY, overrides=None, storage_budget_seconds=3.0):
        """Return a cached profile for this machine, or probe and cache one.

        The cache is keyed by a hardware fingerprint, so moving to a different machine, swapping a
        card or upgrading torch invalidates it automatically: the key simply stops matching and a
        fresh probe runs. `reprofile` forces that anyway.
        """
        if not reprofile:
            cached = cls._load_matching(weights_path, device)
            if cached is not None:
                # Derivations are cheap and depend on live free memory and on overrides, so they
                # are recomputed even on a cache hit. Only the measurements are reused.
                cached.derive(policy=policy, overrides=overrides)
                return cached
        profile = cls.probe(weights_path=weights_path, device=device, policy=policy,
                            overrides=overrides, storage_budget_seconds=storage_budget_seconds)
        profile.save()
        return profile

    @classmethod
    def _load_matching(cls, weights_path, device):
        """Find a cached profile whose fingerprint matches this machine, without re-measuring."""
        directory = user_cache_dir()
        if not directory.is_dir():
            return None
        try:
            dev = caps.resolve_device(device)
            probe = cls._identity_only(dev, weights_path)
        except Exception:
            return None
        path = cls.cache_path(probe)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        try:
            return cls.from_dict(data)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _identity_only(cls, dev, weights_path):
        """Compute the fingerprint from cheap queries alone, to look the cache up without probing."""
        total_device, _ = caps.device_memory(dev)
        total_host, _ = caps.host_memory()
        identity = {
            "schema": SCHEMA_VERSION,
            "backend": caps.backend_of(dev),
            "device": caps.device_name(dev),
            "count": caps.device_count(dev),
            "cc": list(caps.compute_capability(dev)) if caps.compute_capability(dev) else None,
            "arch": caps.architecture_string(dev),
            "device_total": total_device,
            "host_total": total_host,
            "cpu": os.cpu_count(),
            "torch": torch.__version__,
            "machine": caps.python_and_torch_versions()["machine"],
            "storage_path": str(weights_path) if weights_path else None,
        }
        blob = json.dumps(identity, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- presentation --------------------------------------------------------------------------

    def describe(self):
        """The whole profile as text, every derived value next to the formula that made it."""
        lines = []
        add = lines.append
        rule = "=" * 78

        add(rule)
        add("  RocketLLM hardware profile")
        add(rule)
        add(f"  fingerprint   {self.fingerprint}")
        add(f"  probed        {self.probed_at} in {self.probe_seconds:.2f}s")
        add(f"  cache file    {self.cache_path(self.fingerprint)}")
        add("")
        add("  DEVICE")
        add(f"    backend           {self.backend} ({self.device_type})")
        add(f"    name              {self.device_name}")
        add(f"    count             {_fmt(self.device_count)}")
        add(f"    compute cap.      {_fmt(_cc(self.compute_capability))}")
        add(f"    architecture      {_fmt(self.architecture)}")
        add(f"    driver / runtime  {_fmt(self.driver_version)} / {_fmt(self.runtime_version)}")
        add(f"    device memory     {_bytes(self.device_total_bytes)} total, "
            f"{_bytes(self.device_free_bytes)} free")
        add(f"    host RAM          {_bytes(self.host_total_bytes)} total, "
            f"{_bytes(self.host_available_bytes)} available")
        add(f"    cpu cores         {_fmt(self.cpu_count)}")

        add("")
        add("  MEASURED BANDWIDTH")
        add(f"    device memory     {_bw(self.device_memory_bandwidth)}")
        add(f"    host->device pinned   {_bw(self.host_to_device_pinned_bandwidth)}")
        add(f"    host->device pageable {_bw(self.host_to_device_pageable_bandwidth)}")

        storage = self.storage
        add("")
        add("  STORAGE  (the filesystem the weights are on)")
        add(f"    path              {_fmt(storage.get('path'))}")
        add(f"    probed            {'real shards' if storage.get('probed_real_shards') else ''}"
            f"{'synthetic file on the same filesystem' if storage.get('synthetic_probe') else ''}"
            f"{'not probed' if storage.get('error') else ''}"
            f"  ({_bytes(storage.get('bytes_read'))} read in {storage.get('seconds', 0):.2f}s)")
        add(f"    queue depth 1     {_bw(storage.get('queue_depth_1_bytes_per_s'))}")
        for level in sorted(int(k) for k in (storage.get("by_concurrency") or {})):
            marker = "  <- saturates" if level == storage.get("saturating_concurrency") else ""
            add(f"    concurrency {level:<5} {_bw(storage['by_concurrency'][str(level)])}{marker}")
        add(f"    rotational        {_fmt(storage.get('rotational'))}")
        add(f"    page cache        {storage.get('page_cache_influence')}")

        add("")
        add("  CAPABILITIES  (queried, never inferred from the device name)")
        add(f"    bf16 {_fmt(self.dtypes.get('bf16'))}    fp16 {_fmt(self.dtypes.get('fp16'))}    "
            f"fp8 {_fmt(self.dtypes.get('fp8'))}    fp4 {_fmt(self.dtypes.get('fp4'))}")
        add(f"    pinned memory     {self.pinned_memory}")
        add(f"    async streams     {self.async_copy_streams}")
        add(f"    triton            {self.triton}")
        kernels = ", ".join(k for k, v in self.fused_4bit.items() if v and k != "any_usable")
        add(f"    fused 4-bit       {kernels or 'none found'}"
            f"  (usable here: {self.fused_4bit.get('any_usable')})")
        if self.allocator.get("fragmentation_ratio") is not None:
            add(f"    allocator         fragmentation {self.allocator['fragmentation_ratio']:.1%}, "
                f"workspace {_bytes(self.allocator.get('workspace_bytes'))}")

        add("")
        add("  DERIVED TUNING KNOBS")
        for name, derivation in self.derived.items():
            value = derivation.value
            rendered = _bytes(value) if name.endswith("_bytes") else _fmt(value)
            tag = "  [OVERRIDDEN]" if derivation.source == "override" else ""
            add(f"    {name:<26}{rendered}{tag}")
            add(f"      formula: {derivation.formula}")
            for key, val in derivation.inputs.items():
                shown = _bytes(val) if key.endswith("_bytes") else _fmt(val)
                add(f"        {key:<34}{shown}")
            add("")

        if self.warnings:
            add("  WARNINGS")
            for warning in self.warnings:
                add(f"    - {warning}")
            add("")
        add(rule)
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------

def _fmt(value):
    if value is None:
        return "unavailable"
    return str(value)


def _cc(value):
    if not value:
        return None
    return ".".join(str(part) for part in value)


def _bytes(value):
    if value is None:
        return "unavailable"
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    step = 1024.0
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(amount) < step or unit == "TB":
            return f"{int(amount)} B" if unit == "B" else f"{amount:.2f} {unit}"
        amount /= step
    return f"{amount:.2f} TB"


def _bw(value):
    if value is None:
        return "unavailable"
    if value >= 1e9:
        return f"{value / 1e9:.2f} GB/s"
    return f"{value / 1e6:.0f} MB/s"
