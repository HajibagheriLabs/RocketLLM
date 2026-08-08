"""Capability queries and the device abstraction.

Two layers live here.

The **query layer** is a set of free functions, each answering one question about the machine
actually running, and answering it by asking the backend or by attempting the operation -- never by
recognising a device name. A device called "RTX 5090" tells you nothing a query cannot tell you
better, and it tells you nothing at all about the card released next month. Each returns
``True``/``False`` when it knows and ``None`` when the backend cannot say; ``None`` means unknown,
and callers must not fold it into ``False``.

The **device abstraction** is :class:`DeviceCaps` and one subclass per backend. It covers what the
streaming path needs -- memory accounting, pinned staging buffers, copy streams and events, cache
release, the compute dtype, and the fused-kernel decision -- so the engine calls the same handful of
methods everywhere and never branches on a backend itself.

Missing hardware features are not errors. Every one of them has a defined fallback that still
produces correct output, each fallback is announced exactly once when the device is first resolved,
and nothing here raises because an optional accelerator feature is absent.
"""
import dataclasses
import gc
import importlib
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading

import torch

log = logging.getLogger(__name__)

# Backends this build of torch could possibly talk to. Order is fastest-first for auto-selection.
_BACKEND_ORDER = ("cuda", "rocm", "xpu", "mps", "cpu")

_warned = set()
_warn_lock = threading.Lock()


def _announce(key, message, level=logging.INFO):
    """Say something once per process, never per layer.

    A streaming run touches every module hundreds of times per token. A fallback that logs at the
    point it is taken would bury the console; the user needs to know once, at load, and then be
    left alone.
    """
    with _warn_lock:
        if key in _warned:
            return False
        _warned.add(key)
    log.log(level, message)
    return True


def reset_announcements():
    """Forget what has already been announced. For tests that assert on the once-only behaviour."""
    with _warn_lock:
        _warned.clear()


def _spec_exists(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_import_cache = {}


def _importable(name):
    """Whether a module genuinely imports, not merely whether it is on disk.

    A kernel package that is installed but broken -- wrong CUDA build, missing shared object -- is
    worse than an absent one, because presence alone would route the engine down a path that then
    fails mid-forward. So the gate is a real import, done once and remembered.
    """
    if name in _import_cache:
        return _import_cache[name]
    result = False
    if _spec_exists(name):
        try:
            importlib.import_module(name)
            result = True
        except Exception as exc:  # a broken optional package must not take the process down
            log.debug("optional package %s is present but does not import: %s", name, exc)
            result = False
    _import_cache[name] = result
    return result


# ---------------------------------------------------------------------------------------------
# backend and device
# ---------------------------------------------------------------------------------------------

def is_rocm_build():
    """ROCm masquerades as 'cuda' in torch, so the HIP version is what distinguishes it."""
    return getattr(torch.version, "hip", None) is not None


def available_backends():
    found = []
    if torch.cuda.is_available():
        found.append("rocm" if is_rocm_build() else "cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        found.append("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        found.append("mps")
    found.append("cpu")
    return [b for b in _BACKEND_ORDER if b in found]


def resolve_device(requested=None):
    """Pick the device to run on: the caller's choice, else the fastest backend present."""
    if requested is not None:
        return torch.device(requested)
    for backend in available_backends():
        if backend in ("cuda", "rocm"):
            return torch.device("cuda:0")
        if backend == "xpu":
            return torch.device("xpu:0")
        if backend == "mps":
            return torch.device("mps")
    return torch.device("cpu")


def backend_of(device):
    """The vendor-level backend behind a torch device type."""
    if device.type == "cuda":
        return "rocm" if is_rocm_build() else "cuda"
    return device.type


def device_count(device):
    kind = device.type
    try:
        if kind == "cuda":
            return int(torch.cuda.device_count())
        if kind == "xpu" and hasattr(torch, "xpu"):
            return int(torch.xpu.device_count())
        if kind == "mps":
            return 1
    except Exception:
        return None
    return os.cpu_count() if kind == "cpu" else None


def device_name(device):
    kind = device.type
    try:
        if kind == "cuda":
            return torch.cuda.get_device_name(device)
        if kind == "xpu" and hasattr(torch, "xpu"):
            return torch.xpu.get_device_properties(device).name
    except Exception:
        pass
    if kind == "mps":
        return f"Apple Silicon ({platform.machine()})"
    return platform.processor() or platform.machine() or "cpu"


def synchronize(device):
    """Block until queued device work finishes, where the backend has such a notion."""
    kind = device.type
    try:
        if kind == "cuda":
            torch.cuda.synchronize(device)
        elif kind == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        elif kind == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
    except Exception:
        pass


def compute_capability(device):
    """CUDA compute capability as (major, minor); ``None`` where the concept does not apply."""
    if device.type == "cuda" and not is_rocm_build():
        try:
            return tuple(torch.cuda.get_device_capability(device))
        except Exception:
            return None
    return None


def architecture_string(device):
    """ROCm gfx target, or the CUDA capability rendered as a string.

    Only ROCm's ``gcnArchName`` is meaningful: on NVIDIA builds torch populates the same field
    with the marketing name, which is exactly the kind of string nothing here should key off.
    """
    if device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        if is_rocm_build():
            gcn = getattr(props, "gcnArchName", None)
            return str(gcn) if gcn else None
        return f"sm_{props.major}{props.minor}"
    except Exception:
        return None


def driver_and_runtime_versions(device):
    """Whatever the backend will tell us about its driver and runtime."""
    info = {"driver": None, "runtime": None}
    kind = device.type
    if kind == "cuda":
        info["runtime"] = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
        # torch has no driver-version API, so ask the vendor tool if it happens to be installed.
        # Its absence is not an error; the field simply stays unavailable.
        tool = shutil.which("rocm-smi") if is_rocm_build() else shutil.which("nvidia-smi")
        if tool and not is_rocm_build():
            try:
                out = subprocess.run([tool, "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    info["driver"] = out.stdout.strip().splitlines()[0].strip()
            except (OSError, subprocess.SubprocessError):
                pass
    elif kind == "xpu":
        info["runtime"] = getattr(torch.version, "xpu", None)
    return info


# ---------------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------------

def device_memory(device):
    """(total, free) device bytes. Either may be ``None`` where the backend will not say."""
    kind = device.type
    try:
        if kind == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            return int(total), int(free)
        if kind == "xpu" and hasattr(torch, "xpu"):
            if hasattr(torch.xpu, "mem_get_info"):
                free, total = torch.xpu.mem_get_info(device)
                return int(total), int(free)
            props = torch.xpu.get_device_properties(device)
            total = int(props.total_memory)
            allocated = int(torch.xpu.memory_allocated(device))
            return total, max(0, total - allocated)
        if kind == "mps" and hasattr(torch, "mps"):
            total = int(torch.mps.recommended_max_memory())
            used = int(torch.mps.driver_allocated_memory())
            return total, max(0, total - used)
    except Exception:
        pass
    return None, None


def host_memory():
    """(total, available) host bytes, from the OS, with no third-party dependency."""
    if os.name == "nt":
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return None, None

    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = int(os.sysconf("SC_PHYS_PAGES") * page)
    except (ValueError, OSError, AttributeError):
        return None, None

    # MemAvailable is the only figure that accounts for reclaimable cache; free pages alone
    # badly understate what a new allocation could actually get.
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return total, int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return total, int(os.sysconf("SC_AVPHYS_PAGES") * page)
    except (ValueError, OSError, AttributeError):
        return total, None


# ---------------------------------------------------------------------------------------------
# dtype support
#
# Attempted, not assumed. A dtype can exist in torch, be allocatable, and still have no arithmetic
# behind it, so where it is cheap these probes run the actual operation.
# ---------------------------------------------------------------------------------------------

def _can_matmul(device, dtype):
    try:
        a = torch.ones((8, 8), dtype=dtype, device=device)
        result = a @ a
        # Forcing a read catches backends that fail lazily.
        float(result.float().sum().item())
        return True
    except Exception:
        return False


def supports_bf16(device):
    if device.type == "cuda" and not is_rocm_build():
        try:
            if not torch.cuda.is_bf16_supported():
                return False
        except Exception:
            pass
    return _can_matmul(device, torch.bfloat16)


def supports_fp16(device):
    return _can_matmul(device, torch.float16)


def supports_fp8(device):
    """Real fp8 arithmetic, which means a scaled matmul -- not merely an allocatable dtype."""
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        return False
    scaled_mm = getattr(torch, "_scaled_mm", None)
    if scaled_mm is None:
        return False
    try:
        a = torch.zeros((16, 16), dtype=dtype, device=device)
        b = torch.zeros((16, 16), dtype=dtype, device=device).t()
        scale = torch.ones((), dtype=torch.float32, device=device)
        scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


def supports_fp4(device):
    """Native 4-bit float. Everywhere it is missing, 4-bit formats dequantize before the matmul."""
    for name in ("float4_e2m1fn_x2", "float4_e2m1fn"):
        dtype = getattr(torch, name, None)
        if dtype is None:
            continue
        try:
            torch.zeros((16, 16), dtype=dtype, device=device)
            return True
        except Exception:
            continue
    return False


def dtype_support(device):
    return {
        "bf16": supports_bf16(device),
        "fp16": supports_fp16(device),
        "fp8": supports_fp8(device),
        "fp4": supports_fp4(device),
    }


# ---------------------------------------------------------------------------------------------
# transfer and kernel capabilities
# ---------------------------------------------------------------------------------------------

def supports_pinned_memory(device):
    """Page-locked host memory. Attempted rather than assumed from the backend name."""
    if device.type not in ("cuda", "xpu"):
        return False
    try:
        torch.empty(1024, dtype=torch.uint8).pin_memory()
        return True
    except Exception:
        return False


def supports_async_copy_streams(device):
    if device.type != "cuda":
        return False
    try:
        stream = torch.cuda.Stream(device=device)
        return stream is not None
    except Exception:
        return False


def has_triton():
    return _importable("triton")


#: Fused 4-bit matmul providers, in the order we would rather use them.
_FUSED_4BIT_PACKAGES = ("bitsandbytes", "awq_ext", "gptqmodel", "exllamav2", "marlin_kernels")


def fused_4bit_kernels(device):
    """Which fused 4-bit paths this machine could actually take.

    A fused path needs *both* an importable kernel and a device able to run it, which is why the
    packages are imported rather than merely located, and why the verdict is reported separately
    from the inventory. Where nothing is usable the engine dequantizes into scratch and computes
    in the compute dtype -- correct, just slower.
    """
    found = {
        # Shipped with torch itself on some builds; by far the most portable fused int4 path, and
        # an attribute check rather than an import, so it costs nothing.
        "torch_int4pack": hasattr(torch.ops.aten, "_weight_int4pack_mm"),
    }
    for package in _FUSED_4BIT_PACKAGES:
        found[package] = _importable(package)

    # Fused kernels for these formats are written for the accelerator; on CPU they are never taken.
    found["any_usable"] = bool(device.type in ("cuda", "xpu") and any(found.values()))
    return found


def python_and_torch_versions():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }


# ---------------------------------------------------------------------------------------------
# the device abstraction
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class MemoryReport:
    """What the device has, and how confident we are about it."""
    total: object
    free: object
    reserved: object
    allocated: object
    #: Bytes the weight cache may claim right now, after `reserve`.
    budget: int
    #: True when `free` had to be inferred rather than read from the driver.
    estimated: bool
    note: str

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FusedPlan:
    """Whether to compute on packed 4-bit weights, or expand them into scratch first."""
    path: str            # "fused_packed" | "dequant_to_scratch"
    kernel: object       # the provider that won, or None
    reason: str

    @property
    def fused(self):
        return self.path == "fused_packed"

    def to_dict(self):
        return dataclasses.asdict(self)


class _SyncStream:
    """Stand-in for a copy stream on backends that have none.

    Presents the same surface as a real stream so the streaming path is written once. Every method
    is a no-op because the work has already happened: without streams the copy was synchronous.
    """

    is_async = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synchronize(self):
        pass

    def wait_event(self, event):
        pass

    def record_event(self, event=None):
        return event or _SyncEvent()


class _SyncEvent:
    is_async = False

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass

    def wait(self, stream=None):
        pass

    def query(self):
        return True


class _CudaStream:
    """A real copy stream, wrapped so callers use it exactly like the synchronous stand-in."""

    is_async = True

    def __init__(self, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._ctx = None

    def __enter__(self):
        self._ctx = torch.cuda.stream(self.stream)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        ctx, self._ctx = self._ctx, None
        return ctx.__exit__(*exc) if ctx else False

    def synchronize(self):
        self.stream.synchronize()

    def wait_event(self, event):
        self.stream.wait_event(getattr(event, "event", event))

    def record_event(self, event=None):
        event = event or _CudaEvent()
        event.record(self)
        return event


class _CudaEvent:
    is_async = True

    def __init__(self):
        self.event = torch.cuda.Event()

    def record(self, stream=None):
        self.event.record(getattr(stream, "stream", None) or torch.cuda.current_stream())

    def synchronize(self):
        self.event.synchronize()

    def wait(self, stream=None):
        self.event.wait(getattr(stream, "stream", None) or torch.cuda.current_stream())

    def query(self):
        return self.event.query()


class DeviceCaps:
    """What the streaming path may ask of a device, with a defined answer on every backend.

    Subclasses fill in the parts a backend can do natively. Everything a backend cannot do falls
    back to something slower that still produces correct output, announced once at load.
    """

    backend = "cpu"
    #: Support tier from docs/HARDWARE.md, reported so the engine can be honest about what to expect.
    tier = 4

    def __init__(self, device):
        self.device = device
        self._fused_plan = None
        self._compute_dtype = None

    # -- construction --------------------------------------------------------------------------

    def __repr__(self):
        return f"<{type(self).__name__} {self.device} tier={self.tier}>"

    # -- capability gates ----------------------------------------------------------------------

    @property
    def supports_bf16(self):
        return supports_bf16(self.device)

    @property
    def supports_fp16(self):
        return supports_fp16(self.device)

    @property
    def supports_fp8(self):
        return supports_fp8(self.device)

    @property
    def supports_fp4(self):
        return supports_fp4(self.device)

    @property
    def can_pin_memory(self):
        return False

    @property
    def has_async_streams(self):
        return False

    # -- memory accounting ---------------------------------------------------------------------

    def memory(self, reserve_bytes=0):
        """Totals plus the budget the weight cache may claim, given `reserve`.

        The base implementation is the conservative one: no allocator introspection, so whatever
        the backend calls free is taken at face value and the report says it is an estimate.
        """
        total, free = device_memory(self.device)
        if total is None and free is None:
            total, free = host_memory()
            note = ("host memory stands in for device memory on this backend; they are the same "
                    "pool")
        else:
            note = "reported by the backend; no allocator introspection available here"
        budget = max(0, (free or 0) - reserve_bytes)
        return MemoryReport(total=total, free=free, reserved=None, allocated=None,
                            budget=budget, estimated=True, note=note)

    # -- host staging --------------------------------------------------------------------------

    def pinned_empty(self, shape, dtype):
        """A host buffer for staging weights, page-locked where that is possible.

        Never raises for want of pinned memory: an unpinnable buffer is a slower buffer, not a
        failure, and the difference is announced once rather than at every layer.
        """
        buffer = torch.empty(shape, dtype=dtype)
        return self.try_pin(buffer)

    def try_pin(self, tensor):
        """Return a pinned copy of `tensor` if the backend supports pinning, else `tensor`."""
        if not self.can_pin_memory:
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError as exc:
            # Pinned memory is a finite OS resource; running out is expected under load and must
            # degrade rather than abort.
            _announce("pin-exhausted",
                      f"ran out of pinned host memory ({exc}); staging buffers fall back to "
                      f"pageable memory, which transfers more slowly but works identically",
                      logging.INFO)
            return tensor

    # -- streams and events --------------------------------------------------------------------

    def copy_stream(self):
        """A stream for weight transfers. Synchronous stand-in where the backend has none."""
        return _SyncStream()

    def event(self):
        return _SyncEvent()

    def synchronize(self):
        synchronize(self.device)

    # -- cache release -------------------------------------------------------------------------

    def empty_cache(self):
        """Hand freed blocks back. On CPU that means the garbage collector."""
        gc.collect()

    # -- dtype and kernel decisions --------------------------------------------------------------

    def select_compute_dtype(self, requested=None):
        """The dtype to run in: what was asked for, if this device can actually do it.

        Falling back from bf16 to fp16 is not a silent downgrade. fp16's exponent range overflows
        on very deep models and the corruption shows up as plausible-looking wrong tokens rather
        than an error, so it is stated once, loudly, naming the risk.
        """
        if isinstance(requested, str):
            requested = getattr(torch, requested, None)

        if requested is torch.bfloat16 or requested is None:
            if self.supports_bf16:
                return torch.bfloat16
            # Warned whether or not bf16 was named explicitly: what matters is that the run ends
            # up in fp16 *because the device cannot do better*, and the user cannot see that from
            # the output. A checkpoint that simply declared no dtype lands here too.
            _announce(
                f"bf16-{self.backend}",
                "bf16 is not supported on this device, so the run falls back to fp16. "
                "fp16's range is narrow enough to overflow to inf/NaN on very deep models, "
                "which corrupts output silently rather than raising. If results look wrong, "
                "this is the first thing to suspect.",
                logging.WARNING)
            if self.supports_fp16:
                return torch.float16
            _announce(f"fp32-{self.backend}",
                      "neither bf16 nor fp16 is supported on this device; running in fp32, which "
                      "is correct but doubles the bytes moved per token.", logging.WARNING)
            return torch.float32

        if requested is torch.float16 and not self.supports_fp16:
            _announce(f"nofp16-{self.backend}",
                      "fp16 is not supported on this device; running in fp32 instead.",
                      logging.WARNING)
            return torch.bfloat16 if self.supports_bf16 else torch.float32

        return requested

    @property
    def compute_dtype(self):
        if self._compute_dtype is None:
            self._compute_dtype = self.select_compute_dtype(None)
        return self._compute_dtype

    def fused_4bit_plan(self):
        """Whether packed 4-bit weights can be computed on directly here.

        Two conditions, both required: a kernel that imports, and a device that can run it. Either
        one missing selects the dequant-to-scratch path, which every backend can do.
        """
        if self._fused_plan is not None:
            return self._fused_plan

        inventory = fused_4bit_kernels(self.device)
        winner = next((name for name in ("torch_int4pack",) + _FUSED_4BIT_PACKAGES
                       if inventory.get(name)), None)

        if inventory.get("any_usable") and winner:
            plan = FusedPlan("fused_packed", winner,
                             f"{winner} imported successfully and {self.backend} can run it, so "
                             f"weights stay packed through the matmul")
        else:
            if not winner:
                reason = ("no fused 4-bit kernel could be imported, so packed weights are "
                          "expanded into a reusable scratch buffer and computed in the compute "
                          "dtype")
            else:
                reason = (f"{winner} is available but the {self.backend} backend cannot run fused "
                          f"4-bit kernels, so weights are expanded into scratch first")
            plan = FusedPlan("dequant_to_scratch", None, reason)

        self._fused_plan = plan
        return plan

    # -- degradation announcements ---------------------------------------------------------------

    def announce_degradations(self):
        """State every fallback this device implies, once, at load.

        Called when the device is first resolved, so a user learns what they are getting before a
        long run rather than inferring it from the speed afterwards.
        """
        if not self.can_pin_memory:
            _announce(f"nopin-{self.backend}",
                      f"pinned host memory is unavailable on the {self.backend} backend; staging "
                      f"buffers use pageable memory and transfers are slower.", logging.INFO)
        if not self.has_async_streams:
            _announce(f"nostream-{self.backend}",
                      f"async copy streams are unavailable on the {self.backend} backend; weight "
                      f"transfers use the synchronous path and do not overlap with compute.",
                      logging.INFO)
        plan = self.fused_4bit_plan()
        if not plan.fused:
            _announce(f"nofused-{self.backend}", plan.reason, logging.INFO)
        # Touching the property is what emits the bf16 warning, if there is one to emit.
        _ = self.compute_dtype

    def summary(self):
        """The decision table for this device, for logs and bug reports."""
        report = self.memory()
        plan = self.fused_4bit_plan()
        return {
            "backend": self.backend,
            "tier": self.tier,
            "device": str(self.device),
            "device_name": device_name(self.device),
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "bf16": self.supports_bf16,
            "fp16": self.supports_fp16,
            "fp8": self.supports_fp8,
            "fp4": self.supports_fp4,
            "pinned_memory": self.can_pin_memory,
            "async_streams": self.has_async_streams,
            "quant_path": plan.path,
            "quant_kernel": plan.kernel,
            "memory_total": report.total,
            "memory_free": report.free,
            "memory_estimated": report.estimated,
            "memory_note": report.note,
        }


class CudaCaps(DeviceCaps):
    backend = "cuda"

    @property
    def tier(self):
        # Tier is a statement about capability, so it comes from the capability, not the name.
        cc = compute_capability(self.device)
        if cc is None:
            return 2
        return 1 if cc >= (8, 0) else 2

    @property
    def can_pin_memory(self):
        return supports_pinned_memory(self.device)

    @property
    def has_async_streams(self):
        return supports_async_copy_streams(self.device)

    def memory(self, reserve_bytes=0):
        """The budget formula from the caching rules, with real allocator introspection.

        ``mem_get_info`` alone under-reports: the caching allocator holds blocks it has already
        freed, and the driver still counts those as in use. Adding the difference back is the
        gap between a cache that fills the card and one that gives up with room to spare.
        """
        total, free = device_memory(self.device)
        reserved = allocated = None
        try:
            reserved = int(torch.cuda.memory_reserved(self.device))
            allocated = int(torch.cuda.memory_allocated(self.device))
        except Exception:
            pass

        if free is None:
            return super().memory(reserve_bytes)

        held = max(0, (reserved or 0) - (allocated or 0))
        budget = max(0, free + held - reserve_bytes)
        return MemoryReport(
            total=total, free=free, reserved=reserved, allocated=allocated, budget=budget,
            estimated=False,
            note="free_from_driver + (memory_reserved - memory_allocated) - reserve")

    def copy_stream(self):
        if not self.has_async_streams:
            return _SyncStream()
        try:
            return _CudaStream(self.device)
        except Exception as exc:
            _announce(f"streamfail-{self.backend}",
                      f"could not create a copy stream ({exc}); falling back to the synchronous "
                      f"transfer path.", logging.INFO)
            return _SyncStream()

    def event(self):
        if not self.has_async_streams:
            return _SyncEvent()
        try:
            return _CudaEvent()
        except Exception:
            return _SyncEvent()

    def empty_cache(self):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


class RocmCaps(CudaCaps):
    """ROCm speaks the CUDA API in torch, so the mechanics are inherited.

    What differs is what it is honest about: there is no compute capability, fused 4-bit kernels
    written against CUDA rarely load, and the tier is fixed at 2 by the project's own support
    statement rather than by anything measured here.
    """

    backend = "rocm"

    @property
    def tier(self):
        return 2


class XpuCaps(DeviceCaps):
    backend = "xpu"
    tier = 3

    @property
    def can_pin_memory(self):
        return supports_pinned_memory(self.device)

    def memory(self, reserve_bytes=0):
        total, free = device_memory(self.device)
        reserved = allocated = None
        try:
            reserved = int(torch.xpu.memory_reserved(self.device))
            allocated = int(torch.xpu.memory_allocated(self.device))
        except Exception:
            pass
        if free is None:
            return super().memory(reserve_bytes)
        held = max(0, (reserved or 0) - (allocated or 0))
        return MemoryReport(
            total=total, free=free, reserved=reserved, allocated=allocated,
            budget=max(0, free + held - reserve_bytes), estimated=False,
            note="xpu free + (memory_reserved - memory_allocated) - reserve")

    def empty_cache(self):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
        gc.collect()


class MpsCaps(DeviceCaps):
    backend = "mps"
    tier = 3

    def memory(self, reserve_bytes=0):
        """Unified memory, and no allocator introspection, so this is deliberately conservative.

        Apple's device and host share one pool: `recommended_max_memory` is a ceiling Metal
        suggests, not a driver free count, and there is no reserved/allocated split to add back.
        Over-reporting here would push the cache into swapping the whole machine, so what cannot
        be known is under-claimed and the report says it is an estimate.
        """
        total, free = device_memory(self.device)
        if total is None:
            return super().memory(reserve_bytes)
        return MemoryReport(
            total=total, free=free, reserved=None, allocated=None,
            budget=max(0, (free or 0) - reserve_bytes), estimated=True,
            note=("unified memory: recommended_max_memory - driver_allocated_memory, with no "
                  "allocator introspection available, so the budget is intentionally conservative"))

    def empty_cache(self):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()


class CpuCaps(DeviceCaps):
    backend = "cpu"
    tier = 4

    def memory(self, reserve_bytes=0):
        total, available = host_memory()
        return MemoryReport(
            total=total, free=available, reserved=None, allocated=None,
            budget=max(0, (available or 0) - reserve_bytes), estimated=True,
            note="host RAM; there is no separate device pool on this backend")


_BACKEND_CLASSES = {
    "cuda": CudaCaps,
    "rocm": RocmCaps,
    "xpu": XpuCaps,
    "mps": MpsCaps,
    "cpu": CpuCaps,
}

_caps_cache = {}


def get_caps(device=None, announce=True):
    """The capability object for a device, built once and reused.

    Resolving a device is also when the user finds out what they are getting: every fallback this
    hardware implies is announced here, once, before any layer runs.
    """
    dev = resolve_device(device) if not isinstance(device, torch.device) else device
    key = (dev.type, dev.index)
    caps = _caps_cache.get(key)
    if caps is None:
        caps = _BACKEND_CLASSES.get(backend_of(dev), CpuCaps)(dev)
        _caps_cache[key] = caps
        if announce:
            caps.announce_degradations()
    return caps


def reset_caps_cache():
    """Drop cached capability objects. For tests that swap the backend underneath us."""
    _caps_cache.clear()
