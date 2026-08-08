"""Capability queries and the device abstraction.

Every function here answers one question about the machine actually running, and answers it by
asking the backend or by attempting the operation -- never by recognising a device name. A device
called "RTX 5090" tells you nothing a query cannot tell you better, and it tells you nothing at all
about the card released next month.

Each query returns ``True``/``False`` when it knows, ``None`` when the backend cannot say. ``None``
means unavailable, and callers must treat it as "no, but do not claim it is absent" rather than
folding it into ``False``.
"""
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

import torch

# Backends this build of torch could possibly talk to. Order is fastest-first for auto-selection.
_BACKEND_ORDER = ("cuda", "rocm", "xpu", "mps", "cpu")


def _spec_exists(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


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
    if not _spec_exists("triton"):
        return False
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def fused_4bit_kernels(device):
    """Which fused 4-bit paths this machine could actually take.

    A fused path needs both a kernel and a device that can run it, so the presence of a package is
    reported separately from whether anything usable came out of it. Where nothing does, the engine
    dequantizes into scratch and computes in the compute dtype -- correct, just slower.
    """
    found = {
        # Shipped with torch itself on some builds; by far the most portable fused int4 path.
        "torch_int4pack": hasattr(torch.ops.aten, "_weight_int4pack_mm"),
        "bitsandbytes": _spec_exists("bitsandbytes"),
        "awq": _spec_exists("awq_ext") or _spec_exists("awq"),
        "gptq": _spec_exists("gptqmodel") or _spec_exists("auto_gptq") or _spec_exists("exllamav2"),
        "marlin": _spec_exists("marlin_kernels"),
    }
    # Fused kernels for these formats are written for the accelerator; on CPU they are never taken.
    usable = device.type in ("cuda", "xpu") and any(found.values())
    found["any_usable"] = bool(usable)
    return found


def python_and_torch_versions():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }
