"""Per-backend tests for the device abstraction, with the backend queries mocked.

Nobody testing this owns all five backends, so what is checked here is the property that has to
hold on hardware the author cannot reach: every gate returns a decision on every backend, every
fallback is reachable and produces something usable, and nothing raises because a feature is
absent. A missing accelerator feature is a slower path, never an error.

The one real execution is on CPU, which every machine has: a tiny forward pass driven entirely
through the abstraction, proving the fallback surfaces are not just type-correct but work.
"""
import logging
import unittest

import torch

from rocketllm.hw import caps as C
from rocketllm.hw.caps import (CpuCaps, CudaCaps, DeviceCaps, FusedPlan, MemoryReport, MpsCaps,
                               RocmCaps, XpuCaps, _SyncEvent, _SyncStream)

GB = 1024 ** 3

#: One representative of each backend, built directly so no hardware is needed.
BACKENDS = {
    "cuda": (CudaCaps, "cuda:0"),
    "rocm": (RocmCaps, "cuda:0"),
    "xpu": (XpuCaps, "xpu:0"),
    "mps": (MpsCaps, "mps"),
    "cpu": (CpuCaps, "cpu"),
}


def build(backend):
    cls, spec = BACKENDS[backend]
    return cls(torch.device(spec))


class CapsTestCase(unittest.TestCase):
    """Isolates each test from real hardware and from announcements made by earlier tests."""

    def setUp(self):
        C.reset_announcements()
        C.reset_caps_cache()
        self._saved = {name: getattr(C, name) for name in
                       ("supports_bf16", "supports_fp16", "supports_fp8", "supports_fp4",
                        "supports_pinned_memory", "supports_async_copy_streams",
                        "device_memory", "host_memory", "fused_4bit_kernels",
                        "compute_capability", "device_name")}
        # Default the machine to "capable of nothing" so each test opts in to what it needs and
        # no result can leak in from the box the suite happens to run on.
        self.set_queries()

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(C, name, value)
        C.reset_announcements()
        C.reset_caps_cache()

    def set_queries(self, bf16=False, fp16=True, fp8=False, fp4=False, pinned=False,
                    streams=False, device_mem=(None, None), host_mem=(16 * GB, 8 * GB),
                    kernels=None, cc=None):
        kernels = kernels if kernels is not None else {"any_usable": False}
        C.supports_bf16 = lambda device: bf16
        C.supports_fp16 = lambda device: fp16
        C.supports_fp8 = lambda device: fp8
        C.supports_fp4 = lambda device: fp4
        C.supports_pinned_memory = lambda device: pinned
        C.supports_async_copy_streams = lambda device: streams
        C.device_memory = lambda device: device_mem
        C.host_memory = lambda: host_mem
        C.fused_4bit_kernels = lambda device: dict(kernels)
        C.compute_capability = lambda device: cc
        C.device_name = lambda device: "Mock Device"


class TestEveryGateDecidesOnEveryBackend(CapsTestCase):
    def test_dtype_gates_return_a_boolean_everywhere(self):
        for backend in BACKENDS:
            caps = build(backend)
            for gate in ("supports_bf16", "supports_fp16", "supports_fp8", "supports_fp4",
                         "can_pin_memory", "has_async_streams"):
                with self.subTest(backend=backend, gate=gate):
                    self.assertIsInstance(getattr(caps, gate), bool,
                                          f"{backend}.{gate} must decide, not return None")

    def test_compute_dtype_is_always_a_real_dtype(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.assertIsInstance(build(backend).compute_dtype, torch.dtype)

    def test_memory_report_is_complete_everywhere(self):
        for backend in BACKENDS:
            caps = build(backend)
            report = caps.memory(reserve_bytes=0)
            with self.subTest(backend=backend):
                self.assertIsInstance(report, MemoryReport)
                self.assertIsInstance(report.budget, int)
                self.assertGreaterEqual(report.budget, 0)
                self.assertTrue(report.note.strip(), "a memory report must explain itself")

    def test_fused_plan_is_decided_everywhere(self):
        for backend in BACKENDS:
            plan = build(backend).fused_4bit_plan()
            with self.subTest(backend=backend):
                self.assertIsInstance(plan, FusedPlan)
                self.assertIn(plan.path, ("fused_packed", "dequant_to_scratch"))
                self.assertTrue(plan.reason.strip())

    def test_streams_and_events_exist_everywhere(self):
        for backend in BACKENDS:
            caps = build(backend)
            with self.subTest(backend=backend):
                with caps.copy_stream() as stream:
                    self.assertIsInstance(stream.is_async, bool)
                    event = stream.record_event()
                    event.synchronize()
                    stream.wait_event(event)
                stream.synchronize()
                self.assertTrue(caps.event().query() in (True, False))

    def test_empty_cache_never_raises(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                build(backend).empty_cache()

    def test_every_backend_reports_a_support_tier(self):
        for backend in BACKENDS:
            tier = build(backend).tier
            with self.subTest(backend=backend):
                self.assertIn(tier, (1, 2, 3, 4))

    def test_summary_is_complete_everywhere(self):
        for backend in BACKENDS:
            summary = build(backend).summary()
            with self.subTest(backend=backend):
                for key in ("backend", "tier", "compute_dtype", "quant_path", "pinned_memory",
                            "async_streams", "memory_note"):
                    self.assertIn(key, summary)
                    self.assertIsNotNone(summary[key])


class TestBf16Fallback(CapsTestCase):
    def test_bf16_is_used_where_supported(self):
        self.set_queries(bf16=True)
        self.assertIs(build("cuda").select_compute_dtype(torch.bfloat16), torch.bfloat16)

    def test_absent_bf16_falls_back_to_fp16(self):
        self.set_queries(bf16=False, fp16=True)
        self.assertIs(build("cuda").select_compute_dtype(torch.bfloat16), torch.float16)

    def test_the_fp16_fallback_names_the_overflow_risk(self):
        self.set_queries(bf16=False, fp16=True)
        with self.assertLogs("rocketllm.hw.caps", level=logging.WARNING) as captured:
            build("cuda").select_compute_dtype(torch.bfloat16)
        message = "\n".join(captured.output)
        self.assertIn("overflow", message)
        self.assertIn("silently", message)

    def test_the_fp16_warning_is_emitted_once_not_per_call(self):
        """This fires from a forward hook path; per-layer logging would bury the console."""
        self.set_queries(bf16=False, fp16=True)
        caps = build("cuda")
        with self.assertLogs("rocketllm.hw.caps", level=logging.WARNING) as captured:
            for _ in range(50):
                caps.select_compute_dtype(torch.bfloat16)
            logging.getLogger("rocketllm.hw.caps").warning("sentinel")
        overflow_lines = [line for line in captured.output if "overflow" in line]
        self.assertEqual(len(overflow_lines), 1)

    def test_neither_dtype_falls_back_to_fp32(self):
        self.set_queries(bf16=False, fp16=False)
        self.assertIs(build("cuda").select_compute_dtype(None), torch.float32)

    def test_requesting_fp16_without_fp16_prefers_bf16_then_fp32(self):
        self.set_queries(bf16=True, fp16=False)
        self.assertIs(build("cuda").select_compute_dtype(torch.float16), torch.bfloat16)
        self.set_queries(bf16=False, fp16=False)
        C.reset_announcements()
        self.assertIs(build("cuda").select_compute_dtype(torch.float16), torch.float32)

    def test_a_dtype_name_as_a_string_is_accepted(self):
        self.set_queries(bf16=True)
        self.assertIs(build("cuda").select_compute_dtype("bfloat16"), torch.bfloat16)

    def test_an_explicit_supported_request_is_honoured_unchanged(self):
        """Preserves the base model's behaviour: what the checkpoint asks for, it gets."""
        self.set_queries(bf16=True, fp16=True)
        caps = build("cuda")
        self.assertIs(caps.select_compute_dtype(torch.float16), torch.float16)
        self.assertIs(caps.select_compute_dtype(torch.float32), torch.float32)


class TestPinnedMemoryFallback(CapsTestCase):
    def test_backends_without_pinning_return_the_buffer_unchanged(self):
        for backend in ("mps", "cpu"):
            caps = build(backend)
            tensor = torch.zeros(8)
            with self.subTest(backend=backend):
                self.assertIs(caps.try_pin(tensor), tensor)
                self.assertFalse(caps.try_pin(tensor).is_pinned())

    def test_pinned_empty_always_returns_a_usable_buffer(self):
        for backend in BACKENDS:
            caps = build(backend)
            buffer = caps.pinned_empty((4, 4), torch.float32)
            with self.subTest(backend=backend):
                self.assertEqual(tuple(buffer.shape), (4, 4))
                self.assertEqual(buffer.dtype, torch.float32)

    def test_running_out_of_pinned_memory_degrades_instead_of_raising(self):
        """Pinned memory is a finite OS resource; exhaustion under load must not abort a run."""
        self.set_queries(pinned=True)
        caps = build("cuda")
        tensor = torch.zeros(8)

        def explode():
            raise RuntimeError("cannot allocate pinned memory")

        tensor.pin_memory = explode
        with self.assertLogs("rocketllm.hw.caps", level=logging.INFO) as captured:
            result = caps.try_pin(tensor)
        self.assertIs(result, tensor)
        self.assertIn("pageable", "\n".join(captured.output))


class TestStreamFallback(CapsTestCase):
    def test_backends_without_streams_get_the_synchronous_stand_in(self):
        for backend in ("mps", "cpu", "xpu"):
            with self.subTest(backend=backend):
                stream = build(backend).copy_stream()
                self.assertIsInstance(stream, _SyncStream)
                self.assertFalse(stream.is_async)

    def test_cuda_without_stream_support_falls_back(self):
        self.set_queries(streams=False)
        self.assertIsInstance(build("cuda").copy_stream(), _SyncStream)
        self.assertIsInstance(build("cuda").event(), _SyncEvent)

    def test_a_failure_creating_a_stream_falls_back_and_says_so(self):
        self.set_queries(streams=True)
        caps = build("cuda")
        original = C._CudaStream

        def explode(device):
            raise RuntimeError("no streams today")

        C._CudaStream = explode
        try:
            with self.assertLogs("rocketllm.hw.caps", level=logging.INFO) as captured:
                stream = caps.copy_stream()
        finally:
            C._CudaStream = original
        self.assertIsInstance(stream, _SyncStream)
        self.assertIn("synchronous", "\n".join(captured.output))

    def test_the_synchronous_stand_in_supports_the_whole_stream_protocol(self):
        """The streaming path is written once; the stand-in must not need special-casing."""
        stream = _SyncStream()
        with stream as entered:
            event = entered.record_event()
            entered.wait_event(event)
        stream.synchronize()
        event.record(stream)
        event.wait(stream)
        event.synchronize()
        self.assertTrue(event.query())


class TestFusedKernelDecision(CapsTestCase):
    def test_fused_needs_both_a_kernel_and_a_capable_backend(self):
        self.set_queries(kernels={"torch_int4pack": True, "any_usable": True})
        plan = build("cuda").fused_4bit_plan()
        self.assertEqual(plan.path, "fused_packed")
        self.assertEqual(plan.kernel, "torch_int4pack")

    def test_a_kernel_the_backend_cannot_run_selects_dequant(self):
        """Present but unusable is the trap: routing on presence alone fails mid-forward."""
        self.set_queries(kernels={"torch_int4pack": True, "any_usable": False})
        plan = build("mps").fused_4bit_plan()
        self.assertEqual(plan.path, "dequant_to_scratch")
        self.assertIsNone(plan.kernel)
        self.assertIn("cannot run fused", plan.reason)

    def test_no_kernel_at_all_selects_dequant(self):
        self.set_queries(kernels={"any_usable": False})
        plan = build("cuda").fused_4bit_plan()
        self.assertEqual(plan.path, "dequant_to_scratch")
        self.assertIn("no fused 4-bit kernel", plan.reason)

    def test_cpu_never_takes_the_fused_path(self):
        self.set_queries(kernels={"torch_int4pack": True, "any_usable": False})
        self.assertEqual(build("cpu").fused_4bit_plan().path, "dequant_to_scratch")

    def test_the_plan_is_computed_once_and_reused(self):
        self.set_queries(kernels={"torch_int4pack": True, "any_usable": True})
        caps = build("cuda")
        self.assertIs(caps.fused_4bit_plan(), caps.fused_4bit_plan())

    def test_importability_is_what_counts_not_mere_presence(self):
        """A package that is installed but broken must not enable the fused path."""
        C._import_cache.clear()
        original_spec, original_import = C._spec_exists, C.importlib.import_module
        C._spec_exists = lambda name: name == "bitsandbytes"

        def explode(name, *a, **kw):
            if name == "bitsandbytes":
                raise ImportError("compiled against a different CUDA")
            return original_import(name, *a, **kw)

        C.importlib.import_module = explode
        try:
            self.assertFalse(C._importable("bitsandbytes"))
        finally:
            C._spec_exists, C.importlib.import_module = original_spec, original_import
            C._import_cache.clear()


class TestMemoryAccounting(CapsTestCase):
    def test_cuda_adds_back_what_the_allocator_is_sitting_on(self):
        """mem_get_info alone under-reports: freed-but-held blocks still look used to the driver."""
        self.set_queries(device_mem=(10 * GB, 4 * GB))
        caps = build("cuda")
        saved = (torch.cuda.memory_reserved, torch.cuda.memory_allocated)
        torch.cuda.memory_reserved = lambda device=None: 3 * GB
        torch.cuda.memory_allocated = lambda device=None: 1 * GB
        try:
            report = caps.memory(reserve_bytes=1 * GB)
        finally:
            torch.cuda.memory_reserved, torch.cuda.memory_allocated = saved
        # 4 free + (3 reserved - 1 allocated) - 1 reserve = 5
        self.assertEqual(report.budget, 5 * GB)
        self.assertFalse(report.estimated)

    def test_reserve_can_exceed_free_without_going_negative(self):
        self.set_queries(device_mem=(10 * GB, 1 * GB))
        caps = build("cuda")
        saved = (torch.cuda.memory_reserved, torch.cuda.memory_allocated)
        torch.cuda.memory_reserved = lambda device=None: 0
        torch.cuda.memory_allocated = lambda device=None: 0
        try:
            report = caps.memory(reserve_bytes=99 * GB)
        finally:
            torch.cuda.memory_reserved, torch.cuda.memory_allocated = saved
        self.assertEqual(report.budget, 0)

    def test_a_backend_that_reports_nothing_degrades_and_admits_it(self):
        self.set_queries(device_mem=(None, None))
        report = build("mps").memory()
        self.assertTrue(report.estimated)
        self.assertGreaterEqual(report.budget, 0)

    def test_mps_is_conservative_and_says_why(self):
        self.set_queries(device_mem=(16 * GB, 12 * GB))
        report = build("mps").memory()
        self.assertTrue(report.estimated)
        self.assertIn("unified memory", report.note)
        self.assertIsNone(report.reserved)

    def test_cpu_reports_host_ram_as_its_pool(self):
        self.set_queries(host_mem=(32 * GB, 20 * GB))
        report = build("cpu").memory(reserve_bytes=4 * GB)
        self.assertEqual(report.budget, 16 * GB)
        self.assertIn("host RAM", report.note)


class TestTiers(CapsTestCase):
    def test_cuda_tier_comes_from_the_capability_not_the_name(self):
        self.set_queries(cc=(8, 6))
        self.assertEqual(build("cuda").tier, 1)
        self.set_queries(cc=(7, 5))
        self.assertEqual(build("cuda").tier, 2)

    def test_rocm_is_tier_two_and_mps_xpu_are_tier_three(self):
        self.assertEqual(build("rocm").tier, 2)
        self.assertEqual(build("mps").tier, 3)
        self.assertEqual(build("xpu").tier, 3)

    def test_cpu_is_tier_four(self):
        self.assertEqual(build("cpu").tier, 4)


class TestAnnouncements(CapsTestCase):
    def test_every_degradation_is_announced_at_load(self):
        self.set_queries(bf16=False, fp16=True, pinned=False, streams=False)
        with self.assertLogs("rocketllm.hw.caps", level=logging.INFO) as captured:
            build("cpu").announce_degradations()
        message = "\n".join(captured.output)
        for expected in ("pinned host memory", "async copy streams", "scratch buffer", "overflow"):
            self.assertIn(expected, message)

    def test_announcing_twice_says_nothing_the_second_time(self):
        self.set_queries(bf16=False, pinned=False, streams=False)
        caps = build("cpu")
        with self.assertLogs("rocketllm.hw.caps", level=logging.INFO) as first:
            caps.announce_degradations()
        with self.assertLogs("rocketllm.hw.caps", level=logging.INFO) as second:
            caps.announce_degradations()
            logging.getLogger("rocketllm.hw.caps").info("sentinel")
        self.assertGreater(len(first.output), 1)
        self.assertEqual(len(second.output), 1)
        self.assertIn("sentinel", second.output[0])

    def test_a_fully_capable_device_announces_no_degradations(self):
        self.set_queries(bf16=True, fp16=True, pinned=True, streams=True,
                         kernels={"torch_int4pack": True, "any_usable": True})
        logger = logging.getLogger("rocketllm.hw.caps")
        with self.assertLogs(logger, level=logging.INFO) as captured:
            build("cuda").announce_degradations()
            logger.info("sentinel")
        self.assertEqual(len(captured.output), 1, f"unexpected: {captured.output}")


class TestNothingRaisesWhenFeaturesAreMissing(CapsTestCase):
    def test_the_whole_surface_works_on_a_device_that_supports_nothing(self):
        """The degradation contract: absent features make it slower, never broken."""
        self.set_queries(bf16=False, fp16=False, pinned=False, streams=False,
                         device_mem=(None, None), kernels={"any_usable": False})
        for backend in BACKENDS:
            caps = build(backend)
            with self.subTest(backend=backend):
                caps.announce_degradations()
                self.assertIsInstance(caps.compute_dtype, torch.dtype)
                self.assertGreaterEqual(caps.memory().budget, 0)
                self.assertEqual(caps.fused_4bit_plan().path, "dequant_to_scratch")
                buffer = caps.pinned_empty((2, 2), torch.float32)
                with caps.copy_stream() as stream:
                    stream.record_event().synchronize()
                caps.empty_cache()
                self.assertEqual(buffer.numel(), 4)


class TestCpuForwardPass(unittest.TestCase):
    """The CPU backend must actually run, not merely type-check.

    This drives a tiny model the way the streaming path does -- weights parked on meta, staged
    through a host buffer, moved under a copy stream, then a forward -- entirely through the
    abstraction's fallback surfaces.
    """

    def test_a_tiny_model_streams_and_produces_finite_output(self):
        from accelerate.utils.modeling import set_module_tensor_to_device

        C.reset_announcements()
        caps = CpuCaps(torch.device("cpu"))
        dtype = caps.select_compute_dtype(None)

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 8)).to(dtype).eval()
        weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        inputs = torch.randn(2, 16, dtype=dtype)
        expected = model(inputs)

        # Park every parameter on meta, as the engine does between layers.
        for name in list(weights):
            set_module_tensor_to_device(model, name, "meta")
        self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))

        # Stream them back in through the abstraction.
        for name, tensor in weights.items():
            staged = caps.pinned_empty(tuple(tensor.shape), dtype)
            staged.copy_(tensor)
            with caps.copy_stream() as stream:
                moved = staged.to(caps.device, non_blocking=stream.is_async)
                stream.record_event().synchronize()
            set_module_tensor_to_device(model, name, "cpu", value=moved)

        with torch.no_grad():
            output = model(inputs)
        caps.empty_cache()

        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(torch.isfinite(output).all(), "streamed forward produced non-finite values")
        self.assertTrue(torch.equal(output, expected),
                        "streaming the weights back changed the result")


class TestFactory(CapsTestCase):
    def test_get_caps_returns_the_class_for_the_backend(self):
        self.assertIsInstance(C.get_caps(torch.device("cpu"), announce=False), CpuCaps)

    def test_get_caps_caches_per_device(self):
        first = C.get_caps(torch.device("cpu"), announce=False)
        self.assertIs(first, C.get_caps(torch.device("cpu"), announce=False))

    def test_an_unknown_backend_degrades_to_cpu_rather_than_failing(self):
        original = C.backend_of
        C.backend_of = lambda device: "some-future-accelerator"
        try:
            caps = C.get_caps(torch.device("cpu"), announce=False)
        finally:
            C.backend_of = original
        self.assertIsInstance(caps, DeviceCaps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
