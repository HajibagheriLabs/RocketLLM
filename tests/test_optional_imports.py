"""Importing RocketLLM with every optional dependency absent.

A user installs the base package on a machine with no kernel extensions, no server extra and no
accelerator, types ``import rocketllm``, and gets a traceback ending in ``ModuleNotFoundError: No
module named 'bitsandbytes'``. That is the worst first impression this project can make, and it is
entirely avoidable: every optional package has a defined absence, and a missing one must produce a
sentence naming what is gone and what it cost, never a stack trace.

So this file runs the imports with those packages genuinely unimportable and checks what comes out.
Each case runs in its own subprocess, because import state is global: a package imported by an
earlier test is already in ``sys.modules`` and cannot be hidden afterwards.

Absence is simulated by a meta-path finder that hands back a spec whose loader raises. That is
deliberate rather than the more obvious "return None from find_spec": returning nothing does not
hide a package that is really installed, and *raising* from ``find_spec`` breaks third-party code
that legitimately calls ``importlib.util.find_spec`` to test for a package -- transformers probes
for half this list at import time, and would fail for the wrong reason. Failing at load leaves the
probe working and the import broken, which is exactly the shape of a package that is not there.

The distribution METADATA has to go with it, and that is not belt-and-braces. A probe that finds a
spec commonly confirms it by asking importlib.metadata for a version, and the name it asks under is
the distribution's, not the module's -- transformers decides whether images are available by looking
for a module called PIL and a distribution called Pillow. Leave the metadata behind and the probe
says yes, the caller imports the module unguarded, and the run dies on an import error instead of
taking the fallback under test. So the blocker works out which distributions install a blocked
top-level module and makes those disappear too.

The list of packages comes from the doctor's own inventory, so a new optional dependency is covered
here the moment it is documented there, and cannot be added to one without the other.
"""
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketllm.hw.doctor import OPTIONAL_PACKAGES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Everything the doctor knows is optional, plus the module names those packages import under where
#: they differ from the pip name. Anything genuinely absent on the running machine is already
#: hidden; the blocker exists for the ones that happen to be installed here.
#:
#: httpx is deliberately NOT here, though it was: huggingface-hub 1.x imports it at module scope, so
#: it is present in every environment that can import transformers at all. Hiding it would not
#: simulate a bare install, it would simulate one that cannot exist -- and the failure would land on
#: `import transformers`, which is exactly the signal this file exists to keep clean.
BLOCKED = sorted({package.module for package in OPTIONAL_PACKAGES} | {"quanto"})

_BLOCKER = '''
import importlib.abc, importlib.machinery, importlib.metadata, importlib.util, sys

_REQUESTED = {blocked!r}


def _installed(name):
    """Whether this package is genuinely importable here, asked before anything is hidden."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# Only hide what is actually present. A package that was never installed is already absent, and
# hiding it a second time invents a state that cannot occur: find_spec would answer yes for
# something with no distribution metadata and no loadable module, and a prober that checks both --
# transformers checks both, for half this list -- ends in an ImportError instead of the graceful
# "not available" it is written to produce. That failure lands on `import transformers`, which is
# the one thing this file must be able to trust.
BLOCKED = [name for name in _REQUESTED if _installed(name)]


class _Missing(importlib.abc.Loader):
    """A loader that fails, standing in for a package that is not installed."""

    def create_module(self, spec):
        raise ModuleNotFoundError(f"No module named {{spec.name!r}}", name=spec.name)

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named {{module.__name__!r}}",
                                  name=module.__name__)


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            return importlib.machinery.ModuleSpec(fullname, _Missing())
        return None


# `importlib.util.find_spec` is the probe half of the same question, and it has to agree with the
# import half or callers end up in a state no real machine produces. transformers asks it first and
# imports unguarded when it says yes -- `if is_psutil_available(): import psutil` -- so a finder that
# answers yes while the import fails turns a graceful "not available" into an ImportError inside
# transformers.generation. Note this is patched on `importlib.util` rather than raising from the
# finder: raising there breaks every legitimate caller, which is why the finder hands back a spec at
# all.
_real_find_spec = importlib.util.find_spec


def _find_spec(name, package=None):
    if name.split(".")[0].lstrip(".") in BLOCKED:
        return None
    return _real_find_spec(name, package)


importlib.util.find_spec = _find_spec


def _normalise(name):
    return (name or "").strip().lower().replace("-", "_")


def _top_level_names(dist):
    """Every top-level module a distribution installs.

    Read from the files it recorded rather than assumed from its name: PIL comes from Pillow, and a
    probe that asks for one by the other is exactly what this has to cover. top_level.txt is
    consulted first where it exists, because a wheel that ships a namespace package records it there
    and nowhere else.
    """
    names = set()
    try:
        recorded = dist.read_text("top_level.txt") or ""
    except Exception:
        recorded = ""
    names.update(line.strip() for line in recorded.splitlines() if line.strip())
    for path in (dist.files or ()):
        parts = str(path).replace("\\\\", "/").split("/")
        head = parts[0]
        if head in ("", "..") or head.endswith((".dist-info", ".egg-info")):
            continue
        if len(parts) == 1:
            head = head[:-3] if head.endswith(".py") else head
        names.add(head)
    return names


_HIDDEN_DISTRIBUTIONS = set()
for _dist in importlib.metadata.distributions():
    if _top_level_names(_dist) & set(BLOCKED):
        try:
            _HIDDEN_DISTRIBUTIONS.add(_normalise(_dist.metadata["Name"]))
        except Exception:
            pass
_HIDDEN_DISTRIBUTIONS.update(_normalise(name) for name in BLOCKED)
_HIDDEN_DISTRIBUTIONS.discard("")

_real_version = importlib.metadata.version
_real_distribution = importlib.metadata.distribution
_real_metadata = importlib.metadata.metadata


def _hidden(name):
    return _normalise(name) in _HIDDEN_DISTRIBUTIONS


def _version(name):
    if _hidden(name):
        raise importlib.metadata.PackageNotFoundError(name)
    return _real_version(name)


def _distribution(name):
    if _hidden(name):
        raise importlib.metadata.PackageNotFoundError(name)
    return _real_distribution(name)


def _metadata(name):
    if _hidden(name):
        raise importlib.metadata.PackageNotFoundError(name)
    return _real_metadata(name)


importlib.metadata.version = _version
importlib.metadata.distribution = _distribution
importlib.metadata.metadata = _metadata

for _name in [n for n in sys.modules if n.split(".")[0] in BLOCKED]:
    del sys.modules[_name]
sys.meta_path.insert(0, _Blocker())
sys.path.insert(0, {root!r})
'''


def run_without(body, blocked):
    """Run `body` in a fresh interpreter with `blocked` unimportable and their metadata hidden.

    Exposed rather than kept private because a degradation is only worth checking with exactly the
    package it depends on removed: hiding the whole list would also remove pydantic, and the module
    under test would then fail to import for an unrelated reason.
    """
    source = _BLOCKER.format(blocked=sorted(blocked), root=str(REPO_ROOT)) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=600)


def run_without_optionals(body):
    """Run `body` in a fresh interpreter with every optional package unimportable."""
    return run_without(body, BLOCKED)


class OptionalImportCase(unittest.TestCase):
    def assert_clean(self, result, what):
        """The process succeeded, and nothing that came out of it looks like a crash."""
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0,
                         f"{what} exited {result.returncode} with every optional package "
                         f"absent:\n{combined}")
        self.assertNotIn("Traceback (most recent call last)", combined,
                         f"{what} printed a traceback rather than a message:\n{combined}")
        return combined


class TestThePackageImports(OptionalImportCase):
    def test_importing_rocketllm_needs_nothing_optional(self):
        result = run_without_optionals("""
            import rocketllm
            print("imported")
            print("AutoModel", hasattr(rocketllm, "AutoModel"))
            print("RocketModel", hasattr(rocketllm, "RocketModel"))
        """)
        out = self.assert_clean(result, "import rocketllm")
        self.assertIn("imported", out)
        self.assertIn("AutoModel True", out)
        self.assertIn("RocketModel True", out)

    def test_the_hardware_layer_works_with_no_kernels_and_no_accelerator(self):
        """The profile is what every tuning knob comes from; it cannot need an optional package."""
        result = run_without_optionals("""
            from rocketllm.hw import HardwareProfile
            from rocketllm.hw import caps
            profile = HardwareProfile.probe(device="cpu", storage_budget_seconds=0.2)
            print("triton", caps.has_triton())
            print("fused", profile.fused_4bit.get("any_usable"))
            print("compute_dtype", profile.derived["compute_dtype"].value)
            print("reserve", profile.derived["reserve_bytes"].value)
        """)
        out = self.assert_clean(result, "HardwareProfile.probe")
        self.assertIn("triton False", out)
        self.assertIn("fused False", out)

    def test_the_quant_registry_decides_every_format_without_a_reader_installed(self):
        """A format whose reader is missing still has to be describable, or nothing can explain it."""
        result = run_without_optionals("""
            from rocketllm.quant import decision_table
            rows = decision_table()
            print("formats", len(rows))
            for row in rows:
                assert row["path"], row
                assert row["reason"], row
            print("all decided")
        """)
        out = self.assert_clean(result, "the quant decision table")
        self.assertIn("all decided", out)

    def test_the_doctor_runs_and_reports_what_is_missing(self):
        """The command a bug reporter is asked to run is the one that must survive a bare install."""
        result = run_without_optionals("""
            import io
            from rocketllm.hw import doctor
            buffer = io.StringIO()
            collected = doctor.run(device="cpu", storage_budget_seconds=0.2, out=buffer)
            missing = [p["module"] for p in collected["packages"] if not p["present"]]
            print("missing", len(missing))
            print("rendered", len(buffer.getvalue()) > 0)
        """)
        out = self.assert_clean(result, "rocketllm doctor")
        self.assertIn("rendered True", out)
        # Every one of them is blocked, so every one has to be reported missing. A count short of
        # the full list means the inventory and the import path disagree about a package's name.
        self.assertIn(f"missing {len(OPTIONAL_PACKAGES)}", out)


class TestTheMessagesAreMessages(OptionalImportCase):
    """Where something genuinely cannot run, what comes out has to be readable prose."""

    def test_importing_the_server_names_the_extra_that_fixes_it(self):
        result = run_without_optionals("""
            try:
                import rocketllm.server
            except ImportError as exc:
                print("MESSAGE:", exc)
            else:
                raise SystemExit("the server imported without fastapi installed")
        """)
        out = self.assert_clean(result, "importing rocketllm.server")
        self.assertIn("MESSAGE:", out)
        self.assertIn("pip install 'rocketllm[server]'", out,
                      "the error has to say what to type, not just what is missing")

    def test_serve_exits_with_a_message_rather_than_a_traceback(self):
        """`rocketllm serve` without the extra is the most likely way anyone meets this."""
        result = run_without_optionals("""
            from rocketllm.cli import main
            code = main(["serve", "--model", "nonexistent"])
            print("EXIT", code)
        """)
        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, combined)
        self.assertNotIn("Traceback (most recent call last)", combined,
                         f"serve printed a traceback rather than a message:\n{combined}")
        self.assertIn("EXIT 1", combined, "serve without the extra must fail, not carry on")
        self.assertIn("pip install 'rocketllm[server]'", combined)

    def test_profile_and_doctor_still_run_from_the_command_line(self):
        result = run_without_optionals("""
            from rocketllm.cli import main
            print("PROFILE EXIT", main(["profile", "--device", "cpu"]))
            print("DOCTOR EXIT", main(["doctor", "--device", "cpu"]))
        """)
        out = self.assert_clean(result, "rocketllm profile / doctor")
        self.assertIn("PROFILE EXIT 0", out)
        self.assertIn("DOCTOR EXIT 0", out)

    def test_the_doctors_json_stays_machine_readable(self):
        """--json is the form an issue template asks for, so it has to parse on a bare install."""
        result = run_without_optionals("""
            import io, json
            from rocketllm.hw import doctor
            buffer = io.StringIO()
            doctor.run(device="cpu", as_json=True, storage_budget_seconds=0.2, out=buffer)
            payload = json.loads(buffer.getvalue())
            print("JSON KEYS", " ".join(sorted(payload)))
        """)
        out = self.assert_clean(result, "rocketllm doctor --json")
        keys = out.split("JSON KEYS", 1)[1].split("\\n", 1)[0].split()
        for expected in ("capabilities", "packages", "profile", "quant_formats", "storage"):
            self.assertIn(expected, keys)


class TestEachDegradationNamesItsPackage(OptionalImportCase):
    """Not just "it did not crash": the message has to say which package is missing.

    A fallback that happens silently is nearly as bad as a crash. Someone whose 4-bit checkpoint is
    being expanded into scratch on every layer, or whose KV cache quietly stopped being the backend
    they asked for, has no way to discover that from the output -- the run is simply slower or
    different, and there is nothing to search for. So each path below is taken with its package
    genuinely unimportable, and the text that comes out must name the package.
    """

    def test_a_delegated_kv_backend_falls_back_and_names_the_package(self):
        result = run_without_optionals("""
            import logging
            logging.basicConfig(level=logging.INFO)
            from rocketllm.quant.kv_cache import KVCacheConfig, build_kv_cache
            from rocketllm.quant.kv_cache import QuantizedKVCache
            for choice, package in (("hqq", "hqq"), ("quanto", "optimum-quanto")):
                cache = build_kv_cache(choice, KVCacheConfig(), device="cpu")
                assert isinstance(cache, QuantizedKVCache), (choice, type(cache))
                print("FELL BACK", choice, package)
        """)
        combined = self.assert_clean(result, "a delegated KV cache backend")
        for choice, package in (("hqq", "hqq"), ("quanto", "optimum-quanto")):
            self.assertIn(f"FELL BACK {choice}", combined)
            self.assertIn(package, combined,
                          f"the {choice} fallback did not name the package that would restore it")

    def test_a_packed_checkpoint_that_cannot_be_decoded_says_what_to_install(self):
        """The one place in the package that refuses outright. Nothing else can decode the payload,
        so there is no slower-but-correct path -- which makes the message the whole of the fix."""
        result = run_without_optionals("""
            import torch
            from rocketllm.quant.safetensors_quant import CompressedTensorsBackend

            # The decode path only engages for a checkpoint that actually declared itself
            # quantized, so it needs a quantizer and a model to be reached at all. Neither is
            # touched before the refusal, so stubs are enough to get there.
            backend = CompressedTensorsBackend(config={"quant_method": "compressed-tensors"},
                                               hf_quantizer=object(), model=object())
            try:
                backend.prepare_layer({"model.layers.0.mlp.up_proj.weight_packed":
                                       torch.zeros(4, 4, dtype=torch.uint8)})
            except ImportError as exc:
                print("REFUSED:", exc)
            else:
                raise SystemExit("a packed payload was accepted without its decoder installed")
        """)
        combined = result.stdout + result.stderr
        self.assertIn("REFUSED:", combined, combined)
        self.assertIn("compressed-tensors", combined)
        self.assertIn("pip install", combined,
                      "refusing without saying what to type is a dead end")

    def test_no_fused_kernel_says_so_once_and_names_the_path_taken(self):
        result = run_without_optionals("""
            import logging
            logging.basicConfig(level=logging.INFO)
            from rocketllm.hw import caps
            device = caps.get_caps("cpu", announce=False)
            plan = device.fused_4bit_plan()
            print("PATH", plan.path)
            print("REASON", plan.reason)
            print("TRITON", caps.has_triton())
        """)
        combined = self.assert_clean(result, "the fused 4-bit decision")
        self.assertIn("PATH dequant_to_scratch", combined)
        self.assertIn("scratch", combined.split("REASON", 1)[1],
                      "the reason has to say what happens instead, not only that something is off")
        self.assertIn("TRITON False", combined)

    def test_the_mlx_backend_is_not_required_to_import_the_package(self):
        """Apple Silicon is the one platform where this used to be an unguarded import, so
        `import rocketllm` on a Mac without the mlx extra ended in a traceback."""
        result = run_without_optionals("""
            import rocketllm.auto_model as auto
            auto.is_on_mac_os = True          # pretend to be Apple Silicon, without mlx installed
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    auto.AutoModel.from_pretrained("definitely-not-a-real-model")
                except Exception as exc:
                    # It must fail on the missing MODEL, never on the missing backend.
                    print("FAILED ON:", type(exc).__name__)
            print("WARNED:", " | ".join(str(w.message) for w in caught))
        """)
        combined = result.stdout + result.stderr
        self.assertNotIn("Traceback (most recent call last)", combined, combined)
        warned = combined.split("WARNED:", 1)[1]
        self.assertIn("mlx", warned,
                      "falling back off the MLX path has to name the extra that restores it")
        self.assertIn("rocketllm[mlx]", warned)

    def test_every_inventory_package_is_reported_missing_with_its_cost(self):
        """The doctor is where a user actually discovers all of this, so it is checked as a whole."""
        result = run_without_optionals("""
            import io, json
            from rocketllm.hw import doctor
            buffer = io.StringIO()
            collected = doctor.run(device="cpu", storage_budget_seconds=0.2, out=buffer)
            missing = [p for p in collected["packages"] if not p["present"]]
            print(json.dumps([[p["module"], p["without"]] for p in missing]))
        """)
        combined = self.assert_clean(result, "the doctor's package inventory")
        payload = json.loads(combined[combined.index("[["):combined.rindex("]]") + 2])
        reported = {module: without for module, without in payload}
        for package in OPTIONAL_PACKAGES:
            with self.subTest(module=package.module):
                self.assertIn(package.module, reported,
                              "a package the engine can use but did not report missing is one "
                              "nobody will think to install")
                self.assertTrue(reported[package.module].strip())


class TestTheInventoryIsHonest(unittest.TestCase):
    """The doctor's list is what the rest of this file trusts, so it gets checked directly."""

    def test_every_entry_says_what_is_lost_without_it(self):
        for package in OPTIONAL_PACKAGES:
            with self.subTest(module=package.module):
                self.assertTrue(package.unlocks.strip())
                self.assertTrue(package.without.strip(),
                                "an entry that does not say what its absence costs is a package "
                                "list, which pip already provides")

    def test_no_module_is_listed_twice(self):
        modules = [package.module for package in OPTIONAL_PACKAGES]
        self.assertEqual(len(modules), len(set(modules)))

    def test_nothing_declared_optional_is_a_base_dependency(self):
        """A base dependency in this list would promise a fallback that does not exist."""
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        base = pyproject.split("[project.optional-dependencies]")[0]
        for package in OPTIONAL_PACKAGES:
            with self.subTest(module=package.module):
                self.assertNotIn(f'"{package.module}"', base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
