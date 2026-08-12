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

The list of packages comes from the doctor's own inventory, so a new optional dependency is covered
here the moment it is documented there, and cannot be added to one without the other.
"""
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
BLOCKED = sorted({package.module for package in OPTIONAL_PACKAGES} | {"quanto", "httpx"})

_BLOCKER = '''
import importlib.abc, importlib.machinery, sys

BLOCKED = {blocked!r}


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


for _name in [n for n in sys.modules if n.split(".")[0] in BLOCKED]:
    del sys.modules[_name]
sys.meta_path.insert(0, _Blocker())
sys.path.insert(0, {root!r})
'''


def run_without_optionals(body):
    """Run `body` in a fresh interpreter with every optional package unimportable."""
    source = _BLOCKER.format(blocked=BLOCKED, root=str(REPO_ROOT)) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=600)


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
