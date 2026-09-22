"""Appendix D runner contracts without solving beams or tracing RT channels."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_appendix_d as runner


class AppendixDRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "Nulling_CDF_SectorDrop.ipynb"
        self.original_bytes = b'{"cells": [{"outputs": ["preserved"]}], "metadata": {}}\n'
        self.original.write_bytes(self.original_bytes)
        (self.root / "run_appendix_d.py").write_bytes(Path(runner.__file__).read_bytes())
        for name in ("requirements-appendix-d.txt", "APPENDIX_D_EXPERIMENTS.md",
                     "Nulling_CDF_SectorDrop_AppendixD.ipynb", "SceneConfigSionnaSectorDrop.py"):
            (self.root / name).write_text("fixture source: " + name + "\n")
        package = self.root / "appendix_d_experiments"
        package.mkdir()
        (package / "rt_source.py").write_text("# archived fixture RT source\n")
        self.start_patch(patch.object(runner, "ROOT", self.root))
        self.start_patch(patch.object(runner, "__file__", str(self.root / "run_appendix_d.py")))
        self.start_patch(patch.object(runner.importlib.metadata, "version", return_value="test-version"))

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_suite(self, output, **kwargs):
        with redirect_stdout(io.StringIO()):
            return runner.run_suite(output, **kwargs)

    def manifest(self, output):
        return json.loads((output / "manifest.json").read_text())

    def test_fresh_generation_precedes_spatial_evaluation_and_records_actual_source(self):
        output = self.root / "fresh-run"
        generated_cache = output / "fresh_rt_cache" / "new_timestamped_source"
        generated = dict(cache_dir=str(generated_cache), num_macros=12,
                         max_depth=2, temporal_data_available=False)
        events = []

        def generate(destination, **kwargs):
            events.append("generate")
            self.assertEqual(destination, output / "fresh_rt_cache")
            generated_cache.mkdir(parents=True)
            return generated

        def evaluate(destination, source, **kwargs):
            events.append("evaluate")
            self.assertEqual(source, str(generated_cache))
            self.assertEqual(destination, output / "fresh_rt_spatial")
            self.assertNotIn(generated_cache, destination.parents)
            # Provenance and completed generation must already be persisted
            # when the evaluator starts, so a later failure stays traceable.
            running = self.manifest(output)
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["completed"], ["fresh_rt_generation"])
            self.assertEqual(running["spatial"]["source_cache"], str(generated_cache))

        with patch("appendix_d_experiments.rt_source.generate_fresh_rt_cache", side_effect=generate) as generator, \
                patch("appendix_d_experiments.cache_adapter.run_cached_spatial", side_effect=evaluate) as adapter:
            self.assertEqual(self.run_suite(
                output, experiments=(), fresh_rt=True, quick=False, seed=73,
                gamma_db=iter((-12, -3)), cache_max_sectors=3,
                rt_num_macros=12, rt_max_depth=2), output)
        generator.assert_called_once_with(output / "fresh_rt_cache", quick=False,
                                          seed=73, num_macros=12, max_depth=2)
        adapter.assert_called_once_with(output / "fresh_rt_spatial", str(generated_cache),
                                        gamma_db=(-12, -3), seed=73, max_sectors=3)
        self.assertEqual(events, ["generate", "evaluate"])
        manifest = self.manifest(output)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed"], ["fresh_rt_generation", "fresh_rt_spatial"])
        self.assertEqual(manifest["spatial"]["mode"], "fresh_rt")
        self.assertEqual(manifest["spatial"]["generation"], generated)
        self.assertEqual(manifest["gamma_db"], [-12, -3])
        self.assertTrue(manifest["original_notebook_unchanged"])
        self.assertEqual(self.original.read_bytes(), self.original_bytes)
        self.assertEqual(manifest["original_notebook_sha256"],
                         hashlib.sha256(self.original_bytes).hexdigest())
        self.assertIn("SceneConfigSionnaSectorDrop.py", manifest["source_sha256"])
        self.assertIn("appendix_d_experiments/rt_source.py", manifest["source_sha256"])
        for name, digest in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((output / "source" / name).read_bytes()).hexdigest(), digest)

    def test_preserved_cache_uses_cached_route_without_generating_channels(self):
        output = self.root / "cached-run"
        source = self.root / "preserved-cache"
        source.mkdir()
        sentinel = source / "recorded-channel.bin"
        sentinel.write_bytes(b"preserve these recorded channels")
        with patch("appendix_d_experiments.rt_source.generate_fresh_rt_cache") as generator, \
                patch("appendix_d_experiments.cache_adapter.run_cached_spatial") as adapter:
            self.run_suite(output, experiments=(), cache_dir=source,
                           gamma_db=(-9, -2), cache_max_sectors=None)
        generator.assert_not_called()
        adapter.assert_called_once_with(output / "cached_spatial", source,
                                        gamma_db=(-9, -2), seed=20260921, max_sectors=None)
        manifest = self.manifest(output)
        self.assertEqual(manifest["spatial"]["mode"], "cache")
        self.assertEqual(manifest["spatial"]["source_cache"], str(source))
        self.assertNotIn("generation", manifest["spatial"])
        self.assertEqual(manifest["completed"], ["cached_spatial"])
        self.assertEqual(sentinel.read_bytes(), b"preserve these recorded channels")
        self.assertEqual(self.original.read_bytes(), self.original_bytes)

    def test_e4_target_is_forwarded_for_both_profiles(self):
        for quick in (True, False):
            with self.subTest(quick=quick):
                output = self.root / ("quick" if quick else "full")
                with patch("appendix_d_experiments.e45.run_e4") as e4, \
                        patch("appendix_d_experiments.rt_source.generate_fresh_rt_cache") as generator, \
                        patch("appendix_d_experiments.cache_adapter.run_cached_spatial") as adapter:
                    self.run_suite(output, experiments=("E4",), quick=quick,
                                   seed=100, e4_alpha=0.05)
                e4.assert_called_once_with(output / "E4", seed=104, quick=quick, alpha=0.05)
                generator.assert_not_called()
                adapter.assert_not_called()
                manifest = self.manifest(output)
                self.assertEqual(manifest["e4_calibration_alpha"], 0.05)
                self.assertEqual(manifest["profile"], "quick" if quick else "full")
                self.assertEqual(manifest["spatial"]["mode"], "none")
                self.assertEqual(manifest["completed"], ["E4"])

    def test_conflicting_empty_and_invalid_requests_fail_before_creating_output(self):
        cases = [dict(experiments=(), fresh_rt=True, cache_dir="old-cache"),
                 dict(experiments=()), dict(experiments=("unknown",)),
                 dict(rt_num_macros=6), dict(rt_max_depth=1),
                 dict(e4_alpha=0), dict(e4_alpha=1), dict(e4_alpha=float("nan"))]
        for index, options in enumerate(cases):
            with self.subTest(options=options):
                output = self.root / f"invalid-{index}"
                with patch("appendix_d_experiments.rt_source.generate_fresh_rt_cache") as generator, \
                        patch("appendix_d_experiments.e45.run_e4") as e4:
                    with self.assertRaises(ValueError):
                        self.run_suite(output, **options)
                generator.assert_not_called()
                e4.assert_not_called()
                self.assertFalse(output.exists())

    def test_failed_fresh_generation_is_recorded_without_cache_fallback(self):
        output = self.root / "failed-generation"
        with patch("appendix_d_experiments.rt_source.generate_fresh_rt_cache",
                   side_effect=RuntimeError("RT backend unavailable")), \
                patch("appendix_d_experiments.cache_adapter.run_cached_spatial") as adapter:
            with self.assertRaisesRegex(RuntimeError, "RT backend unavailable"):
                self.run_suite(output, experiments=(), fresh_rt=True)
        adapter.assert_not_called()
        manifest = self.manifest(output)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed"], [])
        self.assertEqual(manifest["spatial"]["mode"], "fresh_rt")
        self.assertIsNone(manifest["spatial"]["source_cache"])
        self.assertIn("RT backend unavailable", manifest["error"])
        self.assertIn("finished_utc", manifest)

    def test_failed_spatial_evaluation_retains_completed_generation_provenance(self):
        output = self.root / "failed-spatial"
        generated = dict(cache_dir=str(output / "fresh_rt_cache" / "source"), num_macros=6)
        with patch("appendix_d_experiments.rt_source.generate_fresh_rt_cache", return_value=generated), \
                patch("appendix_d_experiments.cache_adapter.run_cached_spatial",
                      side_effect=RuntimeError("spatial solve failed")):
            with self.assertRaisesRegex(RuntimeError, "spatial solve failed"):
                self.run_suite(output, experiments=(), fresh_rt=True)
        manifest = self.manifest(output)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed"], ["fresh_rt_generation"])
        self.assertEqual(manifest["spatial"]["generation"], generated)
        self.assertEqual(manifest["spatial"]["source_cache"], generated["cache_dir"])

    def test_changed_original_notebook_is_detected_using_an_isolated_fixture(self):
        output = self.root / "changed-original"
        def change_fixture(*args, **kwargs):
            self.original.write_bytes(b"changed fixture, never the real project notebook")
        with patch("appendix_d_experiments.e45.run_e4", side_effect=change_fixture):
            with self.assertRaisesRegex(RuntimeError, "Original notebook changed"):
                self.run_suite(output, experiments=("E4",))
        manifest = self.manifest(output)
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["original_notebook_unchanged"])

    def test_cli_spatial_only_forwards_fresh_source_and_explicit_options(self):
        output = self.root / "cli-run"
        args = ["run_appendix_d.py", "--spatial-only", "--fresh-rt", "--profile", "full",
                "--rt-macros", "12", "--rt-max-depth", "2", "--e4-alpha", "0.05",
                "--output-dir", str(output)]
        with patch.object(runner.sys, "argv", args), patch.object(runner, "run_suite") as run:
            runner.main()
        run.assert_called_once()
        self.assertEqual(run.call_args.args, (output,))
        options = run.call_args.kwargs
        self.assertEqual(options["experiments"], ())
        self.assertTrue(options["fresh_rt"])
        self.assertFalse(options["quick"])
        self.assertEqual(options["rt_num_macros"], 12)
        self.assertEqual(options["rt_max_depth"], 2)
        self.assertEqual(options["e4_alpha"], 0.05)
        self.assertIsNone(options["cache_dir"])


if __name__ == "__main__":
    unittest.main()
