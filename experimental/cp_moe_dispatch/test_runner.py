"""CPU-only regression tests for repository packaging and result isolation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run as launcher


class LauncherTests(unittest.TestCase):
    def test_extension_commit_with_same_python_tree_is_accepted(self):
        with patch.object(launcher.subprocess, "check_output",
                          side_effect=[launcher.PINNED_PYTHON_TREE + "\n", ""]) as query:
            launcher.verify_source(Path("/example/repo"))
        self.assertEqual(query.call_args_list[0].args[0][-1], "HEAD:python")

    def test_changed_runtime_tree_is_rejected(self):
        with patch.object(launcher.subprocess, "check_output", return_value="changed\n"):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                launcher.verify_source(Path("/example/repo"))

    def test_dirty_runtime_is_rejected(self):
        with patch.object(launcher.subprocess, "check_output",
                          side_effect=[launcher.PINNED_PYTHON_TREE + "\n", " M python/file.py"]):
            with self.assertRaisesRegex(RuntimeError, "uncommitted"):
                launcher.verify_source(Path("/example/repo"))

    def test_results_outside_repo_are_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            results = Path(temporary) / "results"
            with patch.object(launcher, "REPO", repo):
                self.assertEqual(launcher.prepare_results_dir(results), results.resolve())
            self.assertTrue(results.is_dir())

    def test_repo_root_and_descendants_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            with patch.object(launcher, "REPO", repo):
                for target in (repo, repo / "results"):
                    with self.subTest(target=target):
                        with self.assertRaisesRegex(ValueError, "outside"):
                            launcher.prepare_results_dir(target)


if __name__ == "__main__":
    unittest.main()
