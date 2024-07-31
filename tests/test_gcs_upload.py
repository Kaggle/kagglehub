"""Test helper functions in kagglehub.
"""

import pathlib
import tempfile

from kagglehub.gcs_upload import filtered_walk, normalize_patterns
from tests.fixtures import BaseTestCase


class TesModelsHelpers(BaseTestCase):
    def testnormalize_patterns(self) -> None:
        default_patterns = [".git/", ".cache/", ".gitignore"]
        self.assertEqual(
            normalize_patterns(default=default_patterns, additional=None),
            [".git/*", ".cache/*", ".gitignore"],
        )
        self.assertEqual(
            normalize_patterns(default=default_patterns, additional=["original/", "*/*.txt", "doc/readme.txt"]),
            [".git/*", ".cache/*", ".gitignore", "original/*", "*/*.txt", "doc/readme.txt"],
        )

    def test_filtered_walk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_p = pathlib.Path(tmp_dir)

            # files to upload
            (tmp_dir_p / "a" / "b").mkdir(parents=True)
            (tmp_dir_p / "weights.txt").touch()
            (tmp_dir_p / "a" / "a.txt").touch()
            (tmp_dir_p / "a" / "b" / "b.txt").touch()
            (tmp_dir_p / "a" / "b" / ".bb").touch()
            expected_files = {
                tmp_dir_p / "weights.txt",
                tmp_dir_p / "a" / "a.txt",
                tmp_dir_p / "a" / "b" / "b.txt",
                tmp_dir_p / "a" / "b" / ".bb",
            }

            # files to ignore
            (tmp_dir_p / ".git").mkdir(parents=True)
            (tmp_dir_p / ".git" / "file").write_text("hidden git file")
            (tmp_dir_p / ".gitignore").write_text("none")

            (tmp_dir_p / "a" / ".git").mkdir(parents=True)
            (tmp_dir_p / "a" / "b" / ".git").mkdir(parents=True)
            (tmp_dir_p / "a" / "b" / ".git" / "abgit.txt").write_text("abgit")

            (tmp_dir_p / "a" / "b" / ".hidden").touch()

            (tmp_dir_p / "original" / "fp8").mkdir(parents=True)
            (tmp_dir_p / "original" / "fp8" / "weights").touch()
            (tmp_dir_p / "original" / "fp16").mkdir(parents=True)
            (tmp_dir_p / "original" / "fp16" / "weights").touch()

            # filtered walk
            ignore_patterns = normalize_patterns(
                default=[".git/", "*/.git/", ".gitignore", "*/.hidden", "original/"], additional=None
            )
            walked_files = []
            for dir_path, _, file_names in filtered_walk(base_dir=tmp_dir, ignore_patterns=ignore_patterns):
                for file_name in file_names:
                    walked_files.append(pathlib.Path(dir_path) / file_name)
            self.assertEqual(set(walked_files), expected_files)
