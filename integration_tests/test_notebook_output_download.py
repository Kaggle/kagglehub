import unittest

from requests import HTTPError

from kagglehub import notebook_output_download

from .utils import assert_files, create_test_cache, unauthenticated

VERSIONED_NOTEBOOK_HANDLE = "alexisbcook/titanic-tutorial/versions/1"
UNVERSIONED_NOTEBOOK_HANDLE = "alexisbcook/titanic-tutorial"


class TestModelDownload(unittest.TestCase):
    def test_download_notebook_unversioned_output_succeeds(self) -> None:
        with create_test_cache():
            actual_path = notebook_output_download(UNVERSIONED_NOTEBOOK_HANDLE)

            expected_files = ["submission.csv"]
            assert_files(self, actual_path, expected_files)

    def test_download_notebook_versioned_output_succeeds(self) -> None:
        with create_test_cache():
            actual_path = notebook_output_download(VERSIONED_NOTEBOOK_HANDLE)

            expected_files = ["my_submission.csv"]
            assert_files(self, actual_path, expected_files)

    def test_download_both_notebook_outputs_succeeds(self) -> None:
        with create_test_cache():
            versioned_path = notebook_output_download(VERSIONED_NOTEBOOK_HANDLE)
            unversioned_path = notebook_output_download(UNVERSIONED_NOTEBOOK_HANDLE)

            expected_versioned_files = ["my_submission.csv"]
            expected_unversioned_files = ["submission.csv"]
            assert_files(self, versioned_path, expected_versioned_files)
            assert_files(self, unversioned_path, expected_unversioned_files)

    def test_download_public_notebook_output_as_unauthenticated_succeeds(self) -> None:
        with create_test_cache():
            with unauthenticated():
                actual_path = notebook_output_download(UNVERSIONED_NOTEBOOK_HANDLE)

                expected_files = ["submission.csv"]
                assert_files(self, actual_path, expected_files)

    def test_download_private_notebook_output_succeeds(self) -> None:
        with create_test_cache():
            actual_path = notebook_output_download("integrationtester/private-titanic-tutorial")

            expected_files = ["submission-01.csv", "submission-02.csv"]

            assert_files(self, actual_path, expected_files)

    def test_download_private_notebook_output_single_file_succeeds(self) -> None:
        with create_test_cache():
            actual_path = notebook_output_download(
                "integrationtester/private-titanic-tutorial", path="submission-02.csv"
            )

            expected_files = ["submission-02.csv"]

            assert_files(self, actual_path, expected_files)

    def test_download_large_notebook_output_warns(self) -> None:
        handle = "integrationtester/titanic-tutorial-many-output-files"
        with create_test_cache():
            # If the model has > 25 files, we warn the user that it's not supported yet
            # TODO(b/379761520): add support for .tar.gz archived downloads
            notebook_output_download(handle)
            msg = f"Too many files in {handle} (capped at 25). Unable to download notebook output."
            self.assertLogs(msg, "WARNING")

    def test_download_private_notebook_output_with_incorrect_file_path_fails(self) -> None:
        with create_test_cache(), self.assertRaises(HTTPError):
            notebook_output_download("integrationtester/titanic-tutorial", path="submission-03.csv")
