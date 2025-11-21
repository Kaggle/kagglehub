import unittest

from requests import HTTPError

from kagglehub import competition_download

from .utils import assert_columns, assert_files, create_test_cache

HANDLE = "titanic"
IEEE_FRAUD_DETECTION_HANDLE = "ieee-fraud-detection"


class TestCompetitionDownload(unittest.TestCase):
    def test_competition_succeeds(self) -> None:
        with create_test_cache():
            expected_files = [
                "gender_submission.csv",
                "test.csv",
                "train.csv",
            ]

            actual_path = competition_download(HANDLE)

            assert_files(self, actual_path, expected_files)

    def test_competition_single_file_succeeds(self) -> None:
        with create_test_cache():
            expected_files = [
                "gender_submission.csv",
            ]

            actual_path = competition_download(HANDLE, "gender_submission.csv")

            assert_files(self, actual_path, expected_files)

    def test_competition_competition_rules_accepted_succeeds(self) -> None:
        with create_test_cache():
            expected_files = [
                "sample_submission.csv",
                "test_identity.csv",
                "test_transaction.csv",
                "train_identity.csv",
                "train_transaction.csv",
            ]

            actual_path = competition_download(IEEE_FRAUD_DETECTION_HANDLE)

            assert_files(self, actual_path, expected_files)

    def test_competition_competition_rules_not_accepted_fails(self) -> None:
        # integrationtester bot has not accepted competiton rules
        with self.assertRaises(HTTPError) as cm:
            competition_download("jane-street-market-prediction")
        exception_msg = str(cm.exception)
        self.assertTrue(exception_msg.startswith("403"))
        self.assertIn("You don't have permission to access resource at URL", exception_msg)

    def test_competition_multiple_files(self) -> None:
        with create_test_cache():
            file_paths = [
                "gender_submission.csv",
                "test.csv",
                "train.csv",
            ]
            for p in file_paths:
                actual_path = competition_download(HANDLE, path=p)
                assert_files(self, actual_path, [p])

    def test_auto_decompress_file(self) -> None:
        with create_test_cache():
            # sample_submission.csv is an auto-compressed CSV with the following columns
            expected_columns = ["TransactionID", "isFraud"]
            actual_path = competition_download(IEEE_FRAUD_DETECTION_HANDLE, path="sample_submission.csv")
            assert_columns(self, actual_path, expected_columns)

    def test_competition_with_incorrect_file_path(self) -> None:
        incorrect_path = "nonxisten/Test"
        with self.assertRaises(HTTPError) as cm:
            competition_download(HANDLE, path=incorrect_path)
        exception_msg = str(cm.exception)
        self.assertTrue(exception_msg.startswith("404"))
        self.assertIn("Resource not found at URL", exception_msg)
