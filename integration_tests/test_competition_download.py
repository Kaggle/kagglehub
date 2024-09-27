import unittest

from requests import HTTPError

from kagglehub import competition_download

from .utils import assert_files, create_test_cache

HANDLE = "titanic"


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

    def test_competition_competition_rules_accepted_succeeds(self) -> None:
        with create_test_cache():
            expected_files = [
                "sample_submission.csv",
                "test_identity.csv",
                "test_transaction.csv",
                "train_identity.csv",
                "train_transaction.csv",
            ]

            actual_path = competition_download("ieee-fraud-detection")

            assert_files(self, actual_path, expected_files)

    def test_competition_competition_rules_not_accepted_fails(self) -> None:
        # integrationtester bot has not accepted competiton rules
        with self.assertRaises(HTTPError) as e:
            competition_download("jane-street-market-prediction")
            self.assertEqual(e.exception.errno, 403)

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

    def test_competition_with_incorrect_file_path(self) -> None:
        incorrect_path = "nonxisten/Test"
        with self.assertRaises(HTTPError) as e:
            competition_download(HANDLE, path=incorrect_path)
            self.assertEqual(e.exception.errno, 403)
