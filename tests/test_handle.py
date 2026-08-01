from unittest import mock

from kagglesdk.kaggle_env import KaggleEnv

from kagglehub.handle import (
    _web_endpoint_fallback,
    parse_competition_handle,
    parse_dataset_handle,
    parse_model_handle,
    parse_notebook_handle,
)
from tests.fixtures import BaseTestCase


class TestHandle(BaseTestCase):
    def test_versioned_model_handle(self) -> None:
        handle = "google/bert/tensorFlow2/answer-equivalence-bem/3"
        h = parse_model_handle(handle)

        self.assertEqual("google", h.owner)
        self.assertEqual("bert", h.model)
        self.assertEqual("tensorFlow2", h.framework)
        self.assertEqual("answer-equivalence-bem", h.variation)
        self.assertEqual(3, h.version)
        self.assertTrue(h.is_versioned())
        self.assertEqual(handle, str(h))

    def test_unversioned_model_handle(self) -> None:
        handle = "google/bert/tensorFlow2/answer-equivalence-bem"
        h = parse_model_handle(handle)
        self.assertEqual("google", h.owner)
        self.assertEqual("bert", h.model)
        self.assertEqual("tensorFlow2", h.framework)
        self.assertEqual("answer-equivalence-bem", h.variation)
        self.assertEqual(None, h.version)
        self.assertFalse(h.is_versioned())
        self.assertEqual(handle, str(h))

    def test_invalid_model_handle(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_handle("invalid")

    def test_invalid_version_model_handle(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_handle("google/bert/tensorFlow2/answer-equivalence-bem/invalid-version-number")

    def test_unversioned_dataset_handle(self) -> None:
        handle = "owner/dataset"
        h = parse_dataset_handle(handle)

        self.assertEqual("owner", h.owner)
        self.assertEqual("dataset", h.dataset)
        self.assertEqual(None, h.version)
        self.assertFalse(h.is_versioned())
        self.assertEqual(handle, str(h))

    def test_invalid_dataset_handle(self) -> None:
        with self.assertRaises(ValueError):
            parse_dataset_handle("a-single-part")

    def test_versioned_dataset_handle(self) -> None:
        handle = "owner/versionedDataset/versions/2"
        h = parse_dataset_handle(handle)

        self.assertEqual("owner", h.owner)
        self.assertEqual("versionedDataset", h.dataset)
        self.assertEqual(2, h.version)
        self.assertTrue(h.is_versioned())
        self.assertEqual(handle, str(h))

    def test_invalid_competition_handle(self) -> None:
        with self.assertRaises(ValueError):
            parse_competition_handle("ownerSlugNotNeeded/onlyCompetitionSlug")

    def test_competition_handle(self) -> None:
        handle = "titanic"

        h = parse_competition_handle(handle)

        self.assertEqual("titanic", h.competition)

    def test_code_handle(self) -> None:
        handle = "owner/notebook"
        h = parse_notebook_handle(handle)

        self.assertEqual("owner", h.owner)
        self.assertEqual("notebook", h.notebook)

    def test_invalid_code_handle(self) -> None:
        with self.assertRaises(ValueError):
            parse_notebook_handle("notebook")

    def test_to_url_uses_web_endpoint_not_api_host(self) -> None:
        # Detail pages must use kaggle.com, not api.kaggle.com (see kagglesdk.get_web_endpoint).
        model = parse_model_handle("google/bert/tensorFlow2/answer-equivalence-bem/3")
        dataset = parse_dataset_handle("owner/dataset/versions/2")
        competition = parse_competition_handle("titanic")
        notebook = parse_notebook_handle("owner/notebook/versions/1")

        with mock.patch("kagglehub.handle.get_env", return_value=KaggleEnv.PROD):
            self.assertEqual(
                "https://kaggle.com/models/google/bert/tensorFlow2/answer-equivalence-bem/3",
                model.to_url(),
            )
            self.assertEqual("https://kaggle.com/datasets/owner/dataset/versions/2", dataset.to_url())
            self.assertEqual("https://kaggle.com/competitions/titanic", competition.to_url())
            self.assertEqual("https://kaggle.com/code/owner/notebook/versions/1", notebook.to_url())

    def test_web_endpoint_fallback_prod_maps_to_kaggle_com(self) -> None:
        self.assertEqual("https://kaggle.com", _web_endpoint_fallback(KaggleEnv.PROD))

    def test_web_endpoint_fallback_non_prod_uses_get_endpoint(self) -> None:
        with mock.patch("kagglehub.handle.get_endpoint", return_value="https://staging.kaggle.com") as get_endpoint:
            self.assertEqual("https://staging.kaggle.com", _web_endpoint_fallback(KaggleEnv.STAGING))
            get_endpoint.assert_called_once_with(KaggleEnv.STAGING)
