from kagglehub.handle import parse_competition_handle, parse_dataset_handle, parse_model_handle, parse_notebook_handle
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
