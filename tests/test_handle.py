from kagglehub.handle import parse_model_handle
from tests.fixtures import BaseTestCase


class TestHandle(BaseTestCase):
    def test_versioned_model_handle(self):
        handle = "google/bert/tensorFlow2/answer-equivalence-bem/3"
        h = parse_model_handle(handle)

        self.assertEqual("google", h.owner)
        self.assertEqual("bert", h.model)
        self.assertEqual("tensorFlow2", h.framework)
        self.assertEqual("answer-equivalence-bem", h.variation)
        self.assertEqual(3, h.version)
        self.assertTrue(h.is_versioned())
        self.assertEqual(handle, str(h))

    def test_unversioned_model_handle(self):
        handle = "google/bert/tensorFlow2/answer-equivalence-bem"
        h = parse_model_handle(handle)
        self.assertEqual("google", h.owner)
        self.assertEqual("bert", h.model)
        self.assertEqual("tensorFlow2", h.framework)
        self.assertEqual("answer-equivalence-bem", h.variation)
        self.assertEqual(None, h.version)
        self.assertFalse(h.is_versioned())
        self.assertEqual(handle, str(h))

    def test_invalid_model_handle(self):
        with self.assertRaises(ValueError):
            parse_model_handle("invalid")

    def test_invalid_version_model_handle(self):
        with self.assertRaises(ValueError):
            parse_model_handle("google/bert/tensorFlow2/answer-equivalence-bem/invalid-version-number")
