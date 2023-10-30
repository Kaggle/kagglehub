import unittest

from kagglehub.handle import parse_model_handle


class TestHandle(unittest.TestCase):
    def test_versioned_model_handle(self):
        h = parse_model_handle("google/bert/tensorFlow2/answer-equivalence-bem/3")

        self.assertEqual("google", h.owner)
        self.assertEqual("bert", h.model)
        self.assertEqual("tensorFlow2", h.framework)
        self.assertEqual("answer-equivalence-bem", h.variation)
        self.assertEqual(3, h.version)
        self.assertTrue(h.is_versioned())

    def test_unversioned_model_handle(self):
        h = parse_model_handle("google/bert/tensorFlow2/answer-equivalence-bem")
        self.assertEqual("google", h.owner)
        self.assertEqual("bert", h.model)
        self.assertEqual("tensorFlow2", h.framework)
        self.assertEqual("answer-equivalence-bem", h.variation)
        self.assertEqual(None, h.version)
        self.assertFalse(h.is_versioned())

    def test_invalid_model_handle(self):
        with self.assertRaises(ValueError):
            parse_model_handle("invalid")

    def test_invalid_version_model_handle(self):
        with self.assertRaises(ValueError):
            parse_model_handle("google/bert/tensorFlow2/answer-equivalence-bem/invalid-version-number")
