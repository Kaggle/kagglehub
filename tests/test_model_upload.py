import unittest

import kagglehub


class TestModelUpload(unittest.TestCase):
    def test_model_upload(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_upload()
