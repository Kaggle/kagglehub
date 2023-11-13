import unittest

import kagglehub
from tests.fixtures import BaseTest


class TestModelUpload(BaseTest):
    def test_model_upload(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_upload()
