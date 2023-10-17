import unittest

import kagglehub

MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"


class TestModelDownload(unittest.TestCase):
    def test_model_download(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_download(MODEL_HANDLE)
