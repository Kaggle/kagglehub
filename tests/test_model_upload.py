import kagglehub
from tests.fixtures import BaseTestCase


class TestModelUpload(BaseTestCase):
    def test_model_upload(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_upload()
