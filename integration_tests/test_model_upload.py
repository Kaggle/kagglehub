import os
import tempfile
import unittest
import uuid
from kagglehub import model_upload
from kagglehub.clients import KaggleApiV1Client

class TestModelUpload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_files = ['dummy_model.h5', 'config.json', 'metadata.json']
        for file in self.dummy_files:
            with open(os.path.join(self.temp_dir, file), 'w') as f:
                f.write('dummy content')
        self.model_uuid = str(uuid.uuid4())
        self.owner_slug = "aminmohamedmohami"
        self.model_slug = "model_{self.model_uuid}"
        self.handle = f'{self.owner_slug}/{self.model_slug}/pyTorch/new-variation'

    def delete_model(self, owner_slug, model_slug):
        api_client = KaggleApiV1Client()
        api_client.post(
            f"/models/{self.owner_slug}/{self.model_slug}/delete",
            {},
        )

    def test_model_upload_and_versioning(self):
        license_name = 'MIT'

        model_upload(self.handle, self.temp_dir, license_name)

        # ... [verify first upload]

        model_upload(self.handle, self.temp_dir, license_name)

        # ... [verify second upload]

    def tearDown(self):
        # Delete the model using the same UUID
        self.delete_model(self.owner_slug, self.model_slug)