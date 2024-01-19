# import os
# import tempfile
# import unittest

# from kagglehub import model_upload, models_helpers

# LICENSE_NAME = "MIT"
# OWNER_SLUG = "integrationtester"
# MODEL_SLUG = "test-private-model"


# class TestModelUpload(unittest.TestCase):
#     def setUp(self):
#         self.temp_dir = tempfile.mkdtemp()
#         self.dummy_files = ["dummy_model.h5", "config.json", "metadata.json"]
#         for file in self.dummy_files:
#             with open(os.path.join(self.temp_dir, file), "w") as f:
#                 f.write("dummy content")
#         self.handle = f"OWNER_SLUG/MODEL_SLUG/pyTorch/new-variation"

#     def test_model_upload_and_versioning(self):
#         # Create Instance
#         model_upload(self.handle, self.temp_dir, LICENSE_NAME)

#         # Create Version
#         model_upload(self.handle, self.temp_dir, LICENSE_NAME)

#         # If delete model does not raise an error, then the upload was successful.

#     def tearDown(self):
#         models_helpers.delete_model(OWNER_SLUG, MODEL_SLUG)
