from pathlib import Path
from tempfile import TemporaryDirectory

import kagglehub
from kagglehub.handle import parse_model_handle
from kagglehub.tracker import read_file, write_file
from tests.fixtures import BaseTestCase

from .server_stubs import model_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

UNVERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata"
VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/3"
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"


class TestHttpRequirements(BaseTestCase):

    def setUp(self) -> None:
        # Clear out our tracking between tests
        kagglehub.tracker._accessed_datasources = {}

    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_two_models(self) -> None:
        with create_test_cache():
            kagglehub.model_download(UNVERSIONED_MODEL_HANDLE)
            kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            with TemporaryDirectory() as d:
                requirements_path = str(Path(d) / "requirements.yaml")
                write_file(requirements_path)
                datasources = read_file(requirements_path)

            self.assertEqual(2, len(datasources))
            # Check the versions of each accessed datasource
            self.assertEqual(3, datasources[parse_model_handle(UNVERSIONED_MODEL_HANDLE)])
            self.assertEqual(3, datasources[parse_model_handle(VERSIONED_MODEL_HANDLE)])

    def test_no_datasources(self) -> None:
        with create_test_cache():
            with TemporaryDirectory() as d:
                requirements_path = str(Path(d) / "requirements.yaml")
                write_file(requirements_path)
                datasources = read_file(requirements_path)

            self.assertEqual(0, len(datasources))
