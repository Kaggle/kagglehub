import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

import kagglehub
from kagglehub.clients import build_kaggle_client, download_file, get_user_agent
from kagglehub.exceptions import DataCorruptionError
from kagglehub.handle import DatasetHandle
from tests.fixtures import BaseTestCase

from .server_stubs import kaggle_api_stub as stub
from .server_stubs import serv

DUMMY_HANDLE = DatasetHandle("dummy", "dataset")


class TestKaggleClient(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_download_with_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            with build_kaggle_client() as api_client:
                r = ApiDownloadDatasetRequest()
                r.dataset_slug = "no-integrity"

                response = api_client.datasets.dataset_api_client.download_dataset(r)
                download_file(response, out_file, DUMMY_HANDLE)

            with open(out_file) as f:
                self.assertEqual("foo", f.read())

    def test_resumable_download_with_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            # If the out_file already has data, we use the 'Range' header to resume download.
            with open(out_file, "w") as f:
                f.write("fo")  # Should download the remaining "o".

            with self.assertLogs("kagglehub", level="INFO") as cm:
                with build_kaggle_client() as api_client:
                    r = ApiDownloadDatasetRequest()
                    r.dataset_slug = "good"

                    response = api_client.datasets.dataset_api_client.download_dataset(r)
                    download_file(response, out_file, DUMMY_HANDLE)

                    self.assertIn("INFO:kagglehub.clients:Resuming download from 2 bytes (1 bytes left)...", cm.output)

            with open(out_file) as f:
                self.assertEqual("foo", f.read())

    def test_download_no_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            with build_kaggle_client() as api_client:
                r = ApiDownloadDatasetRequest()
                r.dataset_slug = "no-integrity"

                response = api_client.datasets.dataset_api_client.download_dataset(r)
                download_file(response, out_file, DUMMY_HANDLE)

            with open(out_file) as f:
                self.assertEqual("foo", f.read())

    def test_download_corrupted_file_fail_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            with build_kaggle_client() as api_client:
                r = ApiDownloadDatasetRequest()
                r.dataset_slug = "corrupted"

                with self.assertRaises(DataCorruptionError):
                    response = api_client.datasets.dataset_api_client.download_dataset(r)
                    download_file(response, out_file, DUMMY_HANDLE)

            # Assert the corrupted file has been deleted.
            self.assertFalse(os.path.exists(out_file))

    @patch.dict("os.environ", {})
    def test_get_user_agent(self) -> None:
        self.assertEqual(get_user_agent(), f"kagglehub/{kagglehub.__version__}")

    @patch.dict(
        "os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Interactive", "KAGGLE_DATA_PROXY_URL": "https://dp.kaggle.net"}
    )
    def test_get_user_agent_kkb(self) -> None:
        self.assertEqual(get_user_agent(), f"kagglehub/{kagglehub.__version__} kkb/unknown")

    @patch.dict(
        "os.environ",
        {
            "COLAB_RELEASE_TAG": "release-colab-20230531-060125-RC00",
        },
    )
    @patch("kagglehub.env._is_google_colab", True)
    def test_get_user_agent_colab(self) -> None:
        self.assertEqual(
            get_user_agent(),
            f"kagglehub/{kagglehub.__version__} colab/release-colab-20230531-060125-RC00-unmanaged",
        )

    @patch("importlib.metadata.version")
    @patch("inspect.ismodule")
    @patch("inspect.stack")
    def test_get_user_agent_keras_nlp(
        self, mock_stack: MagicMock, mock_is_module: MagicMock, mock_version: MagicMock
    ) -> None:
        # Mock the call stack and version information.
        mock_stack.return_value = [
            MagicMock(frame=MagicMock(__name__="kagglehub.clients")),
            MagicMock(frame=MagicMock(__name__="kagglehub.models_helpers")),
            MagicMock(frame=MagicMock(__name__="kagglehub.models")),
            MagicMock(frame=MagicMock(__name__="keras_nlp.src.utils.preset_utils")),
            MagicMock(frame=MagicMock(None)),
        ]
        mock_is_module.return_value = True
        mock_version.return_value = "0.15.0"
        self.assertEqual(get_user_agent(), f"kagglehub/{kagglehub.__version__} keras_nlp/0.15.0")

    @patch("importlib.metadata.version")
    @patch("inspect.ismodule")
    @patch("inspect.stack")
    def test_get_user_agent_keras_hub(
        self, mock_stack: MagicMock, mock_is_module: MagicMock, mock_version: MagicMock
    ) -> None:
        # Mock the call stack and version information.
        mock_stack.return_value = [
            MagicMock(frame=MagicMock(__name__="kagglehub.clients")),
            MagicMock(frame=MagicMock(__name__="kagglehub.models_helpers")),
            MagicMock(frame=MagicMock(__name__="kagglehub.models")),
            MagicMock(frame=MagicMock(__name__="keras_hub.src.utils.preset_utils")),
            MagicMock(frame=MagicMock(None)),
        ]
        mock_is_module.return_value = True
        mock_version.return_value = "0.17.0"
        self.assertEqual(get_user_agent(), f"kagglehub/{kagglehub.__version__} keras_hub/0.17.0")

    @patch("importlib.metadata.version")
    @patch("inspect.ismodule")
    @patch("inspect.stack")
    def test_get_user_agent_torch_tune(
        self, mock_stack: MagicMock, mock_is_module: MagicMock, mock_version: MagicMock
    ) -> None:
        # Mock the call stack and version information.
        mock_stack.return_value = [
            MagicMock(frame=MagicMock(__name__="kagglehub.clients")),
            MagicMock(frame=MagicMock(__name__="kagglehub.models_helpers")),
            MagicMock(frame=MagicMock(__name__="kagglehub.models")),
            MagicMock(frame=MagicMock(__name__="torchtune.src.utils.preset_utils")),
            MagicMock(frame=MagicMock(None)),
        ]
        mock_is_module.return_value = True
        mock_version.return_value = "0.18.0"
        self.assertEqual(get_user_agent(), f"kagglehub/{kagglehub.__version__} torchtune/0.18.0")
