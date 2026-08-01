import ntpath
import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

import kagglehub
from kagglehub.clients import _detect_auto_compressed_member_name, build_kaggle_client, download_file, get_user_agent
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

    def test_detect_auto_compressed_member_name_matches(self) -> None:
        # Sanity check for the normal (non-Windows-specific) case: the server served
        # "shapes.csv" back as "shapes.csv.zip", so the archive member to extract is
        # "shapes.csv".
        out_file = os.path.join("some", "cache", "dir", "shapes.csv")
        response_url = "https://storage.googleapis.com/bucket/path/shapes.csv.zip?token=abc"

        self.assertEqual(_detect_auto_compressed_member_name(out_file, response_url), "shapes.csv")

    def test_detect_auto_compressed_member_name_no_match(self) -> None:
        # The server returned the exact file that was requested (not a same-named zip),
        # so no extraction should be triggered.
        out_file = os.path.join("some", "cache", "dir", "shapes.csv")
        response_url = "https://storage.googleapis.com/bucket/path/shapes.csv?token=abc"

        self.assertIsNone(_detect_auto_compressed_member_name(out_file, response_url))

    def test_detect_auto_compressed_member_name_windows_path_regression(self) -> None:
        # Regression test for https://github.com/Kaggle/kagglehub/issues/252.
        #
        # On Windows, `out_file` is an absolute path like "C:\\Users\\...\\shapes.csv"
        # (drive letter + backslash separators). The previous implementation parsed
        # `out_file` with `urllib.parse.urlparse`, which is meant for URLs, not local
        # paths: it misreads the "C:" drive prefix as a URL scheme, and then splits the
        # remainder on "/" -- which never matches Windows' "\\" separators. So the
        # "expected" filename came out as the *entire remaining path* instead of just
        # "shapes.csv", the comparison against the real (correctly-parsed) URL always
        # failed, and the auto-extraction was silently skipped -- leaving the raw zip
        # bytes cached under the filename that was supposed to be the extracted CSV.
        #
        # We force Windows path semantics with `ntpath` here (rather than relying on the
        # host OS actually being Windows) so this regression is caught on any platform's
        # CI, including the Linux/macOS runners this suite normally runs on.
        windows_out_file = ntpath.join("C:\\Users\\test\\.cache\\kagglehub", "shapes.csv")
        response_url = "https://storage.googleapis.com/bucket/path/shapes.csv.zip?token=abc"

        with patch("kagglehub.clients.os.path", ntpath):
            self.assertEqual(_detect_auto_compressed_member_name(windows_out_file, response_url), "shapes.csv")

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
