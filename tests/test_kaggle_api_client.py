import os
from tempfile import TemporaryDirectory

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import DataCorruptionError
from tests.fixtures import BaseTestCase

from .server_stubs import kaggle_api_stub as stub
from .server_stubs import serv


class TestKaggleApiV1Client(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_download_with_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            api_client = KaggleApiV1Client()
            api_client.download_file("good", out_file)

            with open(out_file) as f:
                self.assertEqual("foo", f.read())

    def test_resumable_download_with_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            # If the out_file already has data, we use the 'Range' header to resume download.
            with open(out_file, "w") as f:
                f.write("fo")  # Should download the remaining "o".

            api_client = KaggleApiV1Client()
            with self.assertLogs("kagglehub", level="INFO") as cm:
                api_client.download_file("good", out_file)
                self.assertIn("INFO:kagglehub.clients:Resuming download from 2 bytes (1 bytes left)...", cm.output)

            with open(out_file) as f:
                self.assertEqual("foo", f.read())

    def test_download_no_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            api_client = KaggleApiV1Client()
            api_client.download_file("no-integrity", out_file)

            with open(out_file) as f:
                self.assertEqual("foo", f.read())

    def test_download_corrupted_file_fail_integrity_check(self) -> None:
        with TemporaryDirectory() as d:
            out_file = os.path.join(d, "out")

            api_client = KaggleApiV1Client()
            with self.assertRaises(DataCorruptionError):
                api_client.download_file("corrupted", out_file)

            # Assert the corrupted file has been deleted.
            self.assertFalse(os.path.exists(out_file))
