import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.models import model_upload
from kagglehub.tracing import TraceContext
from tests.fixtures import BaseTestCase

from .server_stubs import model_upload_stub as stub
from .server_stubs import serv

_CANONICAL_EXAMPLE = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


class TraceContextSuite(unittest.TestCase):
    def test_length(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            self.assertEqual(len(traceparent), len(_CANONICAL_EXAMPLE))

    def test_prefix(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            self.assertEqual(traceparent[0:2], "00")

    def test_suffix(self) -> None:
        # always sample
        with TraceContext() as ctx:
            traceparent = ctx.next()
            self.assertEqual(traceparent[-2:], "01")

    def test_pattern(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            version, trace, span, flag = traceparent.split("-")
            self.assertRegex(version, "^[0-9]{2}$", "version does not meet pattern")
            self.assertRegex(trace, "^[A-Fa-f0-9]{32}$")
            self.assertRegex(span, "^[A-Fa-f0-9]{16}$")
            self.assertRegex(flag, "^[0-9]{2}$")

    def test_notempty(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            _, trace, span, _ = traceparent.split("-")
            self.assertNotEqual(trace, f"{0:016x}")
            self.assertNotEqual(span, f"{0:08x}")


class TestModelUpload(BaseTestCase):
    def setUp(self) -> None:
        stub.reset()

    @classmethod
    def setUpClass(cls):  # noqa: ANN102
        serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):  # noqa: ANN102
        serv.stop_server()

    def test_model_upload_instance_with_valid_handle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / "temp_test_file"
            test_filepath.touch()  # Create a temporary file in the temporary directory
            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, "Apache 2.0", "model_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn("temp_test_file", stub.shared_data.files)
            self.assertGreaterEqual(stub.shared_data.traceparent_header_count, 2)


if __name__ == "__main__":
    unittest.main()
