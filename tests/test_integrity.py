import requests

from kagglehub.integrity import GCS_HASH_HEADER, get_md5_checksum_from_response
from tests.fixtures import BaseTestCase


class TestCache(BaseTestCase):
    def test_get_md5_checksum_from_response_only_md5(self) -> None:
        response = requests.Response()
        response.headers[GCS_HASH_HEADER] = "md5=foo"

        self.assertEqual("foo", get_md5_checksum_from_response(response))

    def test_get_md5_checksum_from_response_many_hash_algorithms(self) -> None:
        response = requests.Response()
        response.headers[GCS_HASH_HEADER] = "crc32c=n03x6A==,md5=bar"

        self.assertEqual("bar", get_md5_checksum_from_response(response))

    def test_get_md5_checksum_from_response_no_header(self) -> None:
        response = requests.Response()

        self.assertIsNone(get_md5_checksum_from_response(response))

    def test_get_md5_checksum_from_response_malformed_header(self) -> None:
        response = requests.Response()
        response.headers[GCS_HASH_HEADER] = "malformed"

        self.assertIsNone(get_md5_checksum_from_response(response))
