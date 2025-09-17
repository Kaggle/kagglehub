import base64
import logging

import requests

# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"
COMPUTE_HASH_CHUNK_SIZE = 8192

logger = logging.getLogger(__name__)


def get_md5_checksum_from_response(response: requests.Response) -> str | None:
    # See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
    # Format is: x-goog-hash: crc32c=n03x6A==,md5=Ojk9c3dhfxgoKVVHYwFbHQ==
    if GCS_HASH_HEADER in response.headers:
        header_value = response.headers[GCS_HASH_HEADER]
        for checksum in header_value.split(","):
            try:
                name, value = checksum.strip().split("=", 1)
                if name == "md5":
                    return value
            except ValueError:
                logger.warning(f"Invalid {GCS_HASH_HEADER} header: {header_value}")
                return None
    return None


def update_hash_from_file(hash_object, out_file: str) -> None:  # noqa: ANN001 - no public type for hashlib hash
    if hash_object is None:
        return

    with open(out_file, "rb") as f:
        chunk = f.read(COMPUTE_HASH_CHUNK_SIZE)
        while chunk:
            hash_object.update(chunk)
            chunk = f.read(COMPUTE_HASH_CHUNK_SIZE)


def to_b64_digest(hash_object) -> str:  # noqa: ANN001 - no public type for hashlib hash
    return base64.b64encode(hash_object.digest()).decode("utf-8")
