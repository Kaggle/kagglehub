import logging
import os
import time
from typing import Optional

from kagglehub.clients import DEFAULT_CONNECT_TIMEOUT, KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME, KaggleJwtClient
from kagglehub.config import is_kaggle_cache_disabled
from kagglehub.exceptions import BackendError
from kagglehub.handle import ModelHandle
from kagglehub.resolver import Resolver

KAGGLE_NOTEBOOK_ENV_VAR_NAME = "KAGGLE_KERNEL_RUN_TYPE"
KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME = "KAGGLE_CACHE_MOUNT_FOLDER"
ATTACH_DATASOURCE_REQUEST_NAME = "AttachDatasourceUsingJwtRequest"
# b/312965617: Using a longer timeout for this RPC.
ATTACH_DATASOURCE_READ_TIMEOUT = 30  # seconds

DEFAULT_KAGGLE_CACHE_MOUNT_FOLDER = "/kaggle/input"


logger = logging.getLogger(__name__)


class KaggleCacheResolver(Resolver):
    def is_supported(self, *_) -> bool:
        if is_kaggle_cache_disabled():
            return False

        if KAGGLE_NOTEBOOK_ENV_VAR_NAME in os.environ:
            # Inside a Kaggle notebook.
            if KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME not in os.environ:
                # Missing endpoint for the Jwt client.
                logger.warning(
                    "Can't use the Kaggle Cache. "
                    f"The '{KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME}' environment variable is not set."
                )
                return False
            return True

        return False

    def __call__(self, h: ModelHandle, path: Optional[str] = None) -> str:
        logger.info(f"Attaching model '{h}' to your Kaggle notebook...")
        client = KaggleJwtClient()
        model_ref = {
            "OwnerSlug": h.owner,
            "ModelSlug": h.model,
            "Framework": h.framework,
            "InstanceSlug": h.variation,
        }
        if h.is_versioned():
            model_ref["VersionNumber"] = str(h.version)

        result = client.post(
            ATTACH_DATASOURCE_REQUEST_NAME,
            {
                "modelRef": model_ref,
            },
            timeout=(DEFAULT_CONNECT_TIMEOUT, ATTACH_DATASOURCE_READ_TIMEOUT),
        )
        if "mountSlug" not in result:
            msg = "'result.mountSlug' field missing from response"
            raise BackendError(msg)

        base_mount_path = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME, DEFAULT_KAGGLE_CACHE_MOUNT_FOLDER)
        cached_path = f"{base_mount_path}/{result['mountSlug']}"

        logger.info(f"Mounting files to {cached_path}...")
        while not os.path.exists(cached_path):
            time.sleep(5)

        logger.info(f"Model '{h}' is attached.")

        if path:
            cached_filepath = f"{cached_path}/{path}"
            if not os.path.exists(cached_filepath):
                msg = (
                    f"'{path}' is not present in the model files. "
                    f"You can access the other files of the attached model at '{cached_path}'"
                )
                raise ValueError(msg)
            return cached_filepath
        return cached_path
