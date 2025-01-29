import logging
import os
import time
from typing import Optional

from kagglehub.clients import ColabClient
from kagglehub.config import is_colab_cache_disabled
from kagglehub.exceptions import BackendError, NotFoundError
from kagglehub.handle import DatasetHandle, ModelHandle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.resolver import Resolver

COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME = "COLAB_CACHE_MOUNT_FOLDER"
DEFAULT_COLAB_CACHE_MOUNT_FOLDER = "/kaggle/input"

logger = logging.getLogger(__name__)


class ModelColabCacheResolver(Resolver[ModelHandle]):
    def is_supported(self, handle: ModelHandle, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if ColabClient.TBE_RUNTIME_ADDR_ENV_VAR_NAME not in os.environ or is_colab_cache_disabled():
            return False

        api_client = ColabClient()
        data = {
            "owner": handle.owner,
            "model": handle.model,
            "framework": handle.framework,
            "variation": handle.variation,
        }

        if handle.is_versioned():
            # Colab treats version as int in the request
            data["version"] = handle.version  # type: ignore

        try:
            api_client.post(data, ColabClient.IS_MODEL_SUPPORTED_PATH, handle)
        except NotFoundError:
            return False
        return True

    def __call__(self, h: ModelHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
        if force_download:
            logger.info(
                "Ignoring `force_download` argument when running inside the Colab notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )

        api_client = ColabClient()
        data = {
            "owner": h.owner,
            "model": h.model,
            "framework": h.framework,
            "variation": h.variation,
        }
        if h.is_versioned():
            # Colab treats version as int in the request
            data["version"] = h.version  # type: ignore

        response = api_client.post(data, ColabClient.MODEL_MOUNT_PATH, h)

        if response is None:
            no_response = "No response received or response was empty."
            raise ValueError(no_response)

        if "slug" not in response:
            msg = "'slug' field missing from response"
            raise BackendError(msg)

        base_mount_path = os.getenv(COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME, DEFAULT_COLAB_CACHE_MOUNT_FOLDER)
        cached_path = f"{base_mount_path}/{response['slug']}"

        if not os.path.exists(cached_path):
            # Only print this if the model is not already mounted.
            logger.info(f"Mounting files to {cached_path}...")

        while not os.path.exists(cached_path):
            time.sleep(5)

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


class DatasetColabCacheResolver(Resolver[DatasetHandle]):
    def is_supported(self, handle: DatasetHandle, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if ColabClient.TBE_RUNTIME_ADDR_ENV_VAR_NAME not in os.environ or is_colab_cache_disabled():
            return False

        api_client = ColabClient()
        data = {
            "owner": handle.owner,
            "dataset": handle.dataset,
        }

        if handle.is_versioned():
            # Colab treats version as int in the request
            data["version"] = handle.version  # type: ignore

        try:
            api_client.post(data, ColabClient.IS_DATASET_SUPPORTED_PATH, handle)
        except NotFoundError:
            return False
        return True

    def __call__(self, h: DatasetHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
        if force_download:
            logger.info(
                "Ignoring `force_download` argument when running inside the Colab notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )

        api_client = ColabClient()
        data = {
            "owner": h.owner,
            "dataset": h.dataset,
        }
        if h.is_versioned():
            # Colab treats version as int in the request
            data["version"] = h.version  # type: ignore

        response = api_client.post(data, ColabClient.DATASET_MOUNT_PATH, h)

        if response is None:
            no_response = "No response received or response was empty."
            raise ValueError(no_response)

        if "slug" not in response:
            msg = "'slug' field missing from response"
            raise BackendError(msg)

        base_mount_path = os.getenv(COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME, DEFAULT_COLAB_CACHE_MOUNT_FOLDER)
        cached_path = f"{base_mount_path}/{response['slug']}"

        if not os.path.exists(cached_path):
            # Only print this if the dataset is not already mounted.
            logger.info(f"Mounting files to {cached_path}...")

        while not os.path.exists(cached_path):
            time.sleep(5)

        if path:
            cached_filepath = f"{cached_path}/{path}"
            if not os.path.exists(cached_filepath):
                msg = (
                    f"'{path}' is not present in the dataset files. "
                    f"You can access the other files of the attached dataset at '{cached_path}'"
                )
                raise ValueError(msg)
            return cached_filepath
        return cached_path
