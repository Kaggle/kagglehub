import logging
import os
import time
from typing import Optional

from kagglehub.clients import (
    DEFAULT_CONNECT_TIMEOUT,
    KaggleJwtClient,
)
from kagglehub.config import is_kaggle_cache_disabled
from kagglehub.env import is_in_kaggle_notebook
from kagglehub.exceptions import BackendError
from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle, NotebookHandle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.resolver import Resolver

KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME = "KAGGLE_CACHE_MOUNT_FOLDER"
ATTACH_DATASOURCE_REQUEST_NAME = "AttachDatasourceUsingJwtRequest"
# b/312965617: Using a longer timeout for this RPC.
ATTACH_DATASOURCE_READ_TIMEOUT = 30  # seconds

# TODO(b/385788821): this couples the client to backend implementation details
#   in AttachDatasourceUsingJwt. We should remove the need to prepend this in
#   kagglehub, and have AttachDatasourceUsingJwt return the actual mount path.
DEFAULT_KAGGLE_CACHE_MOUNT_FOLDER = "/kaggle/input"


logger = logging.getLogger(__name__)


class CompetitionKaggleCacheResolver(Resolver[CompetitionHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if is_kaggle_cache_disabled():
            return False
        if is_in_kaggle_notebook():
            return True
        return False

    def __call__(
        self, h: CompetitionHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False
    ) -> str:
        client = KaggleJwtClient()
        if force_download:
            logger.info(
                "Ignoring `force_download` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )

        competition_ref = {
            "CompetitionSlug": h.competition,
        }
        result = client.post(
            ATTACH_DATASOURCE_REQUEST_NAME,
            {
                "competitionRef": competition_ref,
            },
            timeout=(DEFAULT_CONNECT_TIMEOUT, ATTACH_DATASOURCE_READ_TIMEOUT),
        )

        if "mountSlug" not in result:
            msg = "'result.mountSlug' field missing from response"
            raise BackendError(msg)

        base_mount_path = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME, DEFAULT_KAGGLE_CACHE_MOUNT_FOLDER)
        cached_path = f"{base_mount_path}/{result['mountSlug']}"
        if not os.path.exists(cached_path):
            # Only print this if the competition is not already mounted.
            logger.info(f"Mounting files to {cached_path}...")
        elif path:
            logger.info(
                f"Attaching '{path}' from competition '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
        else:
            logger.info(
                f"Attaching competition '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
        while not os.path.exists(cached_path):
            time.sleep(5)
        if path:
            cached_filepath = f"{cached_path}/{path}"
            if not os.path.exists(cached_filepath):
                msg = (
                    f"'{path}' is not present in the competition files."
                    f"You can acces the other files othe attached competition at '{cached_path}'"
                )
                raise ValueError(msg)
            return cached_filepath
        return cached_path


class DatasetKaggleCacheResolver(Resolver[DatasetHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if is_kaggle_cache_disabled():
            return False

        if is_in_kaggle_notebook():
            return True

        return False

    def __call__(self, h: DatasetHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
        if force_download:
            logger.info(
                "Ignoring `force_download` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
        client = KaggleJwtClient()
        dataset_ref = {
            "OwnerSlug": h.owner,
            "DatasetSlug": h.dataset,
        }
        if h.is_versioned():
            dataset_ref["VersionNumber"] = str(h.version)

        result = client.post(
            ATTACH_DATASOURCE_REQUEST_NAME,
            {
                "datasetRef": dataset_ref,
            },
            timeout=(DEFAULT_CONNECT_TIMEOUT, ATTACH_DATASOURCE_READ_TIMEOUT),
        )
        if "mountSlug" not in result:
            msg = "'result.mountSlug' field missing from response"
            raise BackendError(msg)

        base_mount_path = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME, DEFAULT_KAGGLE_CACHE_MOUNT_FOLDER)
        cached_path = f"{base_mount_path}/{result['mountSlug']}"

        if not os.path.exists(cached_path):
            # Only print this if the dataset is not already mounted.
            logger.info(f"Mounting files to {cached_path}...")
        elif path:
            logger.info(
                f"Attaching '{path}' from dataset '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
        else:
            logger.info(
                f"Attaching dataset '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )

        while not os.path.exists(cached_path):
            time.sleep(5)

        if path:
            cached_filepath = f"{cached_path}/{path}"
            if not os.path.exists(cached_filepath):
                msg = (
                    f"'{path}' is not present in the dataset files."
                    f"You can acces the other files othe attached dataset at '{cached_path}'"
                )
                raise ValueError(msg)
            return cached_filepath
        return cached_path


class ModelKaggleCacheResolver(Resolver[ModelHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if is_kaggle_cache_disabled():
            return False

        if is_in_kaggle_notebook():
            return True

        return False

    def __call__(self, h: ModelHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
        if force_download:
            logger.info(
                "Ignoring `force_download` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
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

        if not os.path.exists(cached_path):
            # Only print this if the model is not already mounted.
            logger.info(f"Mounting files to {cached_path}...")
        elif path:
            logger.info(
                f"Attaching '{path}' from model '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
        else:
            logger.info(
                f"Attaching model '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )

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


class NotebookOutputKaggleCacheResolver(Resolver[NotebookHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if is_kaggle_cache_disabled():
            return False

        if is_in_kaggle_notebook():
            return True

        return False

    def __call__(self, h: NotebookHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
        if force_download:
            logger.info(
                "Ignoring `force_download` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
        client = KaggleJwtClient()
        kernel_ref = {
            "OwnerSlug": h.owner,
            "KernelSlug": h.notebook,
        }
        if h.is_versioned():
            kernel_ref["VersionNumber"] = str(h.version)

        result = client.post(
            ATTACH_DATASOURCE_REQUEST_NAME,
            {
                "kernelRef": kernel_ref,
            },
            timeout=(DEFAULT_CONNECT_TIMEOUT, ATTACH_DATASOURCE_READ_TIMEOUT),
        )

        if "mountSlug" not in result:
            msg = "'result.mountSlug' field missing from response"
            raise BackendError(msg)

        base_mount_path = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME, DEFAULT_KAGGLE_CACHE_MOUNT_FOLDER)
        cached_path = f"{base_mount_path}/{result['mountSlug']}"

        if not os.path.exists(cached_path):
            # Only print this if the notebook output is not already mounted.
            logger.info(f"Mounting files to {cached_path}...")
        elif path:
            logger.info(
                f"Attaching '{path}' from Notebook '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
        else:
            logger.info(
                f"Attaching model '{h}' to your Kaggle notebook...",
                extra={**EXTRA_CONSOLE_BLOCK},
            )

        while not os.path.exists(cached_path):
            time.sleep(5)

        if path:
            cached_filepath = f"{cached_path}/{path}"
            if not os.path.exists(cached_filepath):
                msg = (
                    f"'{path}' is not present in the notebook output files."
                    f"You can access the other files of the attached notebook at '{cached_path}'"
                )
                raise ValueError(msg)
            return cached_filepath
        return cached_path
