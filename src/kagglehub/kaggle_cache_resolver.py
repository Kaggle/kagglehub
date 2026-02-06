import logging
import os
import time

from kagglesdk.kaggle_env import is_in_kaggle_notebook

from kagglehub.clients import (
    DEFAULT_CONNECT_TIMEOUT,
    KaggleJwtClient,
)
from kagglehub.config import is_kaggle_cache_disabled
from kagglehub.exceptions import BackendError
from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle, NotebookHandle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.packages import PackageScope
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

    def _resolve(
        self,
        h: CompetitionHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        if output_dir:
            logger.info(
                "Ignoring `output_dir` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
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
            return cached_filepath, None
        return cached_path, None


class DatasetKaggleCacheResolver(Resolver[DatasetHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if is_kaggle_cache_disabled():
            return False

        if is_in_kaggle_notebook():
            return True

        return False

    def _resolve(
        self,
        h: DatasetHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        if output_dir:
            logger.info(
                "Ignoring `output_dir` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
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
        else:
            # Check if there's a Package in scope which has stored a version number used when it was created.
            version_from_package_scope = PackageScope.get_version(h)
            if version_from_package_scope is not None:
                dataset_ref["VersionNumber"] = str(version_from_package_scope)

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
        version = result.get("versionNumber")  # None if missing

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
            return cached_filepath, version
        return cached_path, version


class ModelKaggleCacheResolver(Resolver[ModelHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if is_kaggle_cache_disabled():
            return False

        if is_in_kaggle_notebook():
            return True

        return False

    def _resolve(
        self,
        h: ModelHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        if output_dir:
            logger.info(
                "Ignoring `output_dir` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
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
        else:
            # Check if there's a Package in scope which has stored a version number used when it was created.
            version_from_package_scope = PackageScope.get_version(h)
            if version_from_package_scope is not None:
                model_ref["VersionNumber"] = str(version_from_package_scope)

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
        version = result.get("versionNumber")  # None if missing

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
            return cached_filepath, version
        return cached_path, version


class NotebookOutputKaggleCacheResolver(Resolver[NotebookHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        if is_kaggle_cache_disabled():
            return False

        if is_in_kaggle_notebook():
            return True

        return False

    def _resolve(
        self,
        h: NotebookHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        if output_dir:
            logger.info(
                "Ignoring `output_dir` argument when running inside the Kaggle notebook environment.",
                extra={**EXTRA_CONSOLE_BLOCK},
            )
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
        else:
            # Check if there's a Package in scope which has stored a version number used when it was created.
            version_from_package_scope = PackageScope.get_version(h)
            if version_from_package_scope is not None:
                kernel_ref["VersionNumber"] = str(version_from_package_scope)

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
        version = result.get("versionNumber")  # None if missing

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
            return cached_filepath, version
        return cached_path, version
