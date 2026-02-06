import logging
import os
import shutil
import tarfile
import zipfile

import requests
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFileRequest, ApiDownloadDataFilesRequest
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest, ApiGetDatasetRequest
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest, ApiGetKernelRequest
from kagglesdk.models.types.model_api_service import (
    ApiDownloadModelInstanceVersionRequest,
    ApiGetModelInstanceRequest,
    ApiListModelInstanceVersionFilesRequest,
)
from tqdm.contrib.concurrent import thread_map

from kagglehub.cache import Cache
from kagglehub.clients import build_kaggle_client, download_file
from kagglehub.config import get_kaggle_credentials
from kagglehub.exceptions import UnauthenticatedError, handle_call
from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle, NotebookHandle, ResourceHandle
from kagglehub.packages import PackageScope
from kagglehub.resolver import Resolver

MAX_NUM_FILES_DIRECT_DOWNLOAD = 25

logger = logging.getLogger(__name__)


class CompetitionHttpResolver(Resolver[CompetitionHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        # Downloading files over HTTP is supported in all environments for all handles / paths.
        return True

    def _resolve(
        self,
        h: CompetitionHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        with build_kaggle_client() as api_client:
            cache = Cache(override_dir=output_dir)
            cached_path = cache.load_from_cache(h, path)
            if cached_path and force_download:
                cache.delete_from_cache(h, path)
                cached_path = None

            if not get_kaggle_credentials():
                if cached_path:
                    return cached_path, None
                raise UnauthenticatedError()

            out_path = cache.get_path(h, path)

            if output_dir:
                _prepare_output_dir(output_dir, path, force_download=bool(force_download))
            if path:
                # For single file downloads.
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                try:
                    r = _build_competition_download_file_request(h, path)
                    response = handle_call(
                        lambda: api_client.competitions.competition_api_client.download_data_file(r), h
                    )
                    download_needed = download_file(
                        response, out_path, h, cached_path, extract_auto_compressed_file=True
                    )
                except requests.exceptions.ConnectionError:
                    if cached_path:
                        return cached_path, None
                    raise

                if not download_needed and cached_path:
                    return cached_path, None
            else:
                # Download, extract, then delete the archive.
                r = _build_competition_download_files_request(h)
                archive_path = cache.get_archive_path(h)
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)

                try:
                    response = handle_call(
                        lambda: api_client.competitions.competition_api_client.download_data_files(r), h
                    )
                    download_needed = download_file(response, archive_path, h, cached_path)
                except requests.exceptions.ConnectionError:
                    if cached_path:
                        if os.path.exists(archive_path):
                            os.remove(archive_path)
                        return cached_path, None
                    raise

                if not download_needed and cached_path:
                    if os.path.exists(archive_path):
                        os.remove(archive_path)
                    return cached_path, None

                _extract_archive(archive_path, out_path)
                os.remove(archive_path)

            cache.mark_as_complete(h, path)
            return out_path, None


class DatasetHttpResolver(Resolver[DatasetHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        # Downloading files over HTTP is supported in all environments for all handles / paths.
        return True

    def _resolve(
        self,
        h: DatasetHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        with build_kaggle_client() as api_client:
            if not h.is_versioned():
                h = h.with_version(_get_current_version(api_client, h))

            cache = Cache(override_dir=output_dir)
            dataset_path = cache.load_from_cache(h, path)
            if dataset_path and not force_download:
                return dataset_path, h.version  # Already cached
            elif dataset_path and force_download:
                cache.delete_from_cache(h, path)

            if output_dir:
                _prepare_output_dir(output_dir, path, force_download=bool(force_download))

            r = _build_dataset_download_request(h, path)
            out_path = cache.get_path(h, path)

            # Create the intermediary directories
            if path:
                # Downloading a single file.
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                response = handle_call(lambda: api_client.datasets.dataset_api_client.download_dataset(r), h)
                download_file(response, out_path, h, extract_auto_compressed_file=True)
            else:
                # TODO(b/345800027) Implement parallel download when < 25 files in databundle.
                # Downloading the full archived bundle.
                archive_path = cache.get_archive_path(h)
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)

                # First, we download the archive.
                response = handle_call(lambda: api_client.datasets.dataset_api_client.download_dataset(r), h)
                download_file(response, archive_path, h)

                _extract_archive(archive_path, out_path)

                # Delete the archive
                os.remove(archive_path)

            cache.mark_as_complete(h, path)
            return out_path, h.version


class ModelHttpResolver(Resolver[ModelHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def _resolve(
        self,
        h: ModelHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        with build_kaggle_client() as api_client:
            if not h.is_versioned():
                h = h.with_version(_get_current_version(api_client, h))

            cache = Cache(override_dir=output_dir)
            model_path = cache.load_from_cache(h, path)
            if model_path and not force_download:
                return model_path, h.version  # Already cached
            if output_dir:
                _prepare_output_dir(output_dir, path, force_download=bool(force_download))
            elif model_path and force_download:
                cache.delete_from_cache(h, path)

            r = _build_model_download_request(h, path)
            out_path = cache.get_path(h, path)

            # Create the intermediary directories
            if path:
                # Downloading a single file.
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                response = handle_call(lambda: api_client.models.model_api_client.download_model_instance_version(r), h)
                download_file(response, out_path, h, extract_auto_compressed_file=True)
            else:
                # List the files and decide how to download them:
                # - <= 25 files: Download files in parallel
                # > 25 files: Download the archive and uncompress
                files, has_more = _list_model_files(api_client, h)
                if has_more:
                    # Downloading the full archived bundle.
                    archive_path = cache.get_archive_path(h)
                    os.makedirs(os.path.dirname(archive_path), exist_ok=True)

                    # First, we download the archive.
                    response = handle_call(
                        lambda: api_client.models.model_api_client.download_model_instance_version(r), h
                    )
                    download_file(response, archive_path, h)

                    _extract_archive(archive_path, out_path)

                    # Delete the archive
                    os.remove(archive_path)
                else:
                    # Download files individually in parallel
                    def _inner_download_file(file: str) -> None:
                        file_out_path = os.path.join(out_path, file)
                        os.makedirs(os.path.dirname(file_out_path), exist_ok=True)
                        r = _build_model_download_request(h, file)
                        response = handle_call(
                            lambda: api_client.models.model_api_client.download_model_instance_version(r), h
                        )
                        download_file(response, file_out_path, h)

                    thread_map(
                        _inner_download_file,
                        files,
                        desc=f"Downloading {len(files)} files",
                        max_workers=8,  # Never use more than 8 threads in parallel to download files.
                    )

            cache.mark_as_complete(h, path)
            return out_path, h.version


class NotebookOutputHttpResolver(Resolver[NotebookHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        # Downloading files over HTTP is supported in all environments for all handles / paths.
        return True

    def _resolve(
        self,
        h: NotebookHandle,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        with build_kaggle_client() as api_client:
            if not h.is_versioned():
                h = h.with_version(_get_current_version(api_client, h))
            cache = Cache(override_dir=output_dir)
            notebook_path = cache.load_from_cache(h, path)
            if notebook_path and not force_download:
                return notebook_path, h.version  # Already cached
            if output_dir:
                _prepare_output_dir(output_dir, path, force_download=bool(force_download))
            elif notebook_path and force_download:
                cache.delete_from_cache(h, path)

            r = _build_notebook_download_request(h, path)
            out_path = cache.get_path(h, path)

            if path:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                response = handle_call(lambda: api_client.kernels.kernels_api_client.download_kernel_output(r), h)
                download_file(response, out_path, h, extract_auto_compressed_file=True)
            else:
                # TODO(b/345800027) Implement parallel download when < 25 files in databundle.
                # Downloading the full archived bundle.
                archive_path = cache.get_archive_path(h)
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)

                # First, we download the archive.
                response = handle_call(lambda: api_client.kernels.kernels_api_client.download_kernel_output(r), h)
                download_file(response, archive_path, h)

                _extract_archive(archive_path, out_path)

                # Delete the archive
                os.remove(archive_path)

            cache.mark_as_complete(h, path)

            return out_path, h.version


def _extract_archive(archive_path: str, out_path: str) -> None:
    # Create the directory to extract the archive to.
    os.makedirs(out_path, exist_ok=True)

    logger.info("Extracting files...")
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as f:
            f.extractall(out_path)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as f:
            f.extractall(out_path)
    else:
        msg = "Unsupported archive type."
        raise ValueError(msg)


def _prepare_output_dir(output_dir: str, path: str | None, *, force_download: bool) -> None:
    if path:
        target_path = os.path.join(output_dir, path)
        if os.path.exists(target_path):
            # This happens when a file is present at output_dir / path but the completion marker isn't set.
            if not force_download:
                msg = f"File already exists at output_dir: {target_path}. Set force_download=True to replace it."
                raise FileExistsError(msg)
            os.remove(target_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        return

    if os.path.exists(output_dir):
        if os.path.isfile(output_dir):
            msg = f"output_dir points to a file: {output_dir}"
            raise FileExistsError(msg)
        if os.listdir(output_dir):
            # This happens when a output_dir has files but the completion marker isn't set.
            if not force_download:
                msg = f"output_dir is not empty: {output_dir}. Set force_download=True to replace it."
                raise FileExistsError(msg)
            _clear_directory(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)


def _clear_directory(directory: str) -> None:
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)


def _get_current_version(api_client: KaggleClient, h: ResourceHandle) -> int:
    # Check if there's a Package in scope which has stored a version number used when it was created.
    version_from_package_scope = PackageScope.get_version(h)
    if version_from_package_scope is not None:
        return version_from_package_scope

    if isinstance(h, ModelHandle):
        r = ApiGetModelInstanceRequest()
        r.owner_slug = h.owner
        r.model_slug = h.model
        r.framework = h.framework_enum()
        r.instance_slug = h.variation
        model_instance = handle_call(lambda: api_client.models.model_api_client.get_model_instance(r))
        return model_instance.version_number

    elif isinstance(h, DatasetHandle):
        r = ApiGetDatasetRequest()
        r.owner_slug = h.owner
        r.dataset_slug = h.dataset
        dataset = handle_call(lambda: api_client.datasets.dataset_api_client.get_dataset(r))
        return dataset.current_version_number

    elif isinstance(h, NotebookHandle):
        r = ApiGetKernelRequest()
        r.user_name = h.owner
        r.kernel_slug = h.notebook
        response = handle_call(lambda: api_client.kernels.kernels_api_client.get_kernel(r))

        return response.metadata.current_version_number

    else:
        msg = f"Invalid ResourceHandle type {h}"
        raise ValueError(msg)


def _list_model_files(api_client: KaggleClient, h: ModelHandle) -> tuple[list[str], bool]:
    r = _build_list_model_instance_version_files_request(h)
    response = handle_call(lambda: api_client.models.model_api_client.list_model_instance_version_files(r))

    files = []
    for f in response.files:
        files.append(f.name)

    has_more = response.next_page_token != ""
    return (files, has_more)


def _build_model_download_request(h: ModelHandle, path: str | None) -> str:
    if not h.is_versioned():
        msg = "No version provided"
        raise ValueError(msg)

    r = ApiDownloadModelInstanceVersionRequest()
    r.owner_slug = h.owner
    r.model_slug = h.model
    r.framework = h.framework_enum()
    r.instance_slug = h.variation
    r.version_number = h.version
    if path:
        r.path = path

    return r


def _build_list_model_instance_version_files_request(h: ModelHandle) -> ApiListModelInstanceVersionFilesRequest:
    if not h.is_versioned():
        msg = "No version provided"
        raise ValueError(msg)

    r = ApiListModelInstanceVersionFilesRequest()
    r.owner_slug = h.owner
    r.model_slug = h.model
    r.framework = h.framework_enum()
    r.instance_slug = h.variation
    r.version_number = h.version
    r.page_size = MAX_NUM_FILES_DIRECT_DOWNLOAD
    return r


def _build_dataset_download_request(h: DatasetHandle, path: str | None) -> ApiDownloadDatasetRequest:
    if not h.is_versioned():
        msg = "No version provided"
        raise ValueError(msg)

    r = ApiDownloadDatasetRequest()
    r.owner_slug = h.owner
    r.dataset_slug = h.dataset
    r.dataset_version_number = h.version
    if path:
        r.file_name = path
    return r


def _build_notebook_download_request(h: NotebookHandle, path: str | None) -> ApiDownloadKernelOutputRequest:
    if not h.is_versioned():
        msg = "No version provided"
        raise ValueError(msg)

    r = ApiDownloadKernelOutputRequest()
    r.owner_slug = h.owner
    r.kernel_slug = h.notebook
    r.version_number = h.version
    if path:
        r.file_path = path

    return r


def _build_competition_download_files_request(h: CompetitionHandle) -> ApiDownloadDataFilesRequest:
    r = ApiDownloadDataFilesRequest()
    r.competition_name = h.competition
    return r


def _build_competition_download_file_request(h: CompetitionHandle, file: str) -> ApiDownloadDataFileRequest:
    r = ApiDownloadDataFileRequest()
    r.competition_name = h.competition
    r.file_name = file
    return r
