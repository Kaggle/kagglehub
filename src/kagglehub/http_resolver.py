import logging
import os
import tarfile
import zipfile
from typing import List, Optional, Tuple

from tqdm.contrib.concurrent import thread_map

from kagglehub.cache import (
    delete_from_cache,
    get_cached_archive_path,
    get_cached_path,
    load_from_cache,
    mark_as_complete,
)
from kagglehub.clients import KaggleApiV1Client
from kagglehub.handle import DatasetHandle, ModelHandle, ResourceHandle
from kagglehub.resolver import Resolver

DATASET_CURRENT_VERSION_FIELD = "currentVersionNumber"

MODEL_INSTANCE_VERSION_FIELD = "versionNumber"
MAX_NUM_FILES_DIRECT_DOWNLOAD = 25

logger = logging.getLogger(__name__)


class DatasetHttpResolver(Resolver[DatasetHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        # Downloading files over HTTP is supported in all environments for all handles / paths.
        return True

    def __call__(self, h: DatasetHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
        api_client = KaggleApiV1Client()

        if not h.is_versioned():
            h.version = _get_current_version(api_client, h)

        dataset_path = load_from_cache(h, path)
        if dataset_path and not force_download:
            return dataset_path  # Already cached
        elif dataset_path and force_download:
            delete_from_cache(h, path)

        url_path = _build_dataset_download_url_path(h)
        out_path = get_cached_path(h, path)

        # Create the intermediary directories
        if path:
            # Downloading a single file.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            api_client.download_file(url_path + "&file_name=" + path, out_path, h)
        else:
            # TODO(b/345800027) Implement parallel download when < 25 files in databundle.
            # Downloading the full archived bundle.
            archive_path = get_cached_archive_path(h)
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)

            # First, we download the archive.
            api_client.download_file(url_path, archive_path, h)

            # Create the directory to extract the archive to.
            os.makedirs(out_path, exist_ok=True)

            _extract_archive(archive_path, out_path)

            # Delete the archive
            os.remove(archive_path)

        mark_as_complete(h, path)
        return out_path


class ModelHttpResolver(Resolver[ModelHandle]):
    def is_supported(self, *_, **__) -> bool:  # noqa: ANN002, ANN003
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def __call__(self, h: ModelHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
        api_client = KaggleApiV1Client()

        if not h.is_versioned():
            h.version = _get_current_version(api_client, h)

        model_path = load_from_cache(h, path)
        if model_path and not force_download:
            return model_path  # Already cached
        elif model_path and force_download:
            delete_from_cache(h, path)

        url_path = _build_download_url_path(h)
        out_path = get_cached_path(h, path)

        # Create the intermediary directories
        if path:
            # Downloading a single file.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            api_client.download_file(url_path + "/" + path, out_path, h)
        else:
            # List the files and decide how to download them:
            # - <= 25 files: Download files in parallel
            # > 25 files: Download the archive and uncompress
            (files, has_more) = _list_files(api_client, h)
            if has_more:
                # Downloading the full archived bundle.
                archive_path = get_cached_archive_path(h)
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)

                # First, we download the archive.
                api_client.download_file(url_path, archive_path, h)

                # Create the directory to extract the archive to.
                os.makedirs(out_path, exist_ok=True)

                _extract_archive(archive_path, out_path)

                # Delete the archive
                os.remove(archive_path)
            else:
                # Download files individually in parallel
                def _inner_download_file(file: str) -> None:
                    file_out_path = out_path + "/" + file
                    os.makedirs(os.path.dirname(file_out_path), exist_ok=True)
                    api_client.download_file(url_path + "/" + file, file_out_path, h)

                thread_map(
                    _inner_download_file,
                    files,
                    desc=f"Downloading {len(files)} files",
                    max_workers=8,  # Never use more than 8 threads in parallel to download files.
                )

        mark_as_complete(h, path)
        return out_path


def _extract_archive(archive_path: str, out_path: str) -> None:
    logger.info("Extracting model files...")
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as f:
            f.extractall(out_path)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as f:
            f.extractall(out_path)
    else:
        msg = "Unsupported archive type."
        raise ValueError(msg)


def _get_current_version(api_client: KaggleApiV1Client, h: ResourceHandle) -> int:
    if isinstance(h, ModelHandle):
        json_response = api_client.get(_build_get_instance_url_path(h), h)
        if MODEL_INSTANCE_VERSION_FIELD not in json_response:
            msg = f"Invalid GetModelInstance API response. Expected to include a {MODEL_INSTANCE_VERSION_FIELD} field"
            raise ValueError(msg)

        return json_response[MODEL_INSTANCE_VERSION_FIELD]

    elif isinstance(h, DatasetHandle):
        json_response = api_client.get(_build_get_dataset_url_path(h), h)
        if DATASET_CURRENT_VERSION_FIELD not in json_response:
            msg = f"Invalid GetDataset API response. Expected to include a {DATASET_CURRENT_VERSION_FIELD} field"
            raise ValueError(msg)

        return json_response[DATASET_CURRENT_VERSION_FIELD]

    else:
        msg = f"Invalid ResourceHandle type {h}"
        raise ValueError(msg)


def _list_files(api_client: KaggleApiV1Client, h: ModelHandle) -> Tuple[List[str], bool]:
    json_response = api_client.get(_build_list_model_instance_version_files_url_path(h), h)
    if "files" not in json_response:
        msg = "Invalid ListModelInstanceVersionFiles API response. Expected to include a 'files' field"
        raise ValueError(msg)

    files = []
    for f in json_response["files"]:
        files.append(f["name"])

    has_more = "nextPageToken" in json_response and json_response["nextPageToken"] != ""

    return (files, has_more)


def _build_get_instance_url_path(h: ModelHandle) -> str:
    return f"models/{h.owner}/{h.model}/{h.framework}/{h.variation}/get"


def _build_download_url_path(h: ModelHandle) -> str:
    return f"models/{h.owner}/{h.model}/{h.framework}/{h.variation}/{h.version}/download"


def _build_list_model_instance_version_files_url_path(h: ModelHandle) -> str:
    return f"models/{h.owner}/{h.model}/{h.framework}/{h.variation}/{h.version}/files\
?page_size={MAX_NUM_FILES_DIRECT_DOWNLOAD}"


def _build_get_dataset_url_path(h: DatasetHandle) -> str:
    return f"datasets/view/{h.owner}/{h.dataset}"


def _build_dataset_download_url_path(h: DatasetHandle) -> str:
    return f"datasets/download/{h.owner}/{h.dataset}?dataset_version_number={h.version}"
