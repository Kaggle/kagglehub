import os
import shutil
from pathlib import Path
from typing import Optional

from kagglehub.config import get_cache_folder
from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle, ResourceHandle

DATASETS_CACHE_SUBFOLDER = "datasets"
COMPETITIONS_CACHE_SUBFOLDER = "competitions"
MODELS_CACHE_SUBFOLDER = "models"
FILE_COMPLETION_MARKER_FOLDER = ".complete"


def load_from_cache(handle: ResourceHandle, path: Optional[str] = None) -> Optional[str]:
    """Return path for the requested resource from the cache.

    Args:
        handle: Resource handle
        path: Optional path to a file within the bundle.

    Returns:
        A string representing the path to the requested resource in the cache or None on cache miss.
    """
    marker_path = _get_completion_marker_filepath(handle, path)
    full_path = get_cached_path(handle, path)
    return full_path if os.path.exists(marker_path) and os.path.exists(full_path) else None


def get_cached_path(handle: ResourceHandle, path: Optional[str] = None) -> str:
    # Can extend to add support for other resources like DatasetHandle.
    if isinstance(handle, ModelHandle):
        return _get_model_path(handle, path)
    elif isinstance(handle, DatasetHandle):
        return _get_dataset_path(handle, path)
    elif isinstance(handle, CompetitionHandle):
        return _get_competition_path(handle, path)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def get_cached_archive_path(handle: ResourceHandle) -> str:
    if isinstance(handle, ModelHandle):
        return _get_model_archive_path(handle)
    elif isinstance(handle, DatasetHandle):
        return _get_dataset_archive_path(handle)
    elif isinstance(handle, CompetitionHandle):
        return _get_competition_archive_path(handle)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def mark_as_complete(handle: ResourceHandle, path: Optional[str] = None) -> None:
    marker_path = _get_completion_marker_filepath(handle, path)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    Path(marker_path).touch()


def _delete_from_cache_folder(path: str) -> Optional[str]:
    # Delete resource(s) at the given path, whether file or directory, from the cache folder.
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

        # Remove empty folders in the given path, up until the cache folder.
        # Avoid using removedirs() because it may remove parents of the cache folder.
        curr_dir = os.path.dirname(path)
        while len(os.listdir(curr_dir)) == 0 and curr_dir != get_cache_folder():
            parent_dir = os.path.dirname(curr_dir)
            os.rmdir(curr_dir)
            curr_dir = parent_dir
        return path
    return None


def mark_as_incomplete(handle: ResourceHandle, path: Optional[str] = None) -> None:
    marker_path = _get_completion_marker_filepath(handle, path)
    _delete_from_cache_folder(marker_path)


def delete_from_cache(handle: ResourceHandle, path: Optional[str] = None) -> Optional[str]:
    """Delete resource from the cache, even if incomplete.

    Args:
        handle: Resource handle
        path: Optional path to a file within the bundle.

    Returns:
        A string representing the path of the deleted resource or None on cache miss.
    """
    mark_as_incomplete(handle, path)
    model_full_path = get_cached_path(handle, path)
    return _delete_from_cache_folder(model_full_path)


def _get_completion_marker_filepath(handle: ResourceHandle, path: Optional[str] = None) -> str:
    # Can extend to add support for other resources like DatasetHandle.
    if isinstance(handle, ModelHandle):
        return _get_models_completion_marker_filepath(handle, path)
    elif isinstance(handle, DatasetHandle):
        return _get_datasets_completion_marker_filepath(handle, path)
    elif isinstance(handle, CompetitionHandle):
        return _get_competitions_completion_marker_filepath(handle, path)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def _get_dataset_path(handle: DatasetHandle, path: Optional[str] = None) -> str:
    base_path = os.path.join(get_cache_folder(), DATASETS_CACHE_SUBFOLDER, handle.owner, handle.dataset)
    if handle.is_versioned():
        base_path = os.path.join(base_path, "versions", str(handle.version))

    return os.path.join(base_path, path) if path else base_path


def _get_competition_path(handle: CompetitionHandle, path: Optional[str] = None) -> str:
    base_path = os.path.join(get_cache_folder(), COMPETITIONS_CACHE_SUBFOLDER, handle.competition)
    return os.path.join(base_path, path) if path else base_path


def _get_model_path(handle: ModelHandle, path: Optional[str] = None) -> str:
    base_path = os.path.join(
        get_cache_folder(),
        MODELS_CACHE_SUBFOLDER,
        handle.owner,
        handle.model,
        handle.framework,
        handle.variation,
        str(handle.version),
    )

    return os.path.join(base_path, path) if path else base_path


def _get_model_archive_path(handle: ModelHandle) -> str:
    return os.path.join(
        get_cache_folder(),
        MODELS_CACHE_SUBFOLDER,
        handle.owner,
        handle.model,
        handle.framework,
        handle.variation,
        f"{handle.version!s}.archive",
    )


def _get_dataset_archive_path(handle: DatasetHandle) -> str:
    return os.path.join(
        get_cache_folder(),
        DATASETS_CACHE_SUBFOLDER,
        handle.owner,
        handle.dataset,
        f"{handle.version!s}.archive",
    )


def _get_competition_archive_path(handle: CompetitionHandle) -> str:
    return os.path.join(
        get_cache_folder(),
        COMPETITIONS_CACHE_SUBFOLDER,
        f"{handle.competition}.archive",
    )


def _get_models_completion_marker_filepath(handle: ModelHandle, path: Optional[str] = None) -> str:
    if path:
        return os.path.join(
            get_cache_folder(),
            MODELS_CACHE_SUBFOLDER,
            handle.owner,
            handle.model,
            handle.framework,
            handle.variation,
            FILE_COMPLETION_MARKER_FOLDER,
            str(handle.version),
            f"{path}.complete",
        )

    return os.path.join(
        get_cache_folder(),
        MODELS_CACHE_SUBFOLDER,
        handle.owner,
        handle.model,
        handle.framework,
        handle.variation,
        f"{handle.version!s}.complete",
    )


def _get_datasets_completion_marker_filepath(handle: DatasetHandle, path: Optional[str] = None) -> str:
    if path:
        return os.path.join(
            get_cache_folder(),
            DATASETS_CACHE_SUBFOLDER,
            handle.owner,
            handle.dataset,
            FILE_COMPLETION_MARKER_FOLDER,
            str(handle.version),
            f"{path}.complete",
        )

    return os.path.join(
        get_cache_folder(),
        DATASETS_CACHE_SUBFOLDER,
        handle.owner,
        handle.dataset,
        f"{handle.version!s}.complete",
    )


def _get_competitions_completion_marker_filepath(handle: CompetitionHandle, path: Optional[str] = None) -> str:
    if path:
        return os.path.join(
            get_cache_folder(),
            COMPETITIONS_CACHE_SUBFOLDER,
            FILE_COMPLETION_MARKER_FOLDER,
            f"{handle.competition}",
            f"{path}.complete",
        )

    return os.path.join(
        get_cache_folder(),
        COMPETITIONS_CACHE_SUBFOLDER,
        f"{handle.competition}.complete",
    )
