import os
from pathlib import Path
from typing import Optional, Union

from kagglehub.config import get_cache_folder
from kagglehub.handle import ModelHandle

MODELS_CACHE_SUBFOLDER = "models"
MODELS_FILE_COMPLETION_MARKER_FOLDER = ".complete"


def load_from_cache(handle: Union[ModelHandle], path: Optional[str] = None) -> Optional[str]:
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


def get_cached_path(handle: Union[ModelHandle], path: Optional[str] = None) -> str:
    # Can extend to add support for other resources like DatasetHandle.
    if isinstance(handle, ModelHandle):
        return _get_model_path(handle, path)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def get_cached_archive_path(handle: Union[ModelHandle]) -> str:
    if isinstance(handle, ModelHandle):
        return _get_model_archive_path(handle)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def mark_as_complete(handle: Union[ModelHandle], path: Optional[str] = None):
    marker_path = _get_completion_marker_filepath(handle, path)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    Path(marker_path).touch()


def _get_completion_marker_filepath(handle: Union[ModelHandle], path: Optional[str] = None) -> str:
    # Can extend to add support for other resources like DatasetHandle.
    if isinstance(handle, ModelHandle):
        return _get_models_completion_marker_filepath(handle, path)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


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


def _get_models_completion_marker_filepath(handle: ModelHandle, path: Optional[str] = None) -> str:
    if path:
        return os.path.join(
            get_cache_folder(),
            MODELS_CACHE_SUBFOLDER,
            handle.owner,
            handle.model,
            handle.framework,
            handle.variation,
            MODELS_FILE_COMPLETION_MARKER_FOLDER,
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
