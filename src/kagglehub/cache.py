import os
from typing import Optional, Union

from kagglehub.config import get_cache_folder
from kagglehub.handle import ModelHandle

MODELS_CACHE_SUBFOLDER = "models"


def load_from_cache(handle: Union[ModelHandle], path: Optional[str] = None) -> Optional[str]:
    """Return path for the requested resource from the cache.

    Args:
        handle: Resource handle
        path: Optional path to a file within the bundle.

    Returns:
        A string representing the path to the requested resource in the cache or None on cache miss.
    """
    full_path = get_cached_path(handle, path)
    return full_path if os.path.exists(full_path) else None


def get_cached_path(handle: Union[ModelHandle], path: Optional[str] = None) -> str:
    # Can extend to add support for other resources like DatasetHandle.
    if isinstance(handle, ModelHandle):
        return _get_model_path(handle, path)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def _get_model_path(handle: ModelHandle, path: Optional[str] = None):
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
