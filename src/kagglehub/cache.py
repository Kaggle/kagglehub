import os
import shutil
from pathlib import Path

from kagglehub.config import get_cache_folder
from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle, NotebookHandle, ResourceHandle

DATASETS_CACHE_SUBFOLDER = "datasets"
NOTEBOOKS_CACHE_SUBFOLDER = "notebooks"  # for resources under kaggle.com/code
COMPETITIONS_CACHE_SUBFOLDER = "competitions"
MODELS_CACHE_SUBFOLDER = "models"
FILE_COMPLETION_MARKER_FOLDER = ".complete"


class Cache:
    """Cache helper that optionally overrides the default cache directory."""

    def __init__(self, override_dir: str | None = None) -> None:
        self._override_dir = override_dir

    def get_path(self, handle: ResourceHandle, path: str | None = None) -> str:
        if self._override_dir:
            return os.path.join(self._override_dir, path) if path else self._override_dir
        return get_cached_path(handle, path)

    def get_archive_path(self, handle: ResourceHandle) -> str:
        if self._override_dir:
            return os.path.join(self._override_dir, _get_override_archive_name(handle))
        return get_cached_archive_path(handle)

    def _get_completion_marker_filepath(self, handle: ResourceHandle, path: str | None = None) -> str:
        if not self._override_dir:
            return _get_completion_marker_filepath(handle, path)

        marker_base = os.path.join(
            self._override_dir,
            FILE_COMPLETION_MARKER_FOLDER,
            _get_override_marker_base(handle),
        )
        if path:
            safe_path = path.lstrip(os.path.sep)
            return os.path.join(marker_base, f"{safe_path}.complete")
        return os.path.join(marker_base, "bundle.complete")

    def load_from_cache(self, handle: ResourceHandle, path: str | None = None) -> str | None:
        """Return path for the requested resource from the cache or output_dir."""
        marker_path = self._get_completion_marker_filepath(handle, path)
        full_path = self.get_path(handle, path)
        return full_path if os.path.exists(marker_path) and os.path.exists(full_path) else None

    def mark_as_complete(self, handle: ResourceHandle, path: str | None = None) -> None:
        marker_path = self._get_completion_marker_filepath(handle, path)
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        Path(marker_path).touch()

    def mark_as_incomplete(self, handle: ResourceHandle, path: str | None = None) -> None:
        marker_path = self._get_completion_marker_filepath(handle, path)
        self._delete_path(marker_path)

    def delete_from_cache(self, handle: ResourceHandle, path: str | None = None) -> str | None:
        """Delete resource from the cache, even if incomplete."""
        self.mark_as_incomplete(handle, path)
        full_path = self.get_path(handle, path)
        return self._delete_path(full_path)

    def _delete_path(self, path: str) -> str | None:
        if not os.path.exists(path):
            return None
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

        if self._override_dir:
            return path

        # Remove empty folders in the given path, up until the cache folder.
        # Avoid using removedirs() because it may remove parents of the cache folder.
        curr_dir = os.path.dirname(path)
        while len(os.listdir(curr_dir)) == 0 and curr_dir != get_cache_folder():
            parent_dir = os.path.dirname(curr_dir)
            os.rmdir(curr_dir)
            curr_dir = parent_dir
        return path


def get_cached_path(handle: ResourceHandle, path: str | None = None) -> str:
    # Can extend to add support for other resources like DatasetHandle.
    if isinstance(handle, ModelHandle):
        return _get_model_path(handle, path)
    elif isinstance(handle, DatasetHandle):
        return _get_dataset_path(handle, path)
    elif isinstance(handle, CompetitionHandle):
        return _get_competition_path(handle, path)
    elif isinstance(handle, NotebookHandle):
        return _get_notebook_output_path(handle, path)
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
    elif isinstance(handle, NotebookHandle):
        return _get_notebook_output_archive_path(handle)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def _delete_from_cache_folder(path: str) -> str | None:
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


def mark_as_incomplete(handle: ResourceHandle, path: str | None = None) -> None:
    marker_path = _get_completion_marker_filepath(handle, path)
    _delete_from_cache_folder(marker_path)


def delete_from_cache(handle: ResourceHandle, path: str | None = None) -> str | None:
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


def _get_completion_marker_filepath(handle: ResourceHandle, path: str | None = None) -> str:
    if isinstance(handle, ModelHandle):
        return _get_models_completion_marker_filepath(handle, path)
    elif isinstance(handle, DatasetHandle):
        return _get_datasets_completion_marker_filepath(handle, path)
    elif isinstance(handle, CompetitionHandle):
        return _get_competitions_completion_marker_filepath(handle, path)
    elif isinstance(handle, NotebookHandle):
        return _get_notebook_output_completion_marker_filepath(handle, path)
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def _get_dataset_path(handle: DatasetHandle, path: str | None = None) -> str:
    base_path = os.path.join(get_cache_folder(), DATASETS_CACHE_SUBFOLDER, handle.owner, handle.dataset)
    if handle.is_versioned():
        base_path = os.path.join(base_path, "versions", str(handle.version))

    return os.path.join(base_path, path) if path else base_path


def _get_notebook_output_path(handle: NotebookHandle, path: str | None = None) -> str:
    base_path = os.path.join(get_cache_folder(), NOTEBOOKS_CACHE_SUBFOLDER, handle.owner, handle.notebook, "output")
    if handle.is_versioned():
        base_path = os.path.join(base_path, "versions", str(handle.version))

    return os.path.join(base_path, path) if path else base_path


def _get_competition_path(handle: CompetitionHandle, path: str | None = None) -> str:
    base_path = os.path.join(get_cache_folder(), COMPETITIONS_CACHE_SUBFOLDER, handle.competition)
    return os.path.join(base_path, path) if path else base_path


def _get_model_path(handle: ModelHandle, path: str | None = None) -> str:
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


def _get_notebook_output_archive_path(handle: NotebookHandle) -> str:
    return os.path.join(
        get_cache_folder(),
        NOTEBOOKS_CACHE_SUBFOLDER,
        handle.owner,
        handle.notebook,
        "output-{handle.version!s}.archive",
    )


def _get_models_completion_marker_filepath(handle: ModelHandle, path: str | None = None) -> str:
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


def _get_datasets_completion_marker_filepath(handle: DatasetHandle, path: str | None = None) -> str:
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


def _get_notebook_output_completion_marker_filepath(handle: NotebookHandle, path: str | None = None) -> str:
    if path:
        return os.path.join(
            get_cache_folder(),
            NOTEBOOKS_CACHE_SUBFOLDER,
            handle.owner,
            handle.notebook,
            FILE_COMPLETION_MARKER_FOLDER,
            f"output-{handle.version!s}",
            f"{path}.complete",
        )

    return os.path.join(
        get_cache_folder(),
        NOTEBOOKS_CACHE_SUBFOLDER,
        handle.owner,
        handle.notebook,
        "output-{handle.version!s}.complete",
    )


def _get_competitions_completion_marker_filepath(handle: CompetitionHandle, path: str | None = None) -> str:
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


def _get_override_marker_base(handle: ResourceHandle) -> str:
    if isinstance(handle, ModelHandle):
        version = handle.version if handle.is_versioned() else "unknown"
        return os.path.join(
            "models",
            handle.owner,
            handle.model,
            handle.framework,
            handle.variation,
            str(version),
        )
    elif isinstance(handle, DatasetHandle):
        version = handle.version if handle.is_versioned() else "unknown"
        return os.path.join("datasets", handle.owner, handle.dataset, str(version))
    elif isinstance(handle, CompetitionHandle):
        return os.path.join("competitions", handle.competition)
    elif isinstance(handle, NotebookHandle):
        version = handle.version if handle.is_versioned() else "unknown"
        return os.path.join("notebooks", handle.owner, handle.notebook, str(version))
    else:
        msg = "Invalid handle"
        raise ValueError(msg)


def _get_override_archive_name(handle: ResourceHandle) -> str:
    if isinstance(handle, ModelHandle):
        version = handle.version if handle.is_versioned() else "unknown"
        return f"{version!s}.archive"
    elif isinstance(handle, DatasetHandle):
        version = handle.version if handle.is_versioned() else "unknown"
        return f"{version!s}.archive"
    elif isinstance(handle, NotebookHandle):
        version = handle.version if handle.is_versioned() else "unknown"
        return f"output-{version!s}.archive"
    elif isinstance(handle, CompetitionHandle):
        return f"{handle.competition}.archive"
    else:
        msg = "Invalid handle"
        raise ValueError(msg)
