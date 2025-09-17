import pathlib
from typing import Any

import yaml

from kagglehub.handle import (
    CompetitionHandle,
    DatasetHandle,
    ModelHandle,
    NotebookHandle,
    PackageHandle,
    ResourceHandle,
    UtilityScriptHandle,
    parse_competition_handle,
    parse_dataset_handle,
    parse_model_handle,
    parse_notebook_handle,
    parse_package_handle,
    parse_utility_script_handle,
)

# Current version of the file format written here
FORMAT_VERSION = "0.1.0"

FORMAT_VERSION_FIELD = "format_version"
DATASOURCES_FIELD = "datasources"
DATASOURCE_TYPE_FIELD = "type"
DATASOURCE_REF_FIELD = "ref"
DATASOURCE_VERSION_FIELD = "version"

HANDLE_TYPE_NAMES = {
    CompetitionHandle: "Competition",
    DatasetHandle: "Dataset",
    ModelHandle: "Model",
    NotebookHandle: "Notebook",
    UtilityScriptHandle: "UtilityScript",
    PackageHandle: "Package",
}

HANDLE_TYPE_PARSERS = {
    HANDLE_TYPE_NAMES[CompetitionHandle]: parse_competition_handle,
    HANDLE_TYPE_NAMES[DatasetHandle]: parse_dataset_handle,
    HANDLE_TYPE_NAMES[ModelHandle]: parse_model_handle,
    HANDLE_TYPE_NAMES[NotebookHandle]: parse_notebook_handle,
    HANDLE_TYPE_NAMES[UtilityScriptHandle]: parse_utility_script_handle,
    HANDLE_TYPE_NAMES[PackageHandle]: parse_package_handle,
}

# Maps requested ResourceHandle (which may include version) to version used
VersionedDatasources = dict[ResourceHandle, int | None]

# Tracks datasources accessed in the current session
_accessed_datasources: VersionedDatasources = {}


def register_datasource_access(handle: ResourceHandle, version: int | None) -> None:
    """Record that a datasource was accessed.

    Link the user-requested handle to the version retrieved."""
    _accessed_datasources[handle] = version


def get_accessed_datasources() -> VersionedDatasources:
    return _accessed_datasources.copy()


def write_file(filepath: str | pathlib.Path) -> None:
    """Write the datasources accessed during this session to a yaml file.

    Args:
        filepath: (str | pathlib.Path) Where to write the yaml file.
    """
    data = {
        FORMAT_VERSION_FIELD: FORMAT_VERSION,
        DATASOURCES_FIELD: [_serialize_datasource(h, version) for h, version in _accessed_datasources.items()],
    }

    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def read_file(filepath: str | pathlib.Path) -> VersionedDatasources:
    """Read a yaml file with datasource + version records.

    Args:
        filepath: (str | pathlib.Path) Path of yaml file to read from.
    Returns:
        Dictionary mapping ResourceHandle to version number, parsed from file.
    """
    with open(filepath) as f:
        data = yaml.safe_load(f)

    format_version = data.get(FORMAT_VERSION_FIELD)
    if format_version != FORMAT_VERSION:
        msg = f"Unsupported tracker file format version: {format_version}"
        raise ValueError(msg)

    versioned_datasources: VersionedDatasources = {}
    for datasource in data.get(DATASOURCES_FIELD, []):
        h, version = _deserialize_datasource(datasource)
        versioned_datasources[h] = version

    return versioned_datasources


def _serialize_datasource(h: ResourceHandle, version: int | None) -> dict:
    data: dict[str, Any] = {
        DATASOURCE_TYPE_FIELD: HANDLE_TYPE_NAMES[type(h)],
        DATASOURCE_REF_FIELD: str(h),
    }

    if version is not None:
        data[DATASOURCE_VERSION_FIELD] = version

    return data


def _deserialize_datasource(data: dict) -> tuple[ResourceHandle, int | None]:
    parser = HANDLE_TYPE_PARSERS[data[DATASOURCE_TYPE_FIELD]]
    h = parser(data[DATASOURCE_REF_FIELD])
    version = _parse_version(data.get(DATASOURCE_VERSION_FIELD, None))

    return h, version


def _parse_version(version: Any) -> int | None:  # noqa: ANN401
    if version is None or isinstance(version, int):
        return version

    if isinstance(version, str):
        try:
            return int(version)
        except:  # noqa: E722, S110
            # Fall through to the raise below
            pass

    msg = f"Invalid version: '{version}'. Expected an integer or None."
    raise ValueError(msg)
