import pathlib
from typing import Any, Optional, Union

import yaml

from kagglehub.handle import (
    CompetitionHandle,
    DatasetHandle,
    ModelHandle,
    NotebookHandle,
    PackageHandle,
    ResourceHandle,
    parse_competition_handle,
    parse_dataset_handle,
    parse_model_handle,
    parse_notebook_handle,
    parse_package_handle,
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
    PackageHandle: "Package",
}

HANDLE_TYPE_PARSERS = {
    "Competition": parse_competition_handle,
    "Dataset": parse_dataset_handle,
    "Model": parse_model_handle,
    "Notebook": parse_notebook_handle,
    "Package": parse_package_handle,
}

# Maps requested ResourceHandle (which may include version) to version used
VersionedDatasources = dict[ResourceHandle, Optional[int]]

# Tracks datasources accessed in the current session
_accessed_datasources: VersionedDatasources = {}


def register_accessed_datasource(handle: ResourceHandle, version: Optional[int]) -> None:
    """Record that a datasource was accessed.

    Link the user-requested handle to the version retrieved."""
    _accessed_datasources[handle] = version


def write_requirements(filepath: str) -> None:
    """Write the datasources accessed during this session to a yaml file."""
    data = {
        FORMAT_VERSION_FIELD: FORMAT_VERSION,
        DATASOURCES_FIELD: [_serialize_datasource(h, version) for h, version in _accessed_datasources.items()],
    }

    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def read_requirements(filepath: Union[str, pathlib.Path]) -> VersionedDatasources:
    """Read a yaml file with datasource + version records."""
    with open(filepath) as f:
        data = yaml.safe_load(f)

    format_version = data.get(FORMAT_VERSION_FIELD)
    if format_version != FORMAT_VERSION:
        msg = f"Unsupported requirements format version: {format_version}"
        raise ValueError(msg)

    versioned_datasources: VersionedDatasources = {}
    for datasource in data.get(DATASOURCES_FIELD, []):
        h, version = _deserialize_datasource(datasource)
        versioned_datasources[h] = version

    return versioned_datasources


def _serialize_datasource(h: ResourceHandle, version: Optional[int]) -> dict:
    data: dict[str, Any] = {
        DATASOURCE_TYPE_FIELD: HANDLE_TYPE_NAMES[type(h)],
        DATASOURCE_REF_FIELD: str(h),
    }

    if version is not None:
        data[DATASOURCE_VERSION_FIELD] = version

    return data


def _deserialize_datasource(data: dict) -> tuple[ResourceHandle, Optional[int]]:
    parser = HANDLE_TYPE_PARSERS[data[DATASOURCE_TYPE_FIELD]]
    h = parser(data[DATASOURCE_REF_FIELD])
    version = data.get(DATASOURCE_VERSION_FIELD, None)

    return h, version