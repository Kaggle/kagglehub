"""Functions to parse resource handles."""

import abc
from dataclasses import dataclass
from typing import Optional

from kagglehub.config import get_kaggle_api_endpoint

NUM_VERSIONED_MODEL_PARTS = 5  # e.g.: <owner>/<model>/<framework>/<variation>/<version>
NUM_UNVERSIONED_MODEL_PARTS = 4  # e.g.: <owner>/<model>/<framework>/<variation>

# TODO(b/313706281): Implement a DatasetHandle class & parse_dataset_handle method.


@dataclass
class ResourceHandle:
    owner: str

    @abc.abstractmethod
    def to_url(self) -> str:
        """Returns URL to the resource detail page."""
        pass


@dataclass
class ModelHandle(ResourceHandle):
    model: str
    framework: str
    variation: str
    version: Optional[int]

    def is_versioned(self) -> bool:
        return self.version is not None and self.version > 0

    def __str__(self) -> str:
        handle_str = f"{self.owner}/{self.model}/{self.framework}/{self.variation}"
        if self.is_versioned():
            return f"{handle_str}/{self.version}"
        return handle_str

    def to_url(self) -> str:
        endpoint = get_kaggle_api_endpoint()
        if self.is_versioned():
            return f"{endpoint}/models/{self.owner}/{self.model}/{self.framework}/{self.variation}/{self.version}"
        else:
            return f"{endpoint}/models/{self.owner}/{self.model}/{self.framework}/{self.variation}"


@dataclass
class DatasetHandle:
    owner: str
    dataset_name: str
    version: Optional[int] = None

    def is_versioned(self) -> bool:
        return self.version is not None

    def __str__(self) -> str:
        handle_str = f"{self.owner}/{self.dataset_name}"
        if self.is_versioned():
            return f"{handle_str}/{self.version}"
        return handle_str

    def to_url(self) -> str:
        endpoint = get_kaggle_api_endpoint()
        base_url = f"{endpoint}/datasets/{self.owner}/{self.dataset_name}"
        if self.is_versioned():
            return f"{base_url}/versions/{self.version}"
        return base_url


def parse_model_handle(handle: str) -> ModelHandle:
    parts = handle.split("/")

    if len(parts) == NUM_VERSIONED_MODEL_PARTS:
        # Versioned handle
        # e.g.: <owner>/<model>/<framework>/<variation>/<version>
        try:
            version = int(parts[4])
        except ValueError as err:
            msg = f"Invalid version number: {parts[4]}"
            raise ValueError(msg) from err

        return ModelHandle(
            owner=parts[0],
            model=parts[1],
            framework=parts[2],
            variation=parts[3],
            version=version,
        )
    elif len(parts) == NUM_UNVERSIONED_MODEL_PARTS:
        # Unversioned handle
        # e.g.: <owner>/<model>/<framework>/<variation>
        return ModelHandle(
            owner=parts[0],
            model=parts[1],
            framework=parts[2],
            variation=parts[3],
            version=None,
        )

    msg = f"Invalid model handle: {handle}"
    raise ValueError(msg)
