import abc
from typing import Generic, Optional, TypeVar

from kagglehub.handle import ResourceHandle
from kagglehub.requirements import register_accessed_datasource

T = TypeVar("T", bound=ResourceHandle)


class Resolver(Generic[T]):
    """Resolver base class: all resolvers inherit from this class."""

    __metaclass__ = abc.ABCMeta

    def __call__(
        self, handle: T, path: Optional[str] = None, *, force_download: Optional[bool] = False
    ) -> tuple[str, Optional[int]]:
        """Resolves a handle into a path with the requested model files.

        Args:
            handle: (string) the model handle to resolve.
            path: (string) Optional path to a file within the model bundle.
            force_download: (bool) Optional flag to force download a model, even if it's cached.

        Returns:
            String representing the path.
        """
        path, version = self._resolve(handle, path, force_download=force_download)

        # Note handles are immutable, so resolve() could not have altered our reference
        register_accessed_datasource(handle, version)

        return path, version

    @abc.abstractmethod
    def _resolve(
        self, handle: T, path: Optional[str] = None, *, force_download: Optional[bool] = False
    ) -> tuple[str, Optional[int]]:
        """Resolves a handle into a path with the requested model files.

        Args:
            handle: (string) the model handle to resolve.
            path: (string) Optional path to a file within the model bundle.
            force_download: (bool) Optional flag to force download a model, even if it's cached.

        Returns:
            A tuple of: 1) string representing the path 2) version number of resolved datasource, if applicable.
        """
        pass

    @abc.abstractmethod
    def is_supported(self, handle: T, path: Optional[str]) -> bool:
        """Returns whether the current environment supports this handle/path."""
        pass
