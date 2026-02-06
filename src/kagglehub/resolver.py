import abc
from typing import Generic, TypeVar

from kagglehub.handle import ResourceHandle
from kagglehub.tracker import register_datasource_access

T = TypeVar("T", bound=ResourceHandle)


class Resolver(Generic[T]):
    """Resolver base class: all resolvers inherit from this class."""

    __metaclass__ = abc.ABCMeta

    def __call__(
        self,
        handle: T,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        """Resolves a handle into a path with the requested file(s) and the resource's version number.

        Args:
            handle: (T) the ResourceHandle to resolve.
            path: (string) Optional path to a file within the resource.
            force_download: (bool) Optional flag to force download, even if it's cached or already in output_dir.
            output_dir: (string) Optional output directory for direct download, bypassing the default cache.

        Returns:
            A tuple of: (string representing the path, version number of resolved datasource if present)
            Some cases where version number might be missing: Competition datasource, API-based models.
        """
        path, version = self._resolve(
            handle,
            path,
            force_download=force_download,
            output_dir=output_dir,
        )

        # Note handles are immutable, so _resolve() could not have altered our reference
        register_datasource_access(handle, version)

        return path, version

    @abc.abstractmethod
    def _resolve(
        self,
        handle: T,
        path: str | None = None,
        *,
        force_download: bool | None = False,
        output_dir: str | None = None,
    ) -> tuple[str, int | None]:
        """Resolves a handle into a path with the requested file(s) and the resource's version number.

        Args:
            handle: (T) the ResourceHandle to resolve.
            path: (string) Optional path to a file within the resource.
            force_download: (bool) Optional flag to force download, even if it's cached or already in output_dir.
            output_dir: (string) Optional output directory for direct download, bypassing the default cache.

        Returns:
            A tuple of: (string representing the path, version number of resolved datasource if present)
            Some cases where version number might be missing: Competition datasource, API-based models.
        """
        pass

    @abc.abstractmethod
    def is_supported(self, handle: T, path: str | None = None) -> bool:
        """Returns whether the current environment supports this handle/path."""
        pass
