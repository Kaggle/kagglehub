import abc
from typing import Optional

from kagglehub.handle import ModelHandle


class ModelResolver:
    """ModelResolver base class: all model resolvers inherit from this class."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(
        self, handle: ModelHandle, path: Optional[str] = None, *, force_download: Optional[bool] = False
    ) -> str:
        """Resolves a handle into a path with the requested model files.

        Args:
            handle: (string) the model handle to resolve.
            path: (string) Optional path to a file within the model bundle.
            force_download: (bool) Optional flag to force download a model, even if it's cached.


        Returns:
            A string representing the path
        """
        pass

    @abc.abstractmethod
    def is_supported(self, handle: ModelHandle, path: Optional[str] = None):
        """Returns whether the current environment supports this handle/path."""
        pass
