import abc
from typing import Optional

from kagglehub.handle import ModelHandle


class Resolver:
    """Resolver base class: all resolvers inherit from this class."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, handle: ModelHandle, path: Optional[str] = None) -> str:
        """Resolves a handle into a path with the requested model files.

        Args:
            handle: (string) the model handle to resolve.
            path: (string) Optional path to a file within the model bundle.


        Returns:
            A string representing the path
        """
        pass

    @abc.abstractmethod
    def is_supported(self, handle: ModelHandle, path: Optional[str] = None):
        """Returns whether the current environment supports this handle/path."""
        pass
