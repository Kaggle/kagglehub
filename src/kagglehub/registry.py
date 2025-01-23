from typing import Generic, TypeVar

from kagglehub.handle import CompetitionHandle, DatasetHandle, ModelHandle, NotebookHandle, ResourceHandle
from kagglehub.resolver import Resolver

T = TypeVar("T", bound=ResourceHandle)


class MultiImplRegistry(Generic[T]):
    """Utility class to inject multiple implementations of class.

    Each implementation must implement __call__ and is_supported with the same set of arguments. The registered
    implementations "is_supported" methods are called in reverse order under which they are registered. The first
    to return true is then invoked via __call__ and the result returned.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._impls: list[Resolver[T]] = []

    def add_implementation(self, impl: Resolver[T]) -> None:
        self._impls.append(impl)

    def __call__(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        fails = []
        for impl in reversed(self._impls):
            if impl.is_supported(*args, **kwargs):
                return impl(*args, **kwargs)
            else:
                fails.append(type(impl).__name__)

        msg = f"Missing implementation that supports: {self._name}(*{args!r}, **{kwargs!r}). Tried {fails!r}"
        raise RuntimeError(msg)


model_resolver = MultiImplRegistry[ModelHandle]("ModelResolver")
dataset_resolver = MultiImplRegistry[DatasetHandle]("DatasetResolver")
competition_resolver = MultiImplRegistry[CompetitionHandle]("CompetitionResolver")
notebook_output_resolver = MultiImplRegistry[NotebookHandle]("NotebookOutputResolver")
