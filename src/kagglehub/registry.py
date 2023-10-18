from typing import Callable, List


class MultiImplRegistry:
    """Utility class to inject multiple implementations of class.

    Each implementation must implement __call__ and is_supported with the same set of arguments. The registered
    implementations "is_supported" methods are called in reverse order under which they are registered. The first
    to return true is then invoked via __call__ and the result returned.
    """

    def __init__(self, name: str):
        self._name = name
        self._impls: List[Callable] = []

    def add_implementation(self, impl: Callable):
        self._impls += [impl]

    def __call__(self, *args, **kwargs):
        fails = []
        for impl in reversed(self._impls):
            if impl.is_supported(*args, **kwargs):
                return impl(*args, **kwargs)
            else:
                fails.append(type(impl).__name__)

        msg = f"Missing implementation that supports: {self._name}(*{args!r}, **{kwargs!r}). Tried {fails!r}"
        raise RuntimeError(msg)


resolver = MultiImplRegistry("resolver")
