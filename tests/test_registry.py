from typing import Any, Callable

from kagglehub import registry
from kagglehub.handle import ResourceHandle
from kagglehub.resolver import Resolver
from tests.fixtures import BaseTestCase

SOME_VALUE: str = "test"


class FakeHandle(ResourceHandle):
    def to_url(self) -> str:
        raise NotImplementedError()


class FakeImpl(Resolver[FakeHandle]):
    def __init__(
        self,
        is_supported_fn: Callable[[FakeHandle], bool],
        call_fn: Callable[[FakeHandle], str],
    ):
        self._is_supported_fn = is_supported_fn
        self._call_fn = call_fn

    def is_supported(self, *args: Any, **kwargs: Any) -> bool:  # noqa: ANN401
        return self._is_supported_fn(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        return self._call_fn(*args, **kwargs)


def fail_fn(*_, **__) -> str:  # noqa: ANN002, ANN003
    msg = "fail_fn should not be called"
    raise AssertionError(msg)


class RegistryTest(BaseTestCase):
    def test_calls_only_supported(self) -> None:
        r = registry.MultiImplRegistry[FakeHandle]("test")
        r.add_implementation(FakeImpl(lambda *_, **__: True, lambda *_, **__: SOME_VALUE))
        r.add_implementation(FakeImpl(lambda *_, **__: False, fail_fn))

        val = r(FakeHandle())

        self.assertEqual(SOME_VALUE, val)

    def test_calls_first_supported_reverse(self) -> None:
        r = registry.MultiImplRegistry[FakeHandle]("test")
        r.add_implementation(FakeImpl(lambda *_, **__: True, fail_fn))
        r.add_implementation(FakeImpl(lambda *_, **__: True, lambda *_, **__: SOME_VALUE))

        val = r(FakeHandle())

        self.assertEqual(SOME_VALUE, val)

    def test_calls_throw_not_supported(self) -> None:
        r = registry.MultiImplRegistry[FakeHandle]("test")
        r.add_implementation(FakeImpl(lambda *_, **__: False, fail_fn))

        self.assertRaisesRegex(RuntimeError, r"Missing implementation", r, SOME_VALUE)
