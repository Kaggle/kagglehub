from typing import Any, Callable

from kagglehub import registry
from tests.fixtures import BaseTestCase

SOME_VALUE = 1


def fail_fn(_) -> None:  # noqa: ANN001
    msg = "fail_fn should not be callted"
    raise AssertionError(msg)


class FakeImpl:
    def __init__(self, is_supported_fn: Callable[..., bool], call_fn: Callable[..., Any]) -> None:
        self._is_supported_fn = is_supported_fn
        self._call_fn = call_fn

    def is_supported(self, *args: Any, **kwargs: Any) -> bool:  # noqa: ANN401
        return self._is_supported_fn(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._call_fn(*args, **kwargs)


class RegistryTest(BaseTestCase):
    def test_calls_only_supported(self) -> None:
        r = registry.MultiImplRegistry("test")
        r.add_implementation(FakeImpl(lambda _: True, lambda _: SOME_VALUE))
        r.add_implementation(FakeImpl(lambda _: False, fail_fn))

        val = r(SOME_VALUE)

        self.assertEqual(SOME_VALUE, val)

    def test_calls_first_supported_reverse(self) -> None:
        r = registry.MultiImplRegistry("test")
        r.add_implementation(FakeImpl(lambda _: True, fail_fn))
        r.add_implementation(FakeImpl(lambda _: True, lambda _: SOME_VALUE))

        val = r(SOME_VALUE)

        self.assertEqual(SOME_VALUE, val)

    def test_calls_throw_not_supported(self) -> None:
        r = registry.MultiImplRegistry("test")
        r.add_implementation(FakeImpl(lambda _: False, fail_fn))

        self.assertRaisesRegex(RuntimeError, r"Missing implementation", r, SOME_VALUE)
