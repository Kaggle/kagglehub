from kagglehub import registry
from tests.fixtures import BaseTestCase

SOME_VALUE = 1


def fail_fn(_):
    msg = "fail_fn should not be called"
    raise AssertionError(msg)


class FakeImpl:
    def __init__(self, is_supported_fn, call_fn):
        self._is_supported_fn = is_supported_fn
        self._call_fn = call_fn

    def is_supported(self, *args, **kwargs):
        return self._is_supported_fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._call_fn(*args, **kwargs)


class RegistryTest(BaseTestCase):
    def test_calls_only_supported(self):
        r = registry.MultiImplRegistry("test")
        r.add_implementation(FakeImpl(lambda _: True, lambda _: SOME_VALUE))
        r.add_implementation(FakeImpl(lambda _: False, fail_fn))

        val = r(SOME_VALUE)

        self.assertEqual(SOME_VALUE, val)

    def test_calls_first_supported_reverse(self):
        r = registry.MultiImplRegistry("test")
        r.add_implementation(FakeImpl(lambda _: True, fail_fn))
        r.add_implementation(FakeImpl(lambda _: True, lambda _: SOME_VALUE))

        val = r(SOME_VALUE)

        self.assertEqual(SOME_VALUE, val)

    def test_calls_throw_not_supported(self):
        r = registry.MultiImplRegistry("test")
        r.add_implementation(FakeImpl(lambda _: False, fail_fn))

        self.assertRaisesRegex(RuntimeError, r"Missing implementation", r, SOME_VALUE)
