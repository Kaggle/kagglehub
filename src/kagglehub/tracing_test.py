import unittest

from kagglehub.tracing import TraceContext

_CANONICAL_EXAMPLE = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


class TraceContextSuite(unittest.TestCase):
    def test_length(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            self.assertEqual(len(traceparent), len(_CANONICAL_EXAMPLE))

    def test_prefix(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            self.assertEqual(traceparent[0:2], "00")

    def test_suffix(self) -> None:
        # always sample
        with TraceContext() as ctx:
            traceparent = ctx.next()
            self.assertEqual(traceparent[-2:], "01")

    def test_pattern(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            version, trace, span, flag = traceparent.split("-")
            self.assertRegex(version, "^[0-9]{2}$", "version does not meet pattern")
            self.assertRegex(trace, "^[A-Fa-f0-9]{32}$")
            self.assertRegex(span, "^[A-Fa-f0-9]{16}$")
            self.assertRegex(flag, "^[0-9]{2}$")

    def test_notempty(self) -> None:
        with TraceContext() as ctx:
            traceparent = ctx.next()
            _, trace, span, _ = traceparent.split("-")
            self.assertNotEqual(trace, f"{0:016x}")
            self.assertNotEqual(span, f"{0:08x}")


if __name__ == "__main__":
    unittest.main()
