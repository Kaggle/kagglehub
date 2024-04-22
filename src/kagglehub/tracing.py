import secrets
from types import TracebackType
from typing import Optional, Type

# Constants can be found at https://www.w3.org/TR/trace-context/#version-format
_TRACE_LENGTH_BYTES = 16
_SPAN_LENGTH_BYTES = 8


class TraceContext:
    """
    Generates and manages identifiers for distributed tracing.

    More information on trace can be found at https://www.w3.org/TR/trace-context/

    Attributes:
        trace: A 16-byte hexadecimal string representing a unique trace ID.
    """

    def __init__(self) -> None:
        self.trace = secrets.token_bytes(_TRACE_LENGTH_BYTES).hex()

    def __enter__(self) -> "TraceContext":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        return

    def next(self) -> str:
        """
        Generates a new span ID within the context of the current trace ID.

        An example traceparent:
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

        Returns:
            A formatted string representing a span ID, in a standard
            distributed tracing format (e.g., "00-{trace}-{span}-01")
        """
        span = secrets.token_bytes(_SPAN_LENGTH_BYTES).hex()
        return f"00-{self.trace}-{span}-01"


def default_context_factory() -> TraceContext:
    return TraceContext()
