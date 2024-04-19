import secrets

class TraceContext:
    """
    Generates and manages identifiers for distributed tracing.

    More information on trace can be found at https://www.w3.org/TR/trace-context/

    Attributes:
        trace: A 16-byte hexadecimal string representing a unique trace ID.
    """
    def __init__(self) -> None:
        self.trace = secrets.token_bytes(16).hex()
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass

    def next(self):
        """
        Generates a new span ID within the context of the current trace ID.

        Returns:
            A formatted string representing a span ID, in a standard 
            distributed tracing format (e.g., "00-{trace}-{span}-01")
        """
        span = secrets.token_bytes(8).hex()
        return f"00-{self.trace}-{span}-01"
