from typing import Optional

from kagglehub import registry
from kagglehub.handle import parse_model_handle


def model_download(handle: str, path: Optional[str] = None):
    """Download model files.

    Args:
        handle: (string) the model handle.
        path: (string) Optional path to a file within the model bundle.

    Returns:
        A string representing the path to the requested model files.
    """
    h = parse_model_handle(handle)
    return registry.resolver(h, path)


def model_upload():
    raise NotImplementedError()
