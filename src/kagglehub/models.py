from typing import Optional

from kagglehub import registry


def model_download(handle: str, path: Optional[str] = None):
    """Download model files.

    Args:
        handle: (string) the model handle.
        path: (string) Optional path to a file within the model bundle.

    Returns:
        A string representing the path to the requested model files.
    """
    return registry.resolver(handle, path)


def model_upload():
    raise NotImplementedError()
