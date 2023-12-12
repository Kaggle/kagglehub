from typing import Optional

from kagglehub import registry
from kagglehub.handle import parse_model_handle


def model_download(handle: str, path: Optional[str] = None, force: Optional[bool] = False):
    """Download model files.

    Args:
        handle: (string) the model handle.
        path: (string) Optional path to a file within the model bundle.
        force: (bool) Optional flag to force download a model, even if it's cached.


    Returns:
        A string representing the path to the requested model files.
    """
    h = parse_model_handle(handle)
    return registry.resolver(h, path, force)


def model_upload():
    raise NotImplementedError()
