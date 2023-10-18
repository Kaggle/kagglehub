from typing import Optional

from kagglehub import registry


def model_download(handle: str, path: Optional[str] = None, cache_dir: Optional[str] = None):
    """Download model files.

    Args:
        handle: (string) the model handle.
        path: (string) Optional path to a file within the model bundle.
        cache_dir: (string) Optional cache directory to use. Defaults to ~/.cache/kagglehub.

    Returns:
        A string representing the path to the requested model files.
    """
    return registry.resolver(handle, path, cache_dir)


def model_upload():
    raise NotImplementedError()
