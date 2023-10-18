from typing import Optional

from kagglehub.cache import load_from_cache
from kagglehub.handle import parse_model_handle
from kagglehub.resolver import Resolver


class HttpResolver(Resolver):
    def is_supported(self, *_):
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def __call__(self, handle: str, path: Optional[str] = None):
        model_handle = parse_model_handle(handle)
        model_path = load_from_cache(model_handle, path)
        if model_path:
            return model_path  # Already cached

        # TODO(b/305947384): Call models download API & implement resumable download.
        raise NotImplementedError()
