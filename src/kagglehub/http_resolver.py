from typing import Optional

from kagglehub import resolver


class HttpResolver(resolver.Resolver):
    def is_supported(self, *_):
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def __call__(self, handle: str, path: Optional[str] = None):
        # TODO(b/305947384): Parse handle, call models download API & implement resumable download.
        raise NotImplementedError()
