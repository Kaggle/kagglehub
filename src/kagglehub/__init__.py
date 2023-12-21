__version__ = "0.1.4"

import kagglehub.logging  # configures the library logger.
from kagglehub import http_resolver, kaggle_cache_resolver, registry
from kagglehub.auth import login
from kagglehub.models import model_download, model_upload

registry.model_resolver.add_implementation(http_resolver.ModelHttpResolver())
registry.model_resolver.add_implementation(kaggle_cache_resolver.ModelKaggleCacheResolver())
# TODO(b/313706281): Register dataset resolvers here.
