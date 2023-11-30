__version__ = "0.1.0"

import kagglehub.logging
from kagglehub import http_resolver, kaggle_cache_resolver, registry
from kagglehub.auth import login
from kagglehub.models import model_download, model_upload

registry.resolver.add_implementation(http_resolver.HttpResolver())
registry.resolver.add_implementation(kaggle_cache_resolver.KaggleCacheResolver())
