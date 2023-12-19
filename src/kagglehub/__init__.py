__version__ = "0.1.4"

from kagglehub import http_resolver, kaggle_cache_resolver, registry

registry.resolver.add_implementation(http_resolver.HttpResolver())
registry.resolver.add_implementation(kaggle_cache_resolver.KaggleCacheResolver())
