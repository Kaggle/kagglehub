__version__ = "0.2.6"

import kagglehub.logger  # configures the library logger.
from kagglehub import colab_cache_resolver, http_resolver, kaggle_cache_resolver, registry
from kagglehub.auth import login, whoami
from kagglehub.datasets import dataset_download
from kagglehub.models import model_download, model_upload

registry.model_resolver.add_implementation(http_resolver.ModelHttpResolver())
registry.model_resolver.add_implementation(kaggle_cache_resolver.ModelKaggleCacheResolver())
registry.model_resolver.add_implementation(colab_cache_resolver.ModelColabCacheResolver())

registry.dataset_resolver.add_implementation(http_resolver.DatasetHttpResolver())
