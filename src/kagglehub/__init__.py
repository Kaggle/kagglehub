__version__ = "0.4.1"

import kagglehub.logger  # configures the library logger.
from kagglehub import colab_cache_resolver, http_resolver, kaggle_cache_resolver, registry
from kagglehub.auth import login, whoami
from kagglehub.competition import competition_download
from kagglehub.datasets import (
    KaggleDatasetAdapter,
    PolarsFrameType,
    dataset_download,
    dataset_load,
    dataset_upload,
    load_dataset,
)
from kagglehub.models import model_download, model_upload
from kagglehub.notebooks import notebook_output_download
from kagglehub.packages import get_package_asset_path, package_import
from kagglehub.utility_scripts import utility_script_install

registry.model_resolver.add_implementation(http_resolver.ModelHttpResolver())
registry.model_resolver.add_implementation(kaggle_cache_resolver.ModelKaggleCacheResolver())
registry.model_resolver.add_implementation(colab_cache_resolver.ModelColabCacheResolver())

registry.dataset_resolver.add_implementation(http_resolver.DatasetHttpResolver())
registry.dataset_resolver.add_implementation(kaggle_cache_resolver.DatasetKaggleCacheResolver())
registry.dataset_resolver.add_implementation(colab_cache_resolver.DatasetColabCacheResolver())

registry.competition_resolver.add_implementation(http_resolver.CompetitionHttpResolver())
registry.competition_resolver.add_implementation(kaggle_cache_resolver.CompetitionKaggleCacheResolver())

registry.notebook_output_resolver.add_implementation(http_resolver.NotebookOutputHttpResolver())
registry.notebook_output_resolver.add_implementation(kaggle_cache_resolver.NotebookOutputKaggleCacheResolver())
