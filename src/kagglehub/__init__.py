__version__ = "0.0.1a1"

from kagglehub import http_resolver, registry
from kagglehub.auth import login
from kagglehub.models import model_download, model_upload

registry.resolver.add_implementation(http_resolver.HttpResolver())
# TODO(b/305947763): Implement Kaggle Cache resolver.
