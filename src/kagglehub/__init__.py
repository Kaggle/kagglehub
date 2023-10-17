__version__ = "0.0.1a0"

from kagglehub.auth import login
from kagglehub.config import _install_resolvers
from kagglehub.models import model_download, model_upload

_install_resolvers()
