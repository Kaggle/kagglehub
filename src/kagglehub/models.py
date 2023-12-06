import os
import shutil
import tempfile
from typing import Optional
import logging
import requests

from kagglehub import registry
from kagglehub.handle import parse_model_handle
import re
from kagglehub.clients import KaggleApiV1Client
from kagglehub.models_helpers import get_or_create_model, create_model_instance_or_version

logger = logging.getLogger(__name__)

DOES_NOT_EXIST_ERROR = 404


def model_download(handle: str, path: Optional[str] = None):
    """Download model files.

    Args:
        handle: (string) the model handle.
        path: (string) Optional path to a file within the model bundle.

    Returns:
        A string representing the path to the requested model files.
    """
    h = parse_model_handle(handle)
    return registry.resolver(h, path)


def model_upload(handle: str, local_model_dir, license, version_notes: Optional[str] = None):
    # parse slug
    h = parse_model_handle(handle)

    # Create the model if it doesn't already exist
    get_or_create_model(h.owner, h.model)
    
    # TODO(aminmohamed): Upload the model files to GCS

    # Create a model instance if it doesn't exist, and create a new instance version if an instance exists
    file_paths = []
    for root, directories, files in os.walk(local_model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    create_model_instance_or_version(h.owner, h.model, h.framework, h.version, license, file_paths, version_notes)


