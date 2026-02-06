import logging

from kagglesdk.blobs.types.blob_api_service import ApiBlobType

from kagglehub import registry
from kagglehub.gcs_upload import normalize_patterns, upload_files_and_directories
from kagglehub.handle import parse_model_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.models_helpers import create_model_if_missing, create_model_instance_or_version
from kagglehub.signing import sign_with_sigstore

logger = logging.getLogger(__name__)

# Patterns that are always ignored for model uploading.
DEFAULT_IGNORE_PATTERNS = [".git/", "*/.git/", ".cache/", ".huggingface/"]


def model_download(
    handle: str,
    path: str | None = None,
    *,
    force_download: bool | None = False,
    output_dir: str | None = None,
) -> str:
    """Download model files.

    Args:
        handle: (string) the model handle.
        path: (string) Optional path to a file within the model bundle.
        force_download: (bool) Optional flag to force download a model, even if it's cached or already in output_dir.
        output_dir: (string) Optional output directory for direct download, bypassing the default cache.

    Returns:
        A string representing the path to the requested model files.
    """
    h = parse_model_handle(handle)
    logger.info(f"Downloading Model: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    path, _ = registry.model_resolver(
        h,
        path,
        force_download=force_download,
        output_dir=output_dir,
    )
    return path


def model_upload(
    handle: str,
    local_model_dir: str,
    license_name: str | None = None,
    version_notes: str = "",
    ignore_patterns: list[str] | str | None = None,
    *,
    sigstore: bool | None = False,
) -> None:
    """Upload model files.

    Args:
        handle: (str) the model handle.
        local_model_dir: (str) path to a file in a local directory.
        license_name: (str) model license.
        version_notes: (str, optional) model versions.
        ignore_patterns (str or list[str], optional):
            Additional ignore patterns to DEFAULT_IGNORE_PATTERNS.
            Files matching any of the patterns are not uploaded.
            Patterns are standard wildcards that can be matched by
            https://docs.python.org/3/library/fnmatch.html.
            Use a pattern ending with "/" to ignore the whole dir,
            e.g., ".git/" is equivalent to ".git/*".
        sigstore: (bool, optional)
            Creates a trasparent ledger on sigstore. User must be an admin/editor of the model.
    """
    # parse slug
    h = parse_model_handle(handle)
    logger.info(f"Uploading Model {h.to_url()} ...")
    if h.is_versioned():
        is_versioned_exception = "The model handle should not include the version"
        raise ValueError(is_versioned_exception)

    # Create the model if it doesn't already exist
    create_model_if_missing(h.owner, h.model)

    # Model can be non-existent. Get token after model creation so signing token can be authorized.
    try:
        if sigstore:
            sigstore = sign_with_sigstore(local_model_dir, h)
    except ImportError:
        import_warning_message = (
            "The 'model_upload(...sign=True)' function requires the 'kagglehub[signing]' extras. "
            "Install them with 'pip install kagglehub[signing]'"
        )
        # Inform the user if we detect that they didn't install everything
        raise ImportError(import_warning_message) from None

    # Upload the model files to GCS
    tokens = upload_files_and_directories(
        local_model_dir,
        item_type=ApiBlobType.MODEL,
        ignore_patterns=normalize_patterns(default=DEFAULT_IGNORE_PATTERNS, additional=ignore_patterns),
    )

    create_model_instance_or_version(h, tokens, license_name, version_notes, sigstore=sigstore)
