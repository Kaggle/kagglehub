import logging
from typing import Any, Optional, Union

from kagglehub import registry
from kagglehub.datasets_helpers import create_dataset_or_version
from kagglehub.gcs_upload import normalize_patterns, upload_files_and_directories
from kagglehub.handle import parse_dataset_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)

# Patterns that are always ignored for dataset uploading.
DEFAULT_IGNORE_PATTERNS = [".git/", "*/.git/", ".cache/", ".huggingface/"]
LOAD_DATASET_IMPORT_WARNING_MESSAGE = "The 'load_dataset' function requires the 'hf-datasets' extras. Install them with 'pip install kagglehub[hf-datasets]'"  # noqa: E501


def dataset_download(handle: str, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
    """Download dataset files
    Args:
        handle: (string) the dataset handle
        path: (string) Optional path to a file within a dataset
        force_download: (bool) Optional flag to force download a dataset, even if it's cached
    Returns:
        A string requesting the path to the requested dataset files.
    """

    h = parse_dataset_handle(handle)
    logger.info(f"Downloading Dataset: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    return registry.dataset_resolver(h, path, force_download=force_download)


def dataset_upload(
    handle: str,
    local_dataset_dir: str,
    version_notes: str = "",
    ignore_patterns: Optional[Union[list[str], str]] = None,
) -> None:
    """Upload dataset files.
    Args:
        handle: (string) the dataset handle.
        local_dataset_dir: (string) path to a file in a local directory.
        version_notes: (string) Optional to write dataset versions.
        ignore_patterns (str or list[str], optional):
            Additional ignore patterns to DEFAULT_IGNORE_PATTERNS.
            Files matching any of the patterns are not uploaded.
            Patterns are standard wildcards that can be matched by
            https://docs.python.org/3/library/fnmatch.html.
            Use a pattern ending with "/" to ignore the whole dir,
            e.g., ".git/" is equivalent to ".git/*".
    """
    h = parse_dataset_handle(handle)
    logger.info(f"Uploading Dataset {h.to_url()} ...")
    if h.is_versioned():
        is_versioned_exception = "The dataset handle should not include the version"
        raise ValueError(is_versioned_exception)

    tokens = upload_files_and_directories(
        local_dataset_dir,
        item_type="dataset",
        ignore_patterns=normalize_patterns(default=DEFAULT_IGNORE_PATTERNS, additional=ignore_patterns),
    )

    create_dataset_or_version(h, tokens, version_notes)


def load_dataset(
    # In the form of {owner_slug}/{dataset_slug} or {owner_slug}/{dataset_slug}/versions/{version_number}
    dataset_handle: str,
    # Kaggle doesn't support test/train splits natively like HF does, so users
    # would specify here. See later for why this is optional with a default.
    train_split_percent: float = 0.8,
    # The path to the tabular file, including the file extension.
    # If None, use the first table determined by breadth-first search from the root of the dataset.
    path: Optional[str] = None,
) -> Any:  # noqa: ANN401
    """Load a Kaggle Dataset as a Huggingface DatasetDict
    Args:
        handle: (string) the dataset handle
        train_split_percent: (float) Optional split for train vs test data, defaults to 0.8
        path: (string) Optional path to a file within a dataset
    Returns:
        A DatasetDict with test and train splits according to the train_split_percent
    """
    h = parse_dataset_handle(dataset_handle)

    try:
        import kagglehub.hf_datasets

        return kagglehub.hf_datasets.load_hf_dataset(h, train_split_percent, path)
    except ImportError:
        # Inform the user if we detect that they didn't install everything
        raise ImportError(LOAD_DATASET_IMPORT_WARNING_MESSAGE) from None
