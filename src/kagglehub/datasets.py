import logging
from enum import Enum
from typing import Any, Optional, Union

from kagglehub import registry
from kagglehub.datasets_helpers import create_dataset_or_version
from kagglehub.gcs_upload import normalize_patterns, upload_files_and_directories
from kagglehub.handle import parse_dataset_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


class KaggleDatasetAdapter(Enum):
    HUGGING_FACE = "hugging_face"
    PANDAS = "pandas"


# Patterns that are always ignored for dataset uploading.
DEFAULT_IGNORE_PATTERNS = [".git/", "*/.git/", ".cache/", ".huggingface/"]
# Mapping of adapters to the optional dependencies required to run them
LOAD_DATASET_ADAPTER_OPTIONAL_DEPENDENCIES_MAP = {
    KaggleDatasetAdapter.HUGGING_FACE: "hf-datasets",
    KaggleDatasetAdapter.PANDAS: "pandas-datasets",
}


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
    adapter: KaggleDatasetAdapter,
    # In the form of {owner_slug}/{dataset_slug} or {owner_slug}/{dataset_slug}/versions/{version_number}
    handle: str,
    path: str,
    *,
    # Kaggle doesn't currently support test/train splits natively like Hugging Face does, so users
    # would specify here if they select the 'hugging_face' adapter.
    train_split_percent: float = 0.8,
    columns: Optional[list] = None,
    sheet_name: str | int | list | None = 0,
    sql_query: Optional[str] = None,
) -> Any:  # noqa: ANN401
    """Load a Kaggle Dataset into a python object based on the selected adapter

    Args:
        adapter: (KaggleDatasetAdapter) The adapter used to load the dataset
        handle: (string) The dataset handle
        path: (string) Path to a file within the dataset
        train_split_percent:
            (float) Optional split for train vs test data, defaults to 0.8. Only used for the 'hugging_face' adapter
        columns:
            (list) Optional subset of columns to load from the file. Only used for CSV, TSV, Excel-like, feather,
            and parquet files.
        sheet_name:
            (string, int, list, or None) Optional argument to be used for Excel-like files.
            Defaults to 0 to select the first sheet. See pandas documentation for details:
            https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
        sql_query:
            (string) Optional argument to be used for SQLite files. See pandas documentation for details:
            https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html
    Returns:
        A python object based on the selected adapter:
            - 'hugging_face': A DatasetDict with test and train splits according to the provided train_split_percent
            - 'pandas': A DataFrame (or dict[int | str, DataFrame] for Excel-like files with multiple sheets)
    """
    try:
        if adapter is KaggleDatasetAdapter.HUGGING_FACE:
            import kagglehub.hf_datasets

            return kagglehub.hf_datasets.load_hf_dataset(
                handle, path, train_split_percent, columns=columns, sheet_name=sheet_name, sql_query=sql_query
            )
        elif adapter is KaggleDatasetAdapter.PANDAS:
            import kagglehub.pandas_datasets

            return kagglehub.pandas_datasets.load_pandas_dataset(
                handle, path, columns=columns, sheet_name=sheet_name, sql_query=sql_query
            )
        else:
            not_implemented_error_message = f"{adapter} is not yet implemented"
            raise NotImplementedError(not_implemented_error_message)
    except ImportError:
        adapter_optional_dependency = LOAD_DATASET_ADAPTER_OPTIONAL_DEPENDENCIES_MAP[adapter]
        import_warning_message = (
            f"The 'load_dataset' function requires the '{adapter_optional_dependency}' extras. "
            "Install them with 'pip install kagglehub[{adapter_optional_dependency}]'"
        )
        # Inform the user if we detect that they didn't install everything
        raise ImportError(import_warning_message) from None
