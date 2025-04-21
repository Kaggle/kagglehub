import logging
import warnings
from typing import Any, Optional, Union

from kagglehub import registry
from kagglehub.datasets_enums import KaggleDatasetAdapter, PolarsFrameType
from kagglehub.datasets_helpers import create_dataset_or_version
from kagglehub.gcs_upload import normalize_patterns, upload_files_and_directories
from kagglehub.handle import parse_dataset_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


# Patterns that are always ignored for dataset uploading.
DEFAULT_IGNORE_PATTERNS = [".git/", "*/.git/", ".cache/", ".huggingface/"]
# Mapping of adapters to the optional dependencies required to run them
DATASET_LOAD_ADAPTER_OPTIONAL_DEPENDENCIES_MAP = {
    KaggleDatasetAdapter.HUGGING_FACE: "hf-datasets",
    KaggleDatasetAdapter.PANDAS: "pandas-datasets",
    KaggleDatasetAdapter.POLARS: "polars-datasets",
}

# Mapping of adapters to the valid kwargs to use for that adapter
_DATASET_LOAD_VALID_KWARGS_MAP = {
    KaggleDatasetAdapter.HUGGING_FACE: {"hf_kwargs", "pandas_kwargs", "sql_query"},
    KaggleDatasetAdapter.PANDAS: {"pandas_kwargs", "sql_query"},
    KaggleDatasetAdapter.POLARS: {"sql_query", "polars_frame_type", "polars_kwargs"},
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
    path, _ = registry.dataset_resolver(h, path, force_download=force_download)
    return path


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


def dataset_load(
    adapter: KaggleDatasetAdapter,
    # In the form of {owner_slug}/{dataset_slug} or {owner_slug}/{dataset_slug}/versions/{version_number}
    handle: str,
    path: str,
    *,
    pandas_kwargs: Any = None,  # noqa: ANN401
    sql_query: Optional[str] = None,
    hf_kwargs: Any = None,  # noqa: ANN401
    polars_frame_type: Optional[PolarsFrameType] = None,
    polars_kwargs: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Load a Kaggle Dataset into a python object based on the selected adapter

    Args:
        adapter: (KaggleDatasetAdapter) The adapter used to load the dataset
        handle: (string) The dataset handle
        path: (string) Path to a file within the dataset
        pandas_kwargs:
            (dict) Optional set of kwargs to pass to the pandas `read_*` method while constructing the DataFrame(s)
        sql_query:
            (string) Argument to be used for SQLite files. Required when reading a SQLite file. See pandas documentation
            for details: https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html
        hf_kwargs:
            (dict) Optional set of kwargs to pass to Dataset.from_pandas() while constructing the Dataset
        polars_frame_type: (PolarsFrameType) Optional control for which Frame to return: LazyFrame or DataFrame. The
            default is PolarsFrameType.LAZY_FRAME.
        polars_kwargs:
            (dict) Optional set of kwargs to pass to the polars `read_*` method while constructing the DataFrame(s)
    Returns:
        A python object based on the selected adapter:
            - 'pandas': A DataFrame (or dict[int | str, DataFrame] for Excel-like files with multiple sheets)
            - 'hugging_face': A Huggingface Dataset (via pandas)
            - 'polars':
                A LazyFrame or DataFrame (or dict[int | str, LazyFrame] / dict[int | str, DataFrame] for Excel-like
                files with multiple sheets)
    """
    validate_dataset_load_args(
        adapter,
        pandas_kwargs=pandas_kwargs,
        sql_query=sql_query,
        hf_kwargs=hf_kwargs,
        polars_frame_type=polars_frame_type,
        polars_kwargs=polars_kwargs,
    )
    polars_frame_type = polars_frame_type if polars_frame_type is not None else PolarsFrameType.LAZY_FRAME
    try:
        if adapter is KaggleDatasetAdapter.HUGGING_FACE:
            import kagglehub.hf_datasets

            return kagglehub.hf_datasets.load_hf_dataset(
                handle,
                path,
                pandas_kwargs=pandas_kwargs,
                sql_query=sql_query,
                hf_kwargs=hf_kwargs,
            )
        elif adapter is KaggleDatasetAdapter.PANDAS:
            import kagglehub.pandas_datasets

            return kagglehub.pandas_datasets.load_pandas_dataset(
                handle, path, pandas_kwargs=pandas_kwargs, sql_query=sql_query
            )
        elif adapter is KaggleDatasetAdapter.POLARS:
            import kagglehub.polars_datasets

            return kagglehub.polars_datasets.load_polars_dataset(
                handle, path, polars_frame_type=polars_frame_type, polars_kwargs=polars_kwargs, sql_query=sql_query
            )
        else:
            not_implemented_error_message = f"{adapter} is not yet implemented"
            raise NotImplementedError(not_implemented_error_message)
    except ImportError:
        adapter_optional_dependency = DATASET_LOAD_ADAPTER_OPTIONAL_DEPENDENCIES_MAP[adapter]
        import_warning_message = (
            f"The 'dataset_load' function requires the '{adapter_optional_dependency}' extras. "
            f"Install them with 'pip install kagglehub[{adapter_optional_dependency}]'"
        )
        # Inform the user if we detect that they didn't install everything
        raise ImportError(import_warning_message) from None


def load_dataset(
    adapter: KaggleDatasetAdapter,
    # In the form of {owner_slug}/{dataset_slug} or {owner_slug}/{dataset_slug}/versions/{version_number}
    handle: str,
    path: str,
    *,
    pandas_kwargs: Any = None,  # noqa: ANN401
    sql_query: Optional[str] = None,
    hf_kwargs: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    warnings.warn(
        "load_dataset is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2
    )
    return dataset_load(adapter, handle, path, pandas_kwargs=pandas_kwargs, sql_query=sql_query, hf_kwargs=hf_kwargs)


def validate_dataset_load_args(
    adapter: KaggleDatasetAdapter,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    valid_kwargs = _DATASET_LOAD_VALID_KWARGS_MAP[adapter]
    invalid_kwargs_list: list[str] = []
    for key, value in kwargs.items():
        if key not in valid_kwargs and value is not None:
            invalid_kwargs_list.append(key)

    if len(invalid_kwargs_list) == 0:
        return

    invalid_kwargs = ", ".join(invalid_kwargs_list)
    logger.warning(f"{invalid_kwargs} are invalid for {adapter} and will be ignored")
