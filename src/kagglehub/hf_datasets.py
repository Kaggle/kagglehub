from typing import Any, Optional

# WARNING: This module is intended to be imported only at runtime, with
# specific error handling to inform users that they need to install the
# appropriate `hf-datasets` extras in order to use these methods. Adding
# any new dependencies here must correspond with an addition to the
# list of `hf-datasets` optional dependencies in pyproject.toml.
from datasets import Dataset

from kagglehub.pandas_datasets import load_pandas_dataset

MULTIPLE_DATA_FRAMES_ERROR_MESSAGE = (
    "Loading a Huggingface dataset requires the production of exactly one DataFrame. "
    "For example, if using an Excel-based dataset, you must specify a single sheet"
)
DEFAULT_PANDAS_KWARGS = {"preserve_index": False}


def load_hf_dataset(
    handle: str,
    path: str,
    *,
    pandas_kwargs: Any = None,  # noqa: ANN401
    sql_query: Optional[str] = None,
    hf_kwargs: Any = None,  # noqa: ANN401
) -> Dataset:
    """Load a Kaggle Dataset into a Huggingface Dataset (via pandas)

    Args:
        handle: (string) The dataset handle
        path: (string) Path to a file within the dataset
        pandas_kwargs:
            (dict) Optional set of kwargs to pass to the pandas `read_*` method while constructing the DataFrame(s)
        sql_query:
            (string) Argument to be used for SQLite files. Required when reading a SQLite file. See pandas documentation
            for details: https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html
        hf_kwargs:
            (dict) Optional set of kwargs to pass to Dataset.from_pandas() while constructing the Dataset
    Returns:
        Dataset
    """
    hf_kwargs = {} if hf_kwargs is None else hf_kwargs
    pandas_kwargs = {} if pandas_kwargs is None else pandas_kwargs
    result = load_pandas_dataset(handle, path, pandas_kwargs=pandas_kwargs, sql_query=sql_query)

    # We're only loading one Dataset at a time, so alert the user if their args produce multiple. This will
    # probably only happen if the user is trying to load an Excel-based file with many sheets and none specified
    if isinstance(result, dict) or isinstance(result, list):
        raise ValueError(MULTIPLE_DATA_FRAMES_ERROR_MESSAGE)

    # NOTE: Order matters here as we're letting users override our specified defaults, namely preserve_index=False.
    # This may be valuable in the edge case that a user does actually want the index persisted as a column.
    merged_kwargs = {**DEFAULT_PANDAS_KWARGS, **hf_kwargs}
    return Dataset.from_pandas(result, **merged_kwargs)
