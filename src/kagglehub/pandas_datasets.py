import os
import sqlite3
from typing import Any, Callable, Optional, Union

# WARNING: This module is intended to be imported only at runtime, with
# specific error handling to inform users that they need to install the
# appropriate `pandas-datasets` extras in order to use these methods. Adding
# any new dependencies here must correspond with an addition to the
# list of `pandas-datasets` optional dependencies in pyproject.toml.
import pandas as pd

from kagglehub.datasets import dataset_download


# This is a thin wrapper around pd.read_sql_query so we get the connection
# closing for free after we've produced the DataFrame
def wrapped_read_sql_query(sql_query: str, path: str) -> pd.DataFrame:
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(sql_query, conn)
        return df


# These are the currently supported read functions for the pandas adapter.
# Taken from https://pandas.pydata.org/docs/reference/io.html. More can be
# added as demand dictates.
SUPPORTED_READ_FUNCTIONS_BY_EXTENSION: dict[str, Callable] = {
    ".csv": pd.read_csv,
    ".tsv": pd.read_csv,  # Additional kwargs for the pandas method are defined below
    ".json": pd.read_json,
    ".jsonl": pd.read_json,  # Additional kwargs for the pandas method are defined below
    ".xml": pd.read_xml,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    # Technically, SQLite files can have any extension (or none), but these are the listed
    # ones on Wikipedia: https://en.wikipedia.org/wiki/SQLite. We can add more robust checking
    # by testing connections if users request it.
    ".sqlite": wrapped_read_sql_query,
    ".sqlite3": wrapped_read_sql_query,
    ".db": wrapped_read_sql_query,
    ".db3": wrapped_read_sql_query,
    ".s3db": wrapped_read_sql_query,
    ".dl3": wrapped_read_sql_query,
    # read_excel supports many file types:
    # https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
    ".xls": pd.read_excel,
    ".xlsx": pd.read_excel,
    ".xlsm": pd.read_excel,
    ".xlsb": pd.read_excel,
    ".odf": pd.read_excel,
    ".ods": pd.read_excel,
    ".odt": pd.read_excel,
}

# Certain extensions leverage a shared method but require additional static kwargs
STATIC_KWARGS_BY_EXTENSION: dict[str, dict[str, Union[str, bool]]] = {".tsv": {"sep": "\t"}, ".jsonl": {"lines": True}}
MISSING_SQL_QUERY_ERROR_MESSAGE = "Loading from a SQLite file requires a SQL query"


def load_pandas_dataset(
    handle: str,
    path: str,
    *,
    pandas_kwargs: Any = None,  # noqa: ANN401
    sql_query: Optional[str],
) -> Union[pd.DataFrame, dict[Union[int, str], pd.DataFrame]]:
    """Creates pandas DataFrame(s) from a file in the dataset

    Args:
        handle: (string) The dataset handle
        path: (string) Path to a file within the dataset
        pandas_kwargs:
            (dict) Optional set of kwargs to pass to the pandas `read_*` method while constructing the DataFrame(s)
        sql_query:
            (string) Argument to be used for SQLite files. Required when reading a SQLite file. See pandas documentation
            for details: https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html

    Returns:
        - dict[int | str, DataFrame] for Excel-like files with multiple sheets
        - A pandas DataFrame for all others

    Raises:
        ValueError: If the file extension is not supported or the file fails to read
    """
    pandas_kwargs = {} if pandas_kwargs is None else pandas_kwargs
    file_extension = os.path.splitext(path)[1]
    read_function = _validate_read_function(file_extension, sql_query)

    # Now that everything has been validated, we can start downloading and processing
    filepath = dataset_download(handle, path)
    try:
        result = read_function(
            *_build_args(read_function, filepath, sql_query),
            **_build_kwargs(file_extension, pandas_kwargs),
        )
    except Exception as e:
        read_error_message = f"Error reading file: {e}"
        raise ValueError(read_error_message) from e

    return result


def _validate_read_function(file_extension: str, sql_query: Optional[str]) -> Callable:
    if file_extension not in SUPPORTED_READ_FUNCTIONS_BY_EXTENSION:
        extension_error_message = (
            f"Unsupported file extension: '{file_extension}'. "
            f"Supported file extensions are: {', '.join(SUPPORTED_READ_FUNCTIONS_BY_EXTENSION.keys())}"
        )
        raise ValueError(extension_error_message) from None

    read_function = SUPPORTED_READ_FUNCTIONS_BY_EXTENSION[file_extension]
    if read_function is wrapped_read_sql_query and not sql_query:
        raise ValueError(MISSING_SQL_QUERY_ERROR_MESSAGE)

    return read_function


def _build_args(read_function: Callable, path: str, sql_query: Optional[str]) -> list:
    # The presence of the sql_query arg was already validated in _validate_read_function
    return [path] if read_function != wrapped_read_sql_query else [sql_query, path]


def _build_kwargs(file_extension: str, pandas_kwargs: Any) -> dict:  # noqa: ANN401
    static_kwargs: dict[str, Any] = (
        {} if file_extension not in STATIC_KWARGS_BY_EXTENSION else STATIC_KWARGS_BY_EXTENSION[file_extension]
    )
    # NOTE: Order matters here as we're letting users override the static args. This may be valuable in the edge case
    # that a CSV/TSV has some other separator than what the extension indicates (we see this in real datasets).
    return {**static_kwargs, **pandas_kwargs}
