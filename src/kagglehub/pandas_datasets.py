import os
import sqlite3
from typing import Any, Callable, Optional

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
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
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
STATIC_KWARGS_BY_EXTENSION: dict[str, dict[str, str | bool]] = {".tsv": {"sep": "\t"}, ".jsonl": {"lines": True}}

COLUMNS_KWARG_NAME_BY_READ_FUNCTION: dict[Callable, str] = {
    pd.read_csv: "usecols",
    pd.read_parquet: "columns",
    pd.read_feather: "columns",
    pd.read_excel: "usecols",
}

MISSING_SQL_QUERY_ERROR_MESSAGE = "Loading from a SQLite file requires a SQL query"


def load_pandas_dataset(
    handle: str,
    path: str,
    *,
    columns: Optional[list] = None,
    sheet_name: str | int | list | None = 0,
    sql_query: Optional[str] = None,
) -> pd.DataFrame | dict[int | str, pd.DataFrame]:
    """Creates pandas DataFrame(s) from a file in the dataset

    Args:
        handle: (string) The dataset handle
        path: (string) Path to a file within the dataset
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
        - dict[int | str, DataFrame] for Excel-like files with multiple sheets
        - A pandas DataFrame for all others

    Raises:
        ValueError: If the file extension is not supported or the file fails to read
    """
    file_extension = os.path.splitext(path)[1]
    read_function = _validate_read_function(file_extension, sql_query)

    # Now that everything has been validated, we can start downloading and processing
    filepath = dataset_download(handle, path)
    try:
        result = read_function(
            *_build_args(read_function, filepath, sql_query),
            **_build_kwargs(read_function, file_extension, columns, sheet_name),
        )
    except Exception as e:
        read_error_message = f"Error reading file: {e}"
        raise ValueError(read_error_message) from e

    return result


def _validate_read_function(file_extension: str, sql_query: Optional[str] = None) -> Callable:
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


def _build_args(read_function: Callable, path: str, sql_query: Optional[str] | None) -> list:
    return [path] if read_function != wrapped_read_sql_query else [sql_query, path]


def _build_kwargs(
    read_function: Callable,
    file_extension: str,
    columns: Optional[list] = None,
    sheet_name: str | int | list | None = None,
) -> dict:
    additional_kwargs: dict[str, Any] = (
        {} if file_extension not in STATIC_KWARGS_BY_EXTENSION else STATIC_KWARGS_BY_EXTENSION[file_extension]
    )
    if read_function is pd.read_excel:
        additional_kwargs["sheet_name"] = sheet_name

    if read_function in COLUMNS_KWARG_NAME_BY_READ_FUNCTION:
        additional_kwargs[COLUMNS_KWARG_NAME_BY_READ_FUNCTION[read_function]] = columns

    return additional_kwargs
