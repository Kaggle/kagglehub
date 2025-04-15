import os
import sqlite3
from typing import Any, Callable, Optional, Union

# WARNING: This module is intended to be imported only at runtime, with
# specific error handling to inform users that they need to install the
# appropriate `polars-datasets` extras in order to use these methods. Adding
# any new dependencies here must correspond with an addition to the
# list of `polars-datasets` optional dependencies in pyproject.toml.
import polars as pl

from kagglehub.datasets import PolarsFrameType, dataset_download


# This is a thin wrapper around pl.read_database so we get the connection
# closing for free after we've produced the DataFrame
def wrapped_read_database(sql_query: str, path: str) -> pl.DataFrame:
    with sqlite3.connect(path) as conn:
        df = pl.read_database(sql_query, conn)
        return df


# These are the currently supported read_* functions for the polars adapter.
# Taken from https://docs.pola.rs/api/python/stable/reference/io.html. More can be
# added as demand dictates.
SUPPORTED_READ_FUNCTIONS_BY_EXTENSION: dict[str, Callable] = {
    ".csv": pl.read_csv,
    ".tsv": pl.read_csv,  # Additional kwargs for the polars method are defined below
    ".json": pl.read_json,
    ".jsonl": pl.read_ndjson,
    ".parquet": pl.read_parquet,
    ".feather": pl.read_ipc,
    # Technically, SQLite files can have any extension (or none), but these are the listed
    # ones on Wikipedia: https://en.wikipedia.org/wiki/SQLite. We can add more robust checking
    # by testing connections if users request it.
    ".sqlite": wrapped_read_database,
    ".sqlite3": wrapped_read_database,
    ".db": wrapped_read_database,
    ".db3": wrapped_read_database,
    ".s3db": wrapped_read_database,
    ".dl3": wrapped_read_database,
    # read_excel supports many file types:
    # https://docs.pola.rs/api/python/stable/reference/api/polars.read_excel.html
    ".xls": pl.read_excel,
    ".xlsx": pl.read_excel,
    ".xlsb": pl.read_excel,
    # The default engine for polars Excel reading is "calamine" which uses "the fastexcel module to bind the Rust-based
    # Calamine parser". This does seem to support xlsm despite polars not saying: https://github.com/tafia/calamine
    ".xlsm": pl.read_excel,
    # There's no indication that polars supports odf or odt files like pandas does, so just allow this one for now
    ".ods": pl.read_excel,
}

# These are the currently supported scan_* functions for the polars adapter.
# Taken from https://docs.pola.rs/api/python/stable/reference/io.html. These are the
# only methods that support LazyFrames (the preferred way of using polars) and are
# a subset of the more extensive read_* methods, which produce eager DataFrames.
SUPPORTED_SCAN_FUNCTIONS_BY_EXTENSION: dict[str, Callable] = {
    ".csv": pl.scan_csv,
    ".tsv": pl.scan_csv,  # Additional kwargs for the polars method are defined below
    ".jsonl": pl.scan_ndjson,
    ".parquet": pl.scan_parquet,
    ".feather": pl.scan_ipc,
}


# Certain extensions leverage a shared method but require additional static kwargs
STATIC_KWARGS_BY_EXTENSION: dict[str, dict[str, Union[str, bool]]] = {".tsv": {"separator": "\t"}}
MISSING_SQL_QUERY_ERROR_MESSAGE = "Loading from a SQLite file requires a SQL query"


def load_polars_dataset(
    handle: str,
    path: str,
    *,
    polars_frame_type: PolarsFrameType = PolarsFrameType.LAZY_FRAME,
    polars_kwargs: Any = None,  # noqa: ANN401
    sql_query: Optional[str],
) -> Union[pl.DataFrame, dict[Union[int, str], pl.DataFrame]]:
    """Creates polars LazyFrame(s) or DataFrame(s) from a file in the dataset

    Args:
        handle: (string) The dataset handle
        path: (string) Path to a file within the dataset
        polars_frame_type:
            (PolarsFrameType) Optional control for which Frame to return: LazyFrame or DataFrame. The default is
            PolarsFrameType.LAZY_FRAME.

            PolarsFrameType.LAZY_FRAME: We attempt to use a scan_* method if it's available for the provided file
            extension. Otherwise, we use a read_* method to produce a DataFrame and return the result after calling
            .lazy() on it. This satisfies the requested polars_frame_type as a LazyFrame, but does require loading the
            file in memory.

            PolarsFrameType.DATA_FRAME: We use whatever read_* method corresponds to the provided file extension and
            return the resulting DatFrame.
        polars_kwargs:
            (dict) Optional set of kwargs to pass to the polars `read_*` method while constructing the DataFrame(s)
        sql_query:
            (string) Argument to be used for SQLite files. Required when reading a SQLite file. See polars documentation
            for details: https://docs.pola.rs/api/python/stable/reference/api/polars.read_database.html

    Returns:
        - dict[int | str, LazyFrame] or dict[int | str, DataFrame] for Excel-like files with multiple sheets
        - A polars LazyFrame or DataFrame for all others

    Raises:
        ValueError: If the file extension is not supported or the file fails to read
    """
    polars_kwargs = {} if polars_kwargs is None else polars_kwargs
    file_extension = os.path.splitext(path)[1]
    io_function, io_frame_type = _validate_io_function(file_extension, sql_query, polars_frame_type)

    # Now that everything has been validated, we can start downloading and processing
    filepath = dataset_download(handle, path)
    try:
        result = io_function(
            *_build_args(io_function, filepath, sql_query),
            **_build_kwargs(file_extension, polars_kwargs),
        )
    except Exception as e:
        read_error_message = f"Error reading file: {e}"
        raise ValueError(read_error_message) from e

    # The user requested a LazyFrame, but there's no scan_* method for the file extension. We need to
    # convert the resulting DataFrame(s) to a LazyFrame(s) before we return the result.
    if io_frame_type is PolarsFrameType.DATA_FRAME and polars_frame_type is PolarsFrameType.LAZY_FRAME:
        if isinstance(result, dict):
            for key, value in result.items():
                # The only time a dict can be returned is for Excel files, which only supports read_excel.
                # So all of these *should* already be DataFrames, but we'll just be extra safe.
                if isinstance(value, pl.DataFrame):
                    result[key] = value.lazy()
        else:
            result = result.lazy()

    return result


def _validate_io_function(
    file_extension: str, sql_query: Optional[str], polars_frame_type: PolarsFrameType
) -> tuple[Callable, PolarsFrameType]:
    if file_extension not in SUPPORTED_READ_FUNCTIONS_BY_EXTENSION:
        extension_error_message = (
            f"Unsupported file extension: '{file_extension}'. "
            f"Supported file extensions are: {', '.join(SUPPORTED_READ_FUNCTIONS_BY_EXTENSION.keys())}"
        )
        raise ValueError(extension_error_message) from None

    read_function = SUPPORTED_READ_FUNCTIONS_BY_EXTENSION[file_extension]
    if read_function is wrapped_read_database and not sql_query:
        raise ValueError(MISSING_SQL_QUERY_ERROR_MESSAGE)

    # Now that we've validated all the inputs, we can determine whether we can/should use a scan_* function
    if polars_frame_type is PolarsFrameType.DATA_FRAME or file_extension not in SUPPORTED_SCAN_FUNCTIONS_BY_EXTENSION:
        return (read_function, PolarsFrameType.DATA_FRAME)

    return (SUPPORTED_SCAN_FUNCTIONS_BY_EXTENSION[file_extension], PolarsFrameType.LAZY_FRAME)


def _build_args(read_function: Callable, path: str, sql_query: Optional[str]) -> list:
    # The presence of the sql_query arg was already validated in _validate_read_function
    return [path] if read_function != wrapped_read_database else [sql_query, path]


def _build_kwargs(file_extension: str, polars_kwargs: Any) -> dict:  # noqa: ANN401
    static_kwargs: dict[str, Any] = (
        {} if file_extension not in STATIC_KWARGS_BY_EXTENSION else STATIC_KWARGS_BY_EXTENSION[file_extension]
    )
    # NOTE: Order matters here as we're letting users override the static args. This may be valuable in the edge case
    # that a CSV/TSV has some other separator than what the extension indicates (we see this in real datasets).
    return {**static_kwargs, **polars_kwargs}
