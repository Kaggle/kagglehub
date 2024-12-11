from typing import Optional

# WARNING: This module is intended to be imported only at runtime, with
# specific error handling to inform users that they need to install the
# appropriate `hf-datasets` extras in order to use these methods. Adding
# any new dependencies here must correspond with an addition to the
# list of `hf-datasets` optional dependencies in pyproject.toml.
from datasets import Dataset, DatasetDict

from kagglehub.pandas_datasets import load_pandas_dataset

MULTIPLE_SHEETS_ERROR_MESSAGE = "Excel-based datasets must specify a single sheet"


def load_hf_dataset(
    handle: str,
    path: str,
    train_split_percent: float = 0.8,
    *,
    columns: Optional[list] = None,
    sheet_name: str | int | list | None = 0,
    sql_query: Optional[str] = None,
) -> DatasetDict:
    """Load a Kaggle Dataset into a Huggingface DatasetDict with test/train splits

    Args:
        handle: (string) The dataset handle
        path: (string) Path to a file within the dataset
        train_split_percent: (float) Optional split for train vs test data, defaults to 0.8
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
        A DatasetDict with test and train splits according to the provided train_split_percent
    """
    # These argument types produce a dictionary of DataFrames from pandas, but we need a single DataFrame
    # to perform splits and loading into the DatasetDict. We validate upfront to respond to the user
    # before doing any loading.
    if sheet_name is None or isinstance(sheet_name, list):
        raise ValueError(MULTIPLE_SHEETS_ERROR_MESSAGE)

    result = load_pandas_dataset(handle, path, columns=columns, sheet_name=sheet_name, sql_query=sql_query)

    # This shouldn't happen since we validated the sheet_name argument already, but we can check again to
    # be robust and enable type safety in the else block.
    if isinstance(result, dict):
        raise ValueError(MULTIPLE_SHEETS_ERROR_MESSAGE)
    else:
        # Split the DataFrame into train/test splits based on the user-specified split
        split_index = int(result.shape[0] * train_split_percent)
        train_split = Dataset.from_pandas(result[:split_index])
        test_split = Dataset.from_pandas(result[split_index:])
        dataset = DatasetDict({"train": train_split, "test": test_split})

    return dataset
