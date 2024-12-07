from typing import Optional

# WARNING: This module is intended to be imported only at runtime, with
# specific error handling to inform users that they need to install the
# appropriate `hf-datasets` extras in order to use these methods. Adding
# any new dependencies here must correspond with an addition to the
# list of `hf-datasets` optional dependencies in pyproject.toml.
from datasets import Dataset, DatasetDict, Value

from kagglehub.handle import DatasetHandle
from kagglehub.pandas_datasets import load_pandas_dataset


def load_hf_dataset(
    handle: DatasetHandle,
    train_split_percent: float,
    table_name: Optional[str] = None,
) -> DatasetDict:
    record_set_df = load_pandas_dataset(handle, table_name)

    # Split the DataFrame into train/test splits based on the user-specified split
    split_index = int(record_set_df.shape[0] * train_split_percent)
    train_split = Dataset.from_pandas(record_set_df[:split_index])
    test_split = Dataset.from_pandas(record_set_df[split_index:])
    dataset = DatasetDict({"train": train_split, "test": test_split})

    return dataset
