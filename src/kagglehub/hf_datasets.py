from typing import Any, Optional

# WARNING: This module is intended to be imported only at runtime, with
# specific error handling to inform users that they need to install the
# appropriate `hf-datasets` extras in order to use these methods. Adding
# any new dependencies here must correspond with an addition to the
# list of `hf-datasets` optional dependencies in pyproject.toml.
import tensorflow_datasets as tfds
from datasets import Dataset, DatasetDict, Value
from tensorflow_datasets.core import DatasetBuilder
from tensorflow_datasets.core.data_sources import array_record

from kagglehub.handle import DatasetHandle


def load_hf_dataset(
    handle: DatasetHandle,
    train_split_percent: float,
    path: Optional[str] = None,
) -> DatasetDict:
    croissant_url = f"https://www.kaggle.com/datasets/{handle.owner}/{handle.dataset}/croissant/download"
    if handle.is_versioned():
        croissant_url += f"?datasetVersionNumber={handle.version}"
    builder = tfds.dataset_builders.CroissantBuilder(
        jsonld=croissant_url,
        file_format="array_record",
    )

    # The builder above may have multiple configs, but will only use the 1st one to download_and_prepare
    # If the dataset has multiple tables, we need a new builder targeting the given table_name
    if path is not None:
        builder = tfds.dataset_builders.CroissantBuilder(
            jsonld=croissant_url,
            file_format="array_record",
            config=next(
                filter(
                    lambda x: x.name == tfds.core.utils.conversion_utils.to_tfds_name(path),
                    builder.BUILDER_CONFIGS,
                ),
                None,
            ),
        )
    builder.download_and_prepare()
    ds = builder.as_data_source()

    # Split the default ArrayRecordDataSource into train/test splits based on the user-specified split
    train_split = Dataset.from_generator(
        _generate_examples,
        gen_kwargs={
            "data_source": ds,
            "builder": builder,
            "split_name": "train",
            "train_split_percent": train_split_percent,
        },
        split="train",
    )
    test_split = Dataset.from_generator(
        _generate_examples,
        gen_kwargs={
            "data_source": ds,
            "builder": builder,
            "split_name": "test",
            "train_split_percent": train_split_percent,
        },
        split="test",
    )
    dataset = DatasetDict({"train": train_split, "test": test_split})

    # The CroissantBuilder uses the @id from the JSON-LD to populate column names. For Kaggle, that means values like
    # my-csv.csv/column_1, my-csv.csv/column_2, etc. in order to avoid column name collisions across tables. We only
    # load one table at a time here, so we should strip the file names so that end users can rename columns as they'd
    # expect (i.e. referencing 'column_1' instead of 'my-csv.csv/column_1').
    for feature_name, feature_value in train_split.features.items():
        # The CroissantBuilder reads strings in as binary, but they should be strings
        if feature_value.dtype == "binary":
            dataset = dataset.cast_column(feature_name, Value(dtype="string"))
        dataset = dataset.rename_column(feature_name, feature_name.split("/")[-1])

    return dataset


def _generate_examples(
    data_source: array_record.ArrayRecordDataSource,
    builder: DatasetBuilder,
    split_name: str,
    train_split_percent: float,
) -> Any:  # noqa: ANN401
    """
    Generator function to yield examples based on the train_split_percent.
    """
    total_examples = builder.info.splits["default"].num_examples
    split_index = int(total_examples * train_split_percent)
    data_source_iter = iter(data_source["default"])

    if split_name == "train":
        for _ in range(split_index):
            yield next(data_source_iter)
    elif split_name == "test":
        # Skip the training examples
        for _ in range(split_index):
            next(data_source_iter)
        # Yield the remaining examples, if any
        for _ in range(total_examples - split_index):
            yield next(data_source_iter)
