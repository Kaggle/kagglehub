from kagglehub.datasets import KaggleDatasetAdapter, load_dataset

dataset = load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "wyattowalsh/basketball",
    table_name="draft_history.csv",
    train_split_percent=0.6,
)
print("Records in dataset: ", dataset.num_rows)  # noqa: T201
print(dataset)  # noqa: T201
