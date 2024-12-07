from kagglehub.datasets import KaggleDatasetAdapter, load_dataset

dataset = load_dataset(KaggleDatasetAdapter.PANDAS, "unsdsn/world-happiness/versions/1", table_name="2015.csv")
print("Records in dataset: ", dataset.shape)  # noqa: T201
print(dataset.head())  # noqa: T201
