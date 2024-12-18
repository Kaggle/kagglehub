from kagglehub.datasets import KaggleDatasetAdapter, load_dataset

dataset = load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "unsdsn/world-happiness",
    "2016.csv",
)
print("Records in dataset: ", dataset.num_rows)
print(dataset)
