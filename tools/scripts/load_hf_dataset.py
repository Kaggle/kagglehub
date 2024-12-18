import kagglehub
from kagglehub import KaggleDatasetAdapter

dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "unsdsn/world-happiness",
    "2016.csv",
)
print("Records in dataset: ", dataset.num_rows)
print(dataset)
