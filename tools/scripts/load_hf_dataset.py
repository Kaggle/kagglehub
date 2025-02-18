import kagglehub
from kagglehub import KaggleDatasetAdapter

dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.HUGGING_FACE,
    "unsdsn/world-happiness",
    "2016.csv",
)
print("Records in dataset: ", dataset.num_rows)
print(dataset)
