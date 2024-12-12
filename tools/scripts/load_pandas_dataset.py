from kagglehub.datasets import KaggleDatasetAdapter, load_dataset

dataset = load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "wyattowalsh/basketball",
    "nba.sqlite",
    sql_query="SELECT person_id, player_name FROM draft_history",
)
print("Records in dataset: ", dataset.shape)  # noqa: T201
print(dataset.head())  # noqa: T201
