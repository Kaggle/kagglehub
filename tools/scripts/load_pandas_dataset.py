import kagglehub
from kagglehub import KaggleDatasetAdapter

dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "wyattowalsh/basketball",
    "nba.sqlite",
    sql_query="SELECT person_id, player_name FROM draft_history",
)
print("Records in dataset: ", dataset.shape)
print(dataset.head())
