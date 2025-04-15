import kagglehub
from kagglehub import KaggleDatasetAdapter, PolarsFrameType

lazy_dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "wyattowalsh/basketball",
    "nba.sqlite",
    sql_query="SELECT person_id, player_name FROM draft_history",
)
materialized_dataset = lazy_dataset.collect()
print("Records in NBA dataset: ", materialized_dataset.shape)
print(materialized_dataset.head())

dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "robikscube/textocr-text-extraction-from-images-dataset",
    "annot.parquet",
    polars_frame_type=PolarsFrameType.DATA_FRAME,
    polars_kwargs={"columns": ["image_id", "bbox", "points", "area"]},
)

print("Records in OCR dataset: ", dataset.shape)
print(dataset.head())
print(str(dataset.head()))
