from enum import Enum


class KaggleDatasetAdapter(Enum):
    HUGGING_FACE = "hugging_face"
    PANDAS = "pandas"
    POLARS = "polars"


# Enum to control how we load the file using polars. LazyFrames are the preferred
# way of working with polars: https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html
# But users should be able to control whether they want a fully loaded DataFrame instead.
class PolarsFrameType(Enum):
    LAZY_FRAME = 1
    DATA_FRAME = 2
