import pandas as pd

class Dataset:
    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Dataset: ...
