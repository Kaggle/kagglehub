from typing import Optional

# WARNING: This module is intended to be imported only at runtime, with
# specific error handling to inform users that they need to install the
# appropriate `pandas-datasets` extras in order to use these methods. Adding
# any new dependencies here must correspond with an addition to the
# list of `pandas-datasets` optional dependencies in pyproject.toml.
import mlcroissant as mlc
import pandas as pd

from kagglehub.handle import DatasetHandle


def load_pandas_dataset(
    handle: DatasetHandle,
    table_name: Optional[str] = None,
) -> pd.DataFrame:
    croissant_url = f"https://www.kaggle.com/datasets/{handle.owner}/{handle.dataset}/croissant/download"
    if handle.is_versioned():
        croissant_url += f"?datasetVersionNumber={handle.version}"
    croissant_dataset = mlc.Dataset(croissant_url)

    if len(croissant_dataset.metadata.record_sets) > 0 and table_name is None:
        raise NameError(handle.to_url() + " has multiple record sets, so the path to a specific one must be provided.")

    # When Kaggle auto-generates Croissant, we only use file paths to disambiguate tables if there are collisions
    # on the file names. We'll take the table_name provided by the user to try and locate a unique record set. If
    # multiple or none match, we can provide the IDs in an error message to make it easy for them to try again.
    record_set_ids = [rs.uuid for rs in croissant_dataset.metadata.record_sets]
    matching_record_set_ids = list(filter(lambda id: id.endswith(table_name), record_set_ids))
    matches = len(matching_record_set_ids)
    if matches != 1:
        multiple_matches_error_message = (
            f"{matches} matches found for {table_name}. Please try one of {', '.join(record_set_ids)}"
        )
        raise NameError(multiple_matches_error_message)
    record_set_df = pd.DataFrame(croissant_dataset.records(matching_record_set_ids[0]))

    # mlcroissant uses the @id from the JSON-LD to populate column names. For Kaggle, that means values like
    # my-csv.csv/column_1, my-csv.csv/column_2, etc. in order to avoid column name collisions across tables. We only
    # load one table at a time, so we should strip the file names so that end users can interact with columns as they'd
    # expect (i.e. referencing 'column_1' instead of 'my-csv.csv/column_1').
    for col in record_set_df.columns:
        # Tables can be deeply nested in the dataset, so pull the column name from the end
        col_name = col.split("/")[-1]
        record_set_df = record_set_df.rename(columns={col: col_name})

        # mlcroissant reads text in as bytes, but we want them as strings
        # If desired, we can implement an option for this later
        if record_set_df[col_name].dtype == "object" and any(
            isinstance(val, bytes) for val in record_set_df[col_name].values
        ):
            record_set_df[col_name] = record_set_df[col_name].astype(str)

    return record_set_df
