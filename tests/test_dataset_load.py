import io
import logging
import os
from typing import Any

import polars as pl

from kagglehub.datasets import KaggleDatasetAdapter, PolarsFrameType, dataset_load, logger
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_download_stub as stub
from .server_stubs import serv
from .utils import AUTO_COMPRESSED_FILE_NAME, create_test_cache

DATASET_HANDLE = "lastplacelarry/fake-dataset"
TEST_SPLIT_SIZE = 1
TRAIN_SPLIT_SIZE = 2
INVALID_KWARG = "this_is_not_a_kwarg"
TEXT_FILE = "foo.txt"
EXCEL_FILE = "my-spreadsheet.xlsx"
SHAPES_COLUMNS = ["shape", "degrees", "sides", "color", "date"]
SHAPES_ROW_COUNT = 3
SHAPES_COLUMNS_SUBSET = ["shape", "sides", "date"]


class TestLoadHfDataset(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _load_hf_dataset_with_invalid_file_type_and_assert_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            dataset_load(
                KaggleDatasetAdapter.HUGGING_FACE,
                DATASET_HANDLE,
                TEXT_FILE,
            )
        self.assertIn(f"Unsupported file extension: '{os.path.splitext(TEXT_FILE)[1]}'", str(cm.exception))

    def _load_hf_dataset_with_other_loader_kwargs_and_assert_warning(self) -> None:
        output_stream = io.StringIO()
        handler = logging.StreamHandler(output_stream)
        logger.addHandler(handler)
        dataset_load(
            KaggleDatasetAdapter.HUGGING_FACE,
            DATASET_HANDLE,
            AUTO_COMPRESSED_FILE_NAME,
            polars_frame_type=PolarsFrameType.LAZY_FRAME,
            polars_kwargs={},
        )
        captured_output = output_stream.getvalue()
        self.assertIn(
            "polars_frame_type, polars_kwargs are invalid for KaggleDatasetAdapter.HUGGING_FACE", captured_output
        )

    def _load_hf_dataset_with_multiple_tables_and_assert_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            dataset_load(
                KaggleDatasetAdapter.HUGGING_FACE, DATASET_HANDLE, EXCEL_FILE, pandas_kwargs={"sheet_name": None}
            )
        self.assertIn(
            "Loading a Huggingface dataset requires the production of exactly one DataFrame", str(cm.exception)
        )

    def _load_hf_dataset_and_assert_loaded(self) -> None:
        hf_dataset = dataset_load(KaggleDatasetAdapter.HUGGING_FACE, DATASET_HANDLE, AUTO_COMPRESSED_FILE_NAME)
        self.assertEqual(SHAPES_ROW_COUNT, hf_dataset.num_rows)
        self.assertEqual(SHAPES_COLUMNS, hf_dataset.column_names)
        self.assertIn("hugging_face_data_loader", stub.shared_data.last_download_user_agent)

    def _load_hf_dataset_with_valid_kwargs_and_assert_loaded(self) -> None:
        hf_dataset = dataset_load(
            KaggleDatasetAdapter.HUGGING_FACE,
            DATASET_HANDLE,
            AUTO_COMPRESSED_FILE_NAME,
            hf_kwargs={"preserve_index": True},
        )
        self.assertEqual(SHAPES_ROW_COUNT, hf_dataset.num_rows)
        self.assertEqual([*SHAPES_COLUMNS, "__index_level_0__"], hf_dataset.column_names)

    def _load_hf_dataset_with_invalid_kwargs_and_assert_raises(self) -> None:
        with self.assertRaises(TypeError) as cm:
            dataset_load(
                KaggleDatasetAdapter.HUGGING_FACE,
                DATASET_HANDLE,
                AUTO_COMPRESSED_FILE_NAME,
                hf_kwargs={INVALID_KWARG: 777},
            )
        self.assertIn(INVALID_KWARG, str(cm.exception))

    def _load_hf_dataset_with_splits_and_assert_loaded(self) -> None:
        hf_dataset = dataset_load(KaggleDatasetAdapter.HUGGING_FACE, DATASET_HANDLE, AUTO_COMPRESSED_FILE_NAME)
        dataset_splits = hf_dataset.train_test_split(
            test_size=TEST_SPLIT_SIZE, train_size=TRAIN_SPLIT_SIZE, shuffle=False
        )
        for split_name, dataset in dataset_splits.items():
            self.assertEqual(TEST_SPLIT_SIZE if split_name == "test" else TRAIN_SPLIT_SIZE, dataset.num_rows)
            self.assertEqual(SHAPES_COLUMNS, dataset.column_names)

    def test_hf_dataset_with_other_loader_kwargs_prints_warning(self) -> None:
        with create_test_cache():
            self._load_hf_dataset_with_other_loader_kwargs_and_assert_warning()

    def test_hf_dataset_with_invalid_file_type_raises(self) -> None:
        with create_test_cache():
            self._load_hf_dataset_with_invalid_file_type_and_assert_raises()

    def test_hf_dataset_with_multiple_tables_raises(self) -> None:
        with create_test_cache():
            self._load_hf_dataset_with_multiple_tables_and_assert_raises()

    def test_hf_dataset_succeeds(self) -> None:
        with create_test_cache():
            self._load_hf_dataset_and_assert_loaded()

    def test_hf_dataset_with_valid_kwargs_succeeds(self) -> None:
        with create_test_cache():
            self._load_hf_dataset_with_valid_kwargs_and_assert_loaded()

    def test_hf_dataset_with_invalid_kwargs_raises(self) -> None:
        with create_test_cache():
            self._load_hf_dataset_with_invalid_kwargs_and_assert_raises()

    def test_hf_dataset_with_splits_succeeds(self) -> None:
        with create_test_cache():
            self._load_hf_dataset_with_splits_and_assert_loaded()


class TestLoadPandasDataset(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _load_pandas_dataset_with_invalid_file_type_and_assert_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            dataset_load(
                KaggleDatasetAdapter.PANDAS,
                DATASET_HANDLE,
                TEXT_FILE,
            )
        self.assertIn(f"Unsupported file extension: '{os.path.splitext(TEXT_FILE)[1]}'", str(cm.exception))

    def _load_pandas_dataset_with_other_loader_kwargs_and_assert_warning(self) -> None:
        output_stream = io.StringIO()
        handler = logging.StreamHandler(output_stream)
        logger.addHandler(handler)
        dataset_load(
            KaggleDatasetAdapter.PANDAS,
            DATASET_HANDLE,
            AUTO_COMPRESSED_FILE_NAME,
            hf_kwargs={},
            polars_frame_type=PolarsFrameType.LAZY_FRAME,
            polars_kwargs={},
        )
        captured_output = output_stream.getvalue()
        self.assertIn(
            "hf_kwargs, polars_frame_type, polars_kwargs are invalid for KaggleDatasetAdapter.PANDAS", captured_output
        )

    def _load_pandas_simple_dataset_and_assert_loaded(
        self,
        file_extension: str,
        pandas_kwargs: Any = None,  # noqa: ANN401
    ) -> None:
        df = dataset_load(
            KaggleDatasetAdapter.PANDAS, DATASET_HANDLE, f"shapes.{file_extension}", pandas_kwargs=pandas_kwargs
        )
        self.assertEqual(SHAPES_ROW_COUNT, len(df))
        self.assertEqual(SHAPES_COLUMNS, list(df.columns))

    def _load_pandas_sqlite_dataset_and_assert_loaded(self) -> None:
        df = dataset_load(KaggleDatasetAdapter.PANDAS, DATASET_HANDLE, "shapes.db", sql_query="SELECT * FROM shapes")
        self.assertEqual(SHAPES_ROW_COUNT, len(df))
        self.assertEqual(SHAPES_COLUMNS, list(df.columns))

    def _load_pandas_dataset_with_multiple_tables_and_assert_loaded(self) -> None:
        result = dataset_load(
            KaggleDatasetAdapter.PANDAS, DATASET_HANDLE, EXCEL_FILE, pandas_kwargs={"sheet_name": None}
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(["Cars", "Animals"], list(result.keys()))

    def _load_pandas_dataset_with_valid_kwargs_and_assert_loaded(self) -> None:
        expected_columns = ["degrees"]
        df = dataset_load(
            KaggleDatasetAdapter.PANDAS,
            DATASET_HANDLE,
            AUTO_COMPRESSED_FILE_NAME,
            pandas_kwargs={"usecols": expected_columns},
        )
        self.assertEqual(SHAPES_ROW_COUNT, len(df))
        self.assertEqual(expected_columns, list(df.columns))
        self.assertIn("pandas_data_loader", stub.shared_data.last_download_user_agent)

    def _load_pandas_dataset_with_invalid_kwargs_and_assert_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            dataset_load(
                KaggleDatasetAdapter.PANDAS,
                DATASET_HANDLE,
                AUTO_COMPRESSED_FILE_NAME,
                pandas_kwargs={INVALID_KWARG: 777},
            )
        self.assertIn(INVALID_KWARG, str(cm.exception))

    def test_pandas_dataset_with_invalid_file_type_raises(self) -> None:
        with create_test_cache():
            self._load_pandas_dataset_with_invalid_file_type_and_assert_raises()

    def test_pandas_dataset_with_other_loader_kwargs_prints_warning(self) -> None:
        with create_test_cache():
            self._load_pandas_dataset_with_other_loader_kwargs_and_assert_warning()

    def test_pandas_dataset_with_multiple_tables_succeeds(self) -> None:
        with create_test_cache():
            self._load_pandas_dataset_with_multiple_tables_and_assert_loaded()

    def test_pandas_dataset_with_valid_kwargs_succeeds(self) -> None:
        with create_test_cache():
            self._load_pandas_dataset_with_valid_kwargs_and_assert_loaded()

    def test_pandas_dataset_with_invalid_kwargs_raises(self) -> None:
        with create_test_cache():
            self._load_pandas_dataset_with_invalid_kwargs_and_assert_raises()

    def test_pandas_simple_dataset_succeeds(self) -> None:
        # This would be better as a parameterized test, but that doesn't work for subclasses:
        # https://docs.pytest.org/en/stable/how-to/unittest.html#pytest-features-in-unittest-testcase-subclasses
        test_cases = [
            ("csv", {}),
            ("tsv", {}),
            ("xml", {"parser": "etree"}),  # The default parser requires a pip install
            ("parquet", {}),
            ("feather", {}),
            ("json", {}),
            ("jsonl", {}),
        ]
        for test_case in test_cases:
            with create_test_cache():
                self._load_pandas_simple_dataset_and_assert_loaded(test_case[0], test_case[1])

    def test_pandas_sqlite_dataset_succeeds(self) -> None:
        with create_test_cache():
            self._load_pandas_sqlite_dataset_and_assert_loaded()


class TestLoadPolarsDataset(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _load_polars_dataset_with_invalid_file_type_and_assert_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            dataset_load(
                KaggleDatasetAdapter.POLARS,
                DATASET_HANDLE,
                TEXT_FILE,
            )
        self.assertIn(f"Unsupported file extension: '{os.path.splitext(TEXT_FILE)[1]}'", str(cm.exception))

    def _load_polars_dataset_with_other_loader_kwargs_and_assert_warning(self) -> None:
        output_stream = io.StringIO()
        handler = logging.StreamHandler(output_stream)
        logger.addHandler(handler)
        dataset_load(
            KaggleDatasetAdapter.POLARS,
            DATASET_HANDLE,
            AUTO_COMPRESSED_FILE_NAME,
            pandas_kwargs={},
            hf_kwargs={},
        )
        captured_output = output_stream.getvalue()
        self.assertIn("pandas_kwargs, hf_kwargs are invalid for KaggleDatasetAdapter.POLARS", captured_output)

    def _load_polars_simple_dataset_and_assert_loaded(
        self,
        file_extension: str,
        polars_frame_type: PolarsFrameType,
    ) -> None:
        polars_frame = dataset_load(
            KaggleDatasetAdapter.POLARS, DATASET_HANDLE, f"shapes.{file_extension}", polars_frame_type=polars_frame_type
        )
        if polars_frame_type is PolarsFrameType.LAZY_FRAME:
            self.assertIsInstance(polars_frame, pl.LazyFrame)
            # Materialize the results to check rows and columns
            polars_frame = polars_frame.collect()
        else:
            self.assertIsInstance(polars_frame, pl.DataFrame)
        self.assertEqual(SHAPES_ROW_COUNT, len(polars_frame))
        self.assertEqual(SHAPES_COLUMNS, list(polars_frame.columns))

    def _load_polars_sqlite_dataset_and_assert_loaded(self) -> None:
        df = dataset_load(
            KaggleDatasetAdapter.POLARS,
            DATASET_HANDLE,
            "shapes.db",
            polars_frame_type=PolarsFrameType.DATA_FRAME,
            sql_query="SELECT * FROM shapes",
        )
        self.assertEqual(SHAPES_ROW_COUNT, len(df))
        self.assertEqual(SHAPES_COLUMNS, list(df.columns))

    def _load_polars_columns_subset_and_assert_loaded(self) -> None:
        df = dataset_load(
            KaggleDatasetAdapter.POLARS,
            DATASET_HANDLE,
            "shapes.csv",
            polars_frame_type=PolarsFrameType.DATA_FRAME,
            polars_kwargs={"columns": SHAPES_COLUMNS_SUBSET},
        )
        lf = dataset_load(
            KaggleDatasetAdapter.POLARS,
            DATASET_HANDLE,
            "shapes.csv",
        )
        self.assertEqual(str(df), str(lf.select(SHAPES_COLUMNS_SUBSET).collect()))

    def _load_polars_dataset_with_multiple_tables_and_assert_loaded(self, polars_frame_type: PolarsFrameType) -> None:
        result = dataset_load(
            KaggleDatasetAdapter.POLARS,
            DATASET_HANDLE,
            EXCEL_FILE,
            polars_frame_type=polars_frame_type,
            # sheet_id of 0 returns all sheets. This differs from pandas where sheet_name of None returns all
            polars_kwargs={"sheet_id": 0},
        )

        expected_type = pl.LazyFrame if polars_frame_type is PolarsFrameType.LAZY_FRAME else pl.DataFrame
        self.assertTrue(all(isinstance(v, expected_type) for v in result.values()))
        self.assertEqual(["Cars", "Animals"], list(result.keys()))

    def _load_polars_dataset_with_valid_kwargs_and_assert_loaded(self) -> None:
        expected_columns = ["degrees"]
        df = dataset_load(
            KaggleDatasetAdapter.POLARS,
            DATASET_HANDLE,
            AUTO_COMPRESSED_FILE_NAME,
            polars_frame_type=PolarsFrameType.DATA_FRAME,
            polars_kwargs={"columns": expected_columns},
        )
        self.assertEqual(SHAPES_ROW_COUNT, len(df))
        self.assertEqual(expected_columns, list(df.columns))
        self.assertIn("polars_data_loader", stub.shared_data.last_download_user_agent)

    def _load_polars_dataset_with_invalid_kwargs_and_assert_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            dataset_load(
                KaggleDatasetAdapter.POLARS,
                DATASET_HANDLE,
                AUTO_COMPRESSED_FILE_NAME,
                polars_kwargs={INVALID_KWARG: 777},
            )
        self.assertIn(INVALID_KWARG, str(cm.exception))

    def test_polars_dataset_with_invalid_file_type_raises(self) -> None:
        with create_test_cache():
            self._load_polars_dataset_with_invalid_file_type_and_assert_raises()

    def test_polars_dataset_with_other_loader_kwargs_prints_warning(self) -> None:
        with create_test_cache():
            self._load_polars_dataset_with_other_loader_kwargs_and_assert_warning()

    def test_polars_dataset_with_multiple_tables_succeeds(self) -> None:
        with create_test_cache():
            self._load_polars_dataset_with_multiple_tables_and_assert_loaded(PolarsFrameType.LAZY_FRAME)
            self._load_polars_dataset_with_multiple_tables_and_assert_loaded(PolarsFrameType.DATA_FRAME)

    def test_polars_dataset_with_valid_kwargs_succeeds(self) -> None:
        with create_test_cache():
            self._load_polars_dataset_with_valid_kwargs_and_assert_loaded()

    def test_polars_dataset_with_invalid_kwargs_raises(self) -> None:
        with create_test_cache():
            self._load_polars_dataset_with_invalid_kwargs_and_assert_raises()

    def test_polars_simple_dataset_succeeds(self) -> None:
        # This would be better as a parameterized test, but that doesn't work for subclasses:
        # https://docs.pytest.org/en/stable/how-to/unittest.html#pytest-features-in-unittest-testcase-subclasses
        test_cases = [
            "csv",
            "tsv",
            "parquet",
            "feather",
            "json",
            "jsonl",
        ]
        for test_case in test_cases:
            with create_test_cache():
                self._load_polars_simple_dataset_and_assert_loaded(test_case, PolarsFrameType.LAZY_FRAME)
                self._load_polars_simple_dataset_and_assert_loaded(test_case, PolarsFrameType.DATA_FRAME)

    def test_polars_sqlite_dataset_succeeds(self) -> None:
        with create_test_cache():
            self._load_polars_sqlite_dataset_and_assert_loaded()

    def test_polars_columns_subset_succeeds(self) -> None:
        with create_test_cache():
            self._load_polars_columns_subset_and_assert_loaded()
