import unittest

from kagglesdk.datasets.types.dataset_enums import DatabundleVersionStatus
from kagglesdk.models.types.model_enums import ModelFramework

from kagglehub.enum import enum_to_str, to_enum


class TestEnumUtils(unittest.TestCase):
    def test_to_enum(self) -> None:
        # Enum values WITHOUT class prefix
        self.assertEqual(DatabundleVersionStatus.READY, to_enum(DatabundleVersionStatus, "ready"))
        self.assertEqual(
            DatabundleVersionStatus.INDIVIDUAL_BLOBS_COMPRESSED,
            to_enum(DatabundleVersionStatus, "individualBlobsCompressed"),
        )
        # Enum values WITH class prefix
        self.assertEqual(ModelFramework.MODEL_FRAMEWORK_JAX, to_enum(ModelFramework, "jax"))
        # self.assertEqual(ModelFramework.MODEL_FRAMEWORK_TENSOR_FLOW_2, to_enum(ModelFramework, "tensorFlow2"))
        self.assertEqual(ModelFramework.MODEL_FRAMEWORK_PY_TORCH, to_enum(ModelFramework, "pyTorch"))
        # Special case for PyTorch. "pyTorch" and "pytorch" are accepted
        self.assertEqual(ModelFramework.MODEL_FRAMEWORK_PY_TORCH, to_enum(ModelFramework, "pytorch"))

    def test_enum_to_str(self) -> None:
        # Enum WITHOUT class prefix
        self.assertEqual("ready", enum_to_str(DatabundleVersionStatus.READY))
        # Enum values WITH class prefix
        self.assertEqual("jax", enum_to_str(ModelFramework.MODEL_FRAMEWORK_JAX))
        self.assertEqual("tensorFlow2", enum_to_str(ModelFramework.MODEL_FRAMEWORK_TENSOR_FLOW_2))
        self.assertEqual("pyTorch", enum_to_str(ModelFramework.MODEL_FRAMEWORK_PY_TORCH))
