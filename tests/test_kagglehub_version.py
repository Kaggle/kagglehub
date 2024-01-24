import kagglehub
from tests.fixtures import BaseTestCase


class TestKagglehubVersion(BaseTestCase):
    def test_version_is_set(self) -> None:
        self.assertIsNotNone(kagglehub.__version__)
