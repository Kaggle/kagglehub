import unittest

import kagglehub
from tests.fixtures import BaseTest


class TestKagglehubVersion(BaseTest):
    def test_version_is_set(self):
        self.assertIsNotNone(kagglehub.__version__)
