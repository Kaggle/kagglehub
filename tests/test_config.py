import os
import unittest
from unittest import mock

from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, DEFAULT_CACHE_FOLDER, get_cache_folder


class TestConfig(unittest.TestCase):
    def test_get_cache_folder_default(self):
        self.assertEqual(DEFAULT_CACHE_FOLDER, get_cache_folder())

    @mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: "/test_cache"})
    def test_get_cache_folder_environment_var_override(self):
        self.assertEqual("/test_cache", get_cache_folder())
