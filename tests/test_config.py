import unittest
from test.support.os_helper import EnvironmentVarGuard

from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, DEFAULT_CACHE_FOLDER, get_cache_folder


class TestConfig(unittest.TestCase):
    def test_get_cache_folder_default(self):
        self.assertEqual(DEFAULT_CACHE_FOLDER, get_cache_folder())

    def test_get_cache_folder_environment_var_override(self):
        env = EnvironmentVarGuard()
        env.set(CACHE_FOLDER_ENV_VAR_NAME, "/test_cache")
        with env:
            self.assertEqual("/test_cache", get_cache_folder())
