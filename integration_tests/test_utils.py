import unittest

from kagglehub import whoami
from kagglehub.exceptions import UnauthenticatedError

from .utils import unauthenticated


class TestUtils(unittest.TestCase):
    def test_unauthenticated(self) -> None:
        with unauthenticated():
            with self.assertRaises(UnauthenticatedError):
                whoami()
