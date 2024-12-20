import contextvars
import importlib
import inspect
import logging
import pathlib
import re
import sys
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Optional

from kagglehub import registry
from kagglehub.handle import ResourceHandle, parse_package_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.requirements import VersionedDatasources, read_requirements

logger = logging.getLogger(__name__)


class PackageScope:
    """Captures data about a Kaggle Package. Use as Context Manager to apply scope.

    When scope is applied, certain `kagglehub` calls will utilize the Package info.
    Specifically, downloading a datasource without a version specified will check
    the PackageScope and use the Package's version if it finds a matching entry.
    `kagglehub.get_package_asset_path` also relies on the current scope.
    """

    _ctx = contextvars.ContextVar("kagglehub_package_scope", default=None)

    def __init__(self, package_path: pathlib.Path):
        self.path: pathlib.Path = package_path
        self.datasources: VersionedDatasources = read_requirements(package_path / "kagglehub_requirements.yaml")

        self._token_stack: list[contextvars.Token] = []

    def __enter__(self):
        token = PackageScope._ctx.set(self)
        self._token_stack.append(token)
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        token = self._token_stack.pop()
        PackageScope._ctx.reset(token)

    @staticmethod
    def get():
        """Gets the currently applied PackageScope, or None if none applied."""
        return PackageScope._ctx.get()

    def apply_to_module(self, module: ModuleType) -> None:
        """Decorates all functions/methods in the module to apply our scope."""

        def decorate(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                with self:
                    return func(*args, **kwargs)

            return wrapper

        stack: list[Any] = [module]
        while stack:
            obj = stack.pop()
            for name, member in inspect.getmembers(obj):
                # Only decorate things defined within the module.
                if getattr(member, "__module__", None) != module.__name__:
                    continue
                # Recurse on a class to decorate its functions / methods.
                # These denylisted entries cause infinite loops
                elif inspect.isclass(member) and name not in ["__base__", "__class__"]:
                    stack.append(member)
                elif inspect.isfunction(member) or inspect.ismethod(member):
                    setattr(obj, name, decorate(member))


def package_import(handle: str, *, force_download: Optional[bool] = False) -> ModuleType:
    """Download a Kaggle Package and import it.

    A Kaggle Package is a Kaggle Notebook which has exported code to a python package format.

    Args:
        handle: (string) the notebook handle under https://kaggle.com/code.
        force_download: (bool) Optional flag to force download motebook output, even if it's cached.
    Returns:
        The imported python package.
    """
    h = parse_package_handle(handle)

    logger.info(f"Downloading Notebook Output for Package: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    package_path = registry.notebook_output_resolver(h, path=None, force_download=force_download)

    # Unique module name based on handle
    module_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"kagglehub_package_{h!s}")
    init_file_path = pathlib.Path(package_path) / "package" / "__init__.py"

    spec = importlib.util.spec_from_file_location(module_name, init_file_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {init_file_path} as {module_name}."
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def get_package_datasource_version_number(h: ResourceHandle) -> Optional[int]:
    """Gets version number for given resource within current PackageScope.

    Returns None if no PackageScope is applied, or if it didn't contain the resource."""
    scope = PackageScope.get()

    return scope.datasources.get(h) if scope else None


def get_package_asset_path(path: str) -> pathlib.Path:
    """Returns a path referring to an asset file for use in Kaggle Packages.

    If within a PackageScope context, returns the path relative to it. This should be true
    any time a Package has been imported, whether via `package_import` above or directly.
    Otherwise, assumes we're in an interactive Kaggle Notebook and creating a Package,
    where package data should be written to the /kaggle/working dir."""
    scope = PackageScope.get()
    package_path = scope.path if scope else pathlib.Path("/kaggle/working/package")

    asset_path = package_path / "assets" / path
    asset_path.mkdir(parents=True, exist_ok=True)

    return asset_path
