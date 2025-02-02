import contextvars
import importlib
import inspect
import logging
import pathlib
import re
import sys
import textwrap
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Optional

from kagglehub import registry
from kagglehub.auth import get_username
from kagglehub.cache import get_cached_path
from kagglehub.env import is_in_colab_notebook, is_in_kaggle_notebook
from kagglehub.exceptions import UserCancelledError
from kagglehub.handle import PackageHandle, ResourceHandle, parse_package_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.requirements import VersionedDatasources, read_requirements

logger = logging.getLogger(__name__)

# Current version of the package format used by Kaggle Packages
PACKAGE_VERSION = "0.1.0"
# Name of module field referring to its package version
PACKAGE_VERSION_NAME = "__package_version__"

# Expected name of the kagglehub requirements file
KAGGLEHUB_REQUIREMENTS_FILENAME = "kagglehub_requirements.yaml"


def package_import(
    handle: str, *, force_download: Optional[bool] = False, bypass_confirmation: bool = False
) -> ModuleType:
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
    notebook_path, version = registry.notebook_output_resolver(h, path=None, force_download=force_download)
    if not h.is_versioned() and version:
        h = h.with_version(version)

    init_file_path = pathlib.Path(notebook_path) / "package" / "__init__.py"
    if not init_file_path.exists():
        msg = f"Notebook '{h!s}' is not a Package, missing 'package/__init__.py' file."
        raise ValueError(msg)

    # Unique module name based on handle + downloaded version
    module_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"kagglehub_package_{h.owner}_{h.notebook}_{version}")

    # If this module was already imported and the user didn't re-download, just return it
    if module_name in sys.modules and not force_download:
        return sys.modules[module_name]

    # We're going to run the downloaded code (via `import`) so get user confirmation first.
    if not bypass_confirmation:
        _confirm_import(h)

    # If this module was already imported but now re-downloaded, clear it and any submodules before re-importing
    if module_name in sys.modules:
        logger.info(
            f"Uninstalling existing package module {module_name} before re-importing.", extra={**EXTRA_CONSOLE_BLOCK}
        )
        del sys.modules[module_name]
        submodule_names = [name for name in sys.modules if name.startswith(f"{module_name}.")]
        for name in submodule_names:
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(module_name, init_file_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {init_file_path} as {module_name}."
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def _confirm_import(h: PackageHandle) -> None:
    """Gets user's manual confirmation before importing a package which would execute arbitrary code.

    This is skipped for known safe cases:
    1. Within a Kaggle or Colab environment (already containerized, not user-owned system)
    2. Package owned by the logged-in user (assumed trusted)
    3. Nested Package (assume confirmation already provided)
    """
    if (
        is_in_kaggle_notebook()
        or is_in_colab_notebook()
        or get_username() == h.owner
        # Nested package, we assume confirmation has already been provided for the top-level Package.
        or PackageScope.get() is not None
    ):
        return

    msg = f"""
      WARNING: You are about to execute untrusted code.
      This code could modify your python environment or operating system.

      Review this code at "{h.to_url()}"
      or in your download cache at "{get_cached_path(h)}"

      It is strongly recommended that you run this code within a container
      such as Docker to provide a secure, isolated execution environment.
      See https://www.kaggle.com/docs/packages for more information.
      """

    confirmation = input(f"{textwrap.dedent(msg)}\nDo you want to proceed? (y)es/[no]: ")
    if confirmation.lower() not in ["y", "yes"]:
        raise UserCancelledError()


def get_package_asset_path(path: str) -> pathlib.Path:
    """Returns a path referring to an asset file for use in Kaggle Packages.

    If within a PackageScope context, returns the path relative to it. This should be true
    any time a Package has been imported, whether via `package_import` above or directly.
    Otherwise, assumes we're in an interactive Kaggle Notebook and creating a Package,
    where package data should be written to a staging directory which then gets copied
    into the exported package when the Notebook is saved."""
    scope = PackageScope.get()

    assets_dir = scope.path / "assets" if scope else pathlib.Path("/kaggle/package_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    return assets_dir / path


def import_submodules(package_module: ModuleType) -> list[str]:
    """Complete the import of a Kaggle Package's python module by importing all submodules.

    Only intended for use by Kaggle auto-generated package __init__.py files.

    Imports all (non-underscore-prefixed) sibling .py files as submodules, scopes members
    from their `__all__` onto the parent module (similar to `from X import *`), and decorates
    them with our PackageScope (see that class for more details).

    Args:
        package_module: (ModuleType) The python module of the Kaggle Package.
    Returns:
        The names of all public members which we scoped onto the module (similar to `__all__`).
    """
    package_version = getattr(package_module, PACKAGE_VERSION_NAME, None)
    if package_version != PACKAGE_VERSION:
        msg = f"Unsupported Kaggle Package version: {package_version}"
        raise ValueError(msg)

    all_names: set[str] = set()
    with PackageScope(package_module) as package_scope:
        for filepath in package_scope.path.glob("[!_]*.py"):
            submodule = importlib.import_module(f".{filepath.stem}", package=package_module.__name__)
            package_scope.apply_to_module(submodule)
            for name in submodule.__all__:
                setattr(package_module, name, getattr(submodule, name))
            all_names.update(submodule.__all__)

    return sorted(all_names)


class PackageScope:
    """Captures data about a Kaggle Package. Use as Context Manager to apply scope.

    Only intended for use by Kaggle auto-generated package __init__.py files.

    When scope is applied, certain `kagglehub` calls will utilize the Package info.
    Specifically, downloading a datasource without a version specified will check
    the PackageScope and use the Package's version if it finds a matching entry.
    `kagglehub.get_package_asset_path` also relies on the current scope to pull
    assets related to a package.
    """

    _ctx = contextvars.ContextVar("kagglehub_package_scope", default=None)

    def __init__(self, package_module: ModuleType):
        """Only intended for use by Kaggle auto-generated package __init__.py files."""
        if not package_module.__file__:
            msg = f"Package module '{package_module.__name__}' missing '__file__'."
            raise Exception(msg)

        self.path: pathlib.Path = pathlib.Path(package_module.__file__).parent
        self.datasources: VersionedDatasources = read_requirements(self.path / KAGGLEHUB_REQUIREMENTS_FILENAME)

        self._token_stack: list[contextvars.Token] = []

    def __enter__(self):
        token = PackageScope._ctx.set(self)
        self._token_stack.append(token)
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        token = self._token_stack.pop()
        PackageScope._ctx.reset(token)

    @staticmethod
    def get() -> Optional["PackageScope"]:
        """Gets the currently applied PackageScope, or None if none applied."""
        return PackageScope._ctx.get()

    @staticmethod
    def get_version(h: ResourceHandle) -> Optional[int]:
        """Gets version number for given resource within current PackageScope (if any).

        Returns None if no PackageScope is applied, or if it didn't contain the resource."""
        scope = PackageScope.get()

        return scope.datasources.get(h) if scope else None

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
                # These denylisted entries cause infinite loops.
                elif inspect.isclass(member) and name not in ["__base__", "__class__"]:
                    stack.append(member)
                elif inspect.isfunction(member) or inspect.ismethod(member):
                    setattr(obj, name, decorate(member))
