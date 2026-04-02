import contextlib
import contextvars
import functools
import importlib
import inspect
import logging
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable
from types import ModuleType
from typing import Any, Optional

from kagglesdk.kaggle_env import is_in_kaggle_notebook

from kagglehub import registry
from kagglehub.auth import get_username
from kagglehub.cache import get_cached_path
from kagglehub.env import is_in_colab_notebook
from kagglehub.exceptions import UserCancelledError
from kagglehub.handle import PackageHandle, ResourceHandle, parse_package_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.tracker import VersionedDatasources, read_file

logger = logging.getLogger(__name__)

# Current version of the package format used by Kaggle Packages
PACKAGE_VERSION = "0.1.0"
# Name of module field referring to its package version
PACKAGE_VERSION_NAME = "__package_version__"

# Name of directory within Notebook Output where the Kaggle Package lives
PACKAGE_NOTEBOOK_DIR = "package"

# Relative path within an exported package where asset files are stored
EXPORTED_PACKAGE_ASSETS_DIR = "assets"
# Absolute path for writing asset files during Package creation in Kaggle Notebooks
KAGGLE_NOTEBOOK_ASSETS_STAGING_PATH = "/kaggle/package_assets"

# Handle for the Kaggle Notebook of the Package's Dependency Manager
DEPENDENCY_MANAGER_HANDLE_NAME = "__dependency_manager_notebook__"
# Filename for Dependency Manager Notebook's install script
DEPENDENCY_MANAGER_INSTALL_FILEPATH = "install_requirements.sh"

# Expected name of the kagglehub requirements file
KAGGLEHUB_REQUIREMENTS_FILENAME = "kagglehub_requirements.yaml"


def package_import(
    handle: str, *, force_download: bool | None = False, bypass_confirmation: bool = False
) -> ModuleType:
    """Download a Kaggle Package and import it.

    A Kaggle Package is a Kaggle Notebook which has exported code to a python package format.

    Args:
        handle: (string) the notebook handle under https://kaggle.com/code.
        force_download: (bool) Optional flag to force download motebook output, even if it's cached.
    Returns:
        The imported python module.
    """
    h = parse_package_handle(handle)

    logger.info(f"Downloading Notebook Output for Package: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    notebook_path, version = registry.notebook_output_resolver(h, path=None, force_download=force_download)
    if not h.is_versioned() and version:
        h = h.with_version(version)

    init_file_path = pathlib.Path(notebook_path) / PACKAGE_NOTEBOOK_DIR / "__init__.py"
    if not init_file_path.exists():
        msg = f"Notebook '{h!s}' is not a Package, missing '{PACKAGE_NOTEBOOK_DIR}/__init__.py' file."
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

      Review this code at {h.to_url()}
      or in your download cache at {get_cached_path(h)}

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
    any time a Package has been imported, whether via `package_import()` or directly.
    Otherwise, assumes we're in an interactive Kaggle Notebook and creating a Package,
    where package data should be written to a staging directory which will later get copied
    into the exported package when the Notebook is saved.

    Args:
        path: (str) The relative path of the desired asset file.
    Returns:
        The absolute path (as pathlib.Path) of the desired asset file.
    """
    package_scope = PackageScope.get()
    if package_scope:
        return package_scope.path / EXPORTED_PACKAGE_ASSETS_DIR / path

    if not is_in_kaggle_notebook():
        msg = (
            "Asset paths should only be retrieved within the execution of an existing Kaggle "
            "Package, or during the creation of a new Kaggle Package within a Kaggle Notebook."
        )
        raise ValueError(msg)

    asset_path = pathlib.Path(KAGGLE_NOTEBOOK_ASSETS_STAGING_PATH) / path
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    return asset_path


def _finalize_package_import(package_module: ModuleType) -> list[str]:
    """Complete the import of a Kaggle Package's python module.

    Only intended for use by auto-generated Package __init__.py files.

    Installs the package's Python Dependencies if applicable, and imports all (public) sibling .py
    files as submodules, exposes their public members onto the parent module, and decorates
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

    # Create our PackageScope and apply it while we complete the import process
    with PackageScope(package_module) as package_scope:
        # Install python dependencies as optionally configured via Kaggle Notebook's Dependency Manager
        _install_dependencies(package_module)

        # Import all public submodules and track their public (__all__) members.
        all_names: set[str] = set()
        for filepath in sorted(package_scope.path.glob("[!_]*.py")):
            # Programmatically import the submodule
            submodule = importlib.import_module(f".{filepath.stem}", package=package_module.__name__)

            # Decorate all functions defined by the submodule to apply our PackageScope
            _apply_context_manager_to_module(submodule, package_scope)

            # Expose all public submodule members onto our parent module
            for name in submodule.__all__:
                setattr(package_module, name, getattr(submodule, name))
                all_names.add(name)

        return sorted(all_names)


def _install_dependencies(package_module: ModuleType) -> None:
    """Download and install a Package's dependencies, if applicable.

    Uses the Dependency Manager feature from Kaggle Notebooks."""
    dependency_manager_handle_str = getattr(package_module, DEPENDENCY_MANAGER_HANDLE_NAME, None)
    if not dependency_manager_handle_str:
        return

    package_name = package_module.__name__
    handle = parse_package_handle(dependency_manager_handle_str)
    if handle.owner != "packagemanager":
        msg = f"Package '{package_name}' has Dependency Manager '{handle!s}' not owned by 'packagemanager' user."
        raise ValueError(msg)

    logger.info(f"Downloading Dependency Manager Notebook: {handle.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    path, _ = registry.notebook_output_resolver(handle)
    install_path = pathlib.Path(path) / DEPENDENCY_MANAGER_INSTALL_FILEPATH
    if not install_path.is_file():
        msg = f"Package '{package_name}' has dependencies configured but installer not found at '{install_path!s}'."
        raise ValueError(msg)

    # Run the install script and log the output to a tempfile
    # TODO(b/393170778): Use more platform-agnostic installation logic
    with tempfile.NamedTemporaryFile(
        prefix="kagglehub-package-dependencies-install-", suffix=".txt", delete=False, mode="w+"
    ) as log_file:
        logger.info(
            f"Installing python dependencies for Package '{package_name}', logging progress to '{log_file.name}'."
        )
        log_file.write(
            f"Installing python dependencies for Package '{package_name}' using install script at '{install_path!s}'.\n"
        )
        log_file.flush()

        subprocess.run(  # noqa: S603
            ["/bin/bash", str(install_path)], check=True, text=True, stdout=log_file, stderr=log_file
        )
        log_file.flush()


class PackageScope:
    """Captures data about a Kaggle Package. Use as Context Manager to apply the scope.

    When scope is applied, certain `kagglehub` calls will utilize the Package info.
    Specifically, downloading a datasource without a version specified will check
    the PackageScope and use the Package's version if it finds a matching entry.
    `kagglehub.get_package_asset_path` also relies on the current scope to pull
    assets related to a package.
    """

    # Global context variable tracking the currently active package scope.
    _current_scope_ctx = contextvars.ContextVar[Optional["PackageScope"]]("kagglehub_package_scope", default=None)
    # Global context variable tracking the context token stack for _current_scope_ctx to
    # revert to previous scope (if applicable) when exiting a given scope.
    # b/417707383: Use a ContextVar here to support multithreaded usage of an imported package.
    _token_stack_ctx = contextvars.ContextVar[list[contextvars.Token] | None](
        "kagglehub_package_scope_token_stack", default=None
    )

    def __init__(self, package_module: ModuleType):
        if not hasattr(package_module, "__file__") or not package_module.__file__:
            msg = f"Package module '{package_module.__name__}' missing '__file__'."
            raise ValueError(msg)

        self.package_module: ModuleType = package_module
        self.path: pathlib.Path = pathlib.Path(package_module.__file__).parent
        self.datasources: VersionedDatasources = read_file(self.path / KAGGLEHUB_REQUIREMENTS_FILENAME)

    def __enter__(self):
        token = PackageScope._current_scope_ctx.set(self)

        if PackageScope._token_stack_ctx.get() is None:
            PackageScope._token_stack_ctx.set([])
        PackageScope._token_stack_ctx.get().append(token)

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        token = PackageScope._token_stack_ctx.get().pop()
        PackageScope._current_scope_ctx.reset(token)

    @staticmethod
    def get() -> Optional["PackageScope"]:
        """Gets the currently applied PackageScope, or None if none applied."""
        return PackageScope._current_scope_ctx.get()

    @staticmethod
    def get_version(h: ResourceHandle) -> int | None:
        """Gets version number for given resource within current PackageScope (if any).

        Returns None if no PackageScope is applied, or if it didn't contain the resource."""
        package_scope = PackageScope.get()

        return package_scope.datasources.get(h) if package_scope else None


def _apply_context_manager_to_module(module: ModuleType, context_manager: contextlib.AbstractContextManager) -> None:
    """Decorates all functions/methods in the module to apply the Context Manager.

    Also includes methods defined on classes."""

    # Our main decorator: Wrap function execution in our Context Manager.
    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            with context_manager:
                return func(*args, **kwargs)

        return wrapper

    # Apply the decorator to all functions/methods defined in the module, also in nested classes.
    stack: list[Any] = [module]
    while stack:
        obj = stack.pop()
        for name, member in inspect.getmembers(obj):
            # Only decorate things defined within the module.
            if getattr(member, "__module__", None) != module.__name__:
                continue
            # Recurse on a class to decorate its functions / methods too.
            # These denylisted entries cause infinite loops.
            elif inspect.isclass(member) and name not in ["__base__", "__class__"]:
                stack.append(member)
            elif inspect.isfunction(member) or inspect.ismethod(member):
                setattr(obj, name, decorate(member))
