import functools
import inspect
import logging
from importlib import metadata  # type: ignore

KAGGLE_NOTEBOOK_ENV_VAR_NAME = "KAGGLE_KERNEL_RUN_TYPE"
KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME = "KAGGLE_DATA_PROXY_URL"

logger = logging.getLogger(__name__)

try:
    from IPython import get_ipython  # type: ignore

    # Set to `True` if script is running in a Google Colab notebook.
    # Taken from https://stackoverflow.com/a/63519730.
    _is_google_colab = "google.colab" in str(get_ipython())
except (NameError, ModuleNotFoundError):
    _is_google_colab = False


def is_in_colab_notebook() -> bool:
    return _is_google_colab


@functools.cache
def read_kaggle_build_date() -> str:
    build_date_file = "/etc/build_date"
    try:
        with open(build_date_file) as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.warning(f"Build date file {build_date_file} not found in Kaggle Notebook environment.")
        return "unknown"


def search_lib_in_call_stack(lib_name: str) -> str | None:
    """Search the call stack for a given library name and get its information.

    Args:
        lib_name (str):
            The name of the library to search for.
            We use str.startswith so the lib_name must match the exact module name from beginning.

    Returns:
        str: A formatted string f"{lib_name}/{lib_version}" if found, otherwise None.
    """
    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if module and hasattr(module, "__name__"):
            module_name = module.__name__
        else:
            module_name = None

        if module_name is not None and module_name.startswith(lib_name):
            try:
                lib_version = metadata.version(lib_name)
                return f"{lib_name}/{lib_version}"
            except metadata.PackageNotFoundError:
                continue
    return None
