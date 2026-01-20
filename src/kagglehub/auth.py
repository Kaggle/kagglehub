import getpass
import io
import logging
from collections.abc import Generator
from contextlib import contextmanager

from kagglesdk.models.types.model_api_service import ApiGetModelRequest

from kagglehub.clients import build_kaggle_client
from kagglehub.config import get_kaggle_credentials, set_kaggle_api_token
from kagglehub.exceptions import UnauthenticatedError

_logger = logging.getLogger(__name__)

INVALID_CREDENTIALS_ERROR = 401

NOTEBOOK_LOGIN_TOKEN_HTML_START = """<center> <img
src=https://www.kaggle.com/static/images/site-logo.png
alt='Kaggle'> <br> Create an API token from <a
href="https://www.kaggle.com/settings/account" target="_blank">your Kaggle
settings page</a> and paste it below. <br> </center>"""

NOTEBOOK_LOGIN_TOKEN_HTML_END = """
<b>Thank You</b></center>"""


@contextmanager
def _capture_logger_output() -> Generator[io.StringIO, None, None]:
    """Capture output that is logged using the logger.

    Example:
    ```py
    >>> output = capture_logger_output()
    >>> logger.info("This is an info message")
    >>> logger.error("This is an error message")
    >>> print(output)
    ```
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    _logger.addHandler(handler)
    try:
        yield buffer
    finally:
        _logger.removeHandler(handler)


def _is_in_notebook() -> bool:
    """Return `True` if code is executed in a notebook (Jupyter, Kaggle Notebook, Colab, QTconsole).

    Taken from https://stackoverflow.com/a/39662359.
    Adapted to make it work with Google colab as well.
    """
    try:
        from IPython import get_ipython  # type: ignore # noqa: PLC0415

        shell_class = get_ipython().__class__
        for parent_class in shell_class.__mro__:  # e.g. "is subclass of"
            if parent_class.__name__ == "ZMQInteractiveShell":
                return True  # Jupyter notebook, Kaggle Noteboo, Google colab or qtconsole
        return False
    except (NameError, ModuleNotFoundError):
        return False  # Probably standard Python interpreter


def _notebook_login(validate_credentials: bool) -> None:  # noqa: FBT001
    """Prompt the user for their Kaggle token and save it in a widget (Jupyter or Colab)."""
    library_error = "You need the `ipywidgets` module: `pip install ipywidgets`."
    try:
        from IPython.display import display  # type: ignore # noqa: PLC0415
        from ipywidgets import widgets  # type: ignore # noqa: PLC0415
    except ImportError:
        raise ImportError(library_error)  # noqa: B904

    box_layout = widgets.Layout(display="flex", flex_flow="column", align_items="center", width="50%")

    token_widget = widgets.Password(description="API Token:")
    login_button = widgets.Button(description="Login")

    login_token_widget = widgets.VBox(
        [
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_START),
            token_widget,
            login_button,
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_END),
        ],
        layout=box_layout,
    )
    display(login_token_widget)

    def on_click_login_button(_: str) -> None:
        token = token_widget.value
        # Erase token and clear value to make sure it's not saved in the notebook.
        token_widget.value = ""

        # Hide inputs
        login_token_widget.children = [widgets.Label("Connecting...")]
        try:
            with _capture_logger_output() as captured:
                # Set Kaggle credentials
                set_kaggle_api_token(token)

                # Validate credentials if necessary
                if validate_credentials is True:
                    _validate_credentials_helper()
            message = captured.getvalue()
        except Exception as error:
            message = str(error)
        # Print result (success message or error)
        login_token_widget.children = [widgets.Label(line) for line in message.split("\n") if line.strip()]

    login_button.on_click(on_click_login_button)


def _validate_credentials_helper(*, verbose: bool = True) -> str | None:
    with build_kaggle_client() as api_client:
        # HACK(b/307576378): Remove once kagglesdk generates binding for the DiagnosticsService.
        http_client = api_client.http_client()
        http_client._init_session()
        request = http_client._prepare_request("api.v1.DiagnosticsService", "Hello", ApiGetModelRequest())
        response = http_client._session.send(request).json()
        if "userName" in response:
            if verbose:
                _logger.info("Kaggle credentials successfully validated.")
            return response["userName"]
        elif "code" in response and response["code"] == INVALID_CREDENTIALS_ERROR:
            if verbose:
                _logger.error(
                    "Invalid Kaggle credentials. You can obtain a Kaggle API token on your [Kaggle settings page](https://www.kaggle.com/settings/account)."
                )
        elif verbose:
            _logger.warning("Unable to validate Kaggle credentials at this time.")
        return None


def login(validate_credentials: bool = True) -> None:  # noqa: FBT002, FBT001
    """Prompt the user for their Kaggle API token and save it globally."""

    if _is_in_notebook():
        _notebook_login(validate_credentials)
        return
    else:
        token = getpass.getpass("Enter your Kaggle API token (input will not be visible): ")

    set_kaggle_api_token(token)

    if not validate_credentials:
        return

    _validate_credentials_helper()


def whoami(*, verbose: bool = True) -> dict:
    """
    Return a dictionary with the username of the authenticated Kaggle user or raise an error if unauthenticated.
    """
    if not get_kaggle_credentials():
        raise UnauthenticatedError()
    try:
        username = _validate_credentials_helper(verbose=verbose)
        if username:
            return {"username": username}
        raise UnauthenticatedError()
    except Exception as e:
        raise UnauthenticatedError() from e


def get_username() -> str | None:
    """Returns the username of the authenticated logged-in user if configured, otherwise None."""
    try:
        return whoami(verbose=False)["username"]
    except Exception:
        return None
