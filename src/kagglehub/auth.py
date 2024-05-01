import io
import logging
from contextlib import contextmanager
from typing import Generator

from kagglehub.clients import KaggleApiV1Client
from kagglehub.config import get_kaggle_credentials, set_kaggle_credentials
from kagglehub.exceptions import UnauthenticatedError

logger = logging.getLogger(__name__)

INVALID_CREDENTIALS_ERROR = 401

NOTEBOOK_LOGIN_TOKEN_HTML_START = """<center> <img
src=https://www.kaggle.com/static/images/site-logo.png
alt='Kaggle'> <br> Create an API token from <a
href="https://www.kaggle.com/settings/account" target="_blank">your Kaggle
settings page</a> and paste it below along with your Kaggle username. <br> </center>"""

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
    logger.addHandler(handler)
    try:
        yield buffer
    finally:
        logger.removeHandler(handler)


def _is_in_notebook() -> bool:
    """Return `True` if code is executed in a notebook (Jupyter, Kaggle Notebook, Colab, QTconsole).

    Taken from https://stackoverflow.com/a/39662359.
    Adapted to make it work with Google colab as well.
    """
    try:
        from IPython import get_ipython  # type: ignore

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
        from IPython.display import display  # type: ignore
        from ipywidgets import widgets  # type: ignore
    except ImportError:
        raise ImportError(library_error)  # noqa: B904

    box_layout = widgets.Layout(display="flex", flex_flow="column", align_items="center", width="50%")

    username_widget = widgets.Text(description="Username:")
    token_widget = widgets.Password(description="Token:")
    login_button = widgets.Button(description="Login")

    login_token_widget = widgets.VBox(
        [
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_START),
            username_widget,
            token_widget,
            login_button,
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_END),
        ],
        layout=box_layout,
    )
    display(login_token_widget)

    def on_click_login_button(_: str) -> None:
        username = username_widget.value
        token = token_widget.value
        # Erase token and clear value to make sure it's not saved in the notebook.
        token_widget.value = ""

        # Hide inputs
        login_token_widget.children = [widgets.Label("Connecting...")]
        try:
            with _capture_logger_output() as captured:
                # Set Kaggle credentials
                set_kaggle_credentials(username=username, api_key=token)

                # Validate credentials if necessary
                if validate_credentials is True:
                    _validate_credentials_helper()
            message = captured.getvalue()
        except Exception as error:
            message = str(error)
        # Print result (success message or error)
        login_token_widget.children = [widgets.Label(line) for line in message.split("\n") if line.strip()]

    login_button.on_click(on_click_login_button)


def _validate_credentials_helper() -> None:
    api_client = KaggleApiV1Client()
    response = api_client.get("/hello")
    if "code" not in response:
        logger.info("Kaggle credentials successfully validated.")
    elif response["code"] == INVALID_CREDENTIALS_ERROR:
        logger.error(
            "Invalid Kaggle credentials. You can check your credentials on the [Kaggle settings page](https://www.kaggle.com/settings/account)."
        )
    else:
        logger.warning("Unable to validate Kaggle credentials at this time.")


def login(validate_credentials: bool = True) -> None:  # noqa: FBT002, FBT001
    """Prompt the user for their Kaggle username and API key and save them globally."""

    if _is_in_notebook():
        _notebook_login(validate_credentials)
        return
    else:
        username = input("Enter your Kaggle username: ")
        api_key = input("Enter your Kaggle API key: ")

    set_kaggle_credentials(username=username, api_key=api_key)

    if not validate_credentials:
        return

    _validate_credentials_helper()


def whoami() -> dict:
    """
    Return a dictionary with the username of the authenticated Kaggle user or raise an error if unauthenticated.
    """
    try:
        credentials = get_kaggle_credentials()
        if credentials and credentials.username:
            return {'username': credentials.username}
        else:
            raise UnauthenticatedError()
    except Exception as e:
        raise UnauthenticatedError() from e
