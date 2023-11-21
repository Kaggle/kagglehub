import logging
import sys
import sys
import io

import requests

from kagglehub.clients import KaggleApiV1Client
from kagglehub.config import set_kaggle_credentials

logger = logging.getLogger(__name__)

INVALID_CREDENTIALS_ERROR = 403

NOTEBOOK_LOGIN_TOKEN_HTML_START = """<center> <img
src=https://www.kaggle.com/static/images/site-logo.png
alt='Kaggle'> <br> Copy a token from <a
href="https://www.kaggle.com/settings/account" target="_blank">your Kaggle
settings page</a> and paste it below. <br> </center>"""

NOTEBOOK_LOGIN_TOKEN_HTML_END = """
<b>Thank You</b></center>"""

def is_in_notebook():
    return "IPython" in sys.modules

def notebook_login(validate_credentials) -> None:
    """Prompt the user for their Kaggle token and save it in a widget (Jupyter or Colab)."""
    try:
        import ipywidgets.widgets as widgets  # type: ignore
        from IPython.display import display  # type: ignore
    except ImportError:
        raise ImportError(
            "You need the `ipywidgets` module: `pip install ipywidgets`."
        )

    box_layout = widgets.Layout(
        display="flex", flex_flow="column", align_items="center", width="50%"
    )

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

    # On click events
    def login_token_event():
        username = username_widget.value
        token = token_widget.value
        # Erase token and clear value to make sure it's not saved in the notebook.
        token_widget.value = ""

        # Hide inputs
        login_token_widget.children = [widgets.Label("Connecting...")]
        try:
            # Redirect stdout to an in-memory StringIO object
            output_buffer = io.StringIO()
            sys.stdout = output_buffer

            # Set Kaggle credentials
            set_kaggle_credentials(username=username, api_key=token)

            # Validate credentials if necessary
            if validate_credentials is True:
                validate_credentials_helper()

            # Capture the output from stdout and assign it to message
            captured_output = output_buffer.getvalue()
            message = captured_output

            # Reset stdout back to the original console
            sys.stdout = sys.__stdout__
        except Exception as error:
            message = str(error)
        message = "test"
        # Print result (success message or error)
        login_token_widget.children = [widgets.Label(line) for line in message.split("\n") if line.strip()]
    login_button.on_click(login_token_event)
    
def validate_credentials_helper():
    try:
        api_client = KaggleApiV1Client()
        api_client.get("/hello")
        logger.info("Kaggle credentials successfully validated.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == INVALID_CREDENTIALS_ERROR:
            logger.error(
                "Invalid Kaggle credentials. You can check your credentials on the [Kaggle settings page](https://www.kaggle.com/settings/account)."
            )
        else:
            logger.warning("Unable to validate Kaggle credentials at this time.")


def login(validate_credentials=True):  # noqa: FBT002
    """Prompt the user for their Kaggle username and API key and save them globally."""

    if is_in_notebook():
        notebook_login(validate_credentials)
        return
    else:
        username = input("Enter your Kaggle username: ")
        api_key = input("Enter your Kaggle API key: ")

    set_kaggle_credentials(username=username, api_key=api_key)

    logger.info("Kaggle credentials set.")

    if not validate_credentials:
        return

    validate_credentials_helper()
