import logging
from pathlib import Path

from kagglehub.handle import ModelHandle
from kagglehub.models_helpers import signing_token

logger = logging.getLogger(__name__)


def sign_with_sigstore(local_model_dir: str, handle: ModelHandle) -> bool:
    from model_signing.signing import Config  # noqa: PLC0415

    try:
        token = signing_token(handle.owner, handle.model)
        if token:
            signing_file = Path(local_model_dir) / ".kaggle" / "signing.json"
            signing_file.parent.mkdir(exist_ok=True, parents=True)
            signing_file.unlink(missing_ok=True)
            # The below will throw an exception if the token can't be verified (Needs to be a production token)
            # Setting KAGGLE_API_ENDPOINT to localhost will throw the exception as stated above.
            Config().use_sigstore_signer(identity_token=token, use_staging=False).sign(
                Path(local_model_dir), signing_file
            )
            return True
        else:
            # skips transparency log publishing as we are unable to get a token
            logger.warning("Unable to retrieve identity token. Skipping signing...")
            return False
    except Exception:
        logger.exception("Signing failed. Skipping signing...")
        return False
