from typing import Optional
import logging
import requests
from http import HTTPStatus


from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import BackendError, KaggleApiHTTPError
from kagglehub.handle import ModelHandle

logger = logging.getLogger(__name__)


def create_model(owner_slug, model_slug):
    data = {"ownerSlug": owner_slug,
            "slug": model_slug,
            "title": model_slug,
            "isPrivate": True}
    api_client = KaggleApiV1Client()
    response = api_client.post("/models/create/new", data)
    # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
    if 'error' in response and response['error'] != "":
        raise BackendError(response['error'])
    logger.info("Model Created.")

def create_model_instance(model_handle: ModelHandle, license_name: str, files=None):
    data = {
        "instanceSlug": model_handle.variation,
        "framework": model_handle.framework,
        "licenseName": license_name,
        # TODO(mohamed): You need to upload the file to GCS first.
        # And then create the files in the proper format.
        # See: https://github.com/Kaggle/kaggleazure/blob/ecd6e72f278c8aed2b5dc76cb458eaea08283360/Kaggle.Sdk/models/model_api_service.proto#L154
        # And: https://github.com/Kaggle/kaggleazure/blob/ecd6e72f278c8aed2b5dc76cb458eaea08283360/Kaggle.Sdk/datasets/dataset_api_service.proto#L297-L301
        # "files": files
    }
    try:
        api_client = KaggleApiV1Client()
        response = api_client.post(f"/models/{model_handle.owner}/{model_handle.model}/create/instance", data)
        # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
        if 'error' in response and response['error'] != "":
            raise BackendError(response['error'])
        logger.info("Model Instance Created.")
    except requests.exceptions.HTTPError as e:
        print(e)
        logger.error(
            "Unable to create model instance at this time."
        )

def create_model_instance_version(model_handle: ModelHandle, version_notes: Optional[str] = None):
    data = {
        "versionNotes": version_notes
    }
    try:
        api_client = KaggleApiV1Client()
        api_client.post(f"/models/{model_handle}/create/version", data)
        # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
        if 'error' in response and response['error'] != "":
            raise BackendError(response['error'])
        logger.info("Model Instance Version Created.")
    except requests.exceptions.HTTPError as e:
        logger.error(
                "Unable to create model instance version at this time."
            )


def create_model_instance_or_version(model_handle: ModelHandle, license_name: str, files=None, version_notes: Optional[str] = None):
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/models/{model_handle}/get")
        # the instance exist, create a new version.
        create_model_instance_version(model_handle, version_notes)
    except KaggleApiHTTPError as e:
        if e.response.status_code == HTTPStatus.NOT_FOUND:
            create_model_instance(model_handle, license_name, files)
            return
        raise(e)

def get_or_create_model(owner_slug, model_slug):
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/models/{owner_slug}/{model_slug}/get")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == DOES_NOT_EXIST_ERROR:
            logger.error(
                f"Model '{model_slug}' does not exist for user '{owner_slug}'. Creating Model..."
            )
            create_model(owner_slug, model_slug)
        else:
            logger.warning("Unable to validate model exists at this time.")
#import kagglehub; from kagglehub.models_helpers import create_model_instance, create_model_instance_or_version; from kagglehub.auth import login; kagglehub.login()
#from kagglehub.models_helpers import create_model_instance_or_version
#create_model_instance_or_version('aminmohamedmohami', 'test', 'PyTorch', '13j', "Apache 2.0", ["/usr/local/google/home/aminmohamed/labeled_test_impression.csv"])
#create_model_instance("/usr/local/google/home/aminmohamed/labeled_test_impression.csv"])