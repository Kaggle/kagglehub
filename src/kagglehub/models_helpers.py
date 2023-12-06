from typing import Optional
import logging
import requests

from kagglehub.clients import KaggleApiV1Client

logger = logging.getLogger(__name__)

DOES_NOT_EXIST_ERROR = 404

def create_model(owner_slug, model_slug, title):
    try:
        data = {"ownerSlug": owner_slug,
                "slug": model_slug,
                "title": title,
                "isPrivate": True}
        api_client = KaggleApiV1Client()
        api_client.post("/models/create/new", data)
        logger.info("Model Created.")
    except requests.exceptions.HTTPError as e:
        print(e)
        logger.error(
                "Unable to create model at this time."
            )

def create_model_instance(owner_slug, model_slug, framework, instance_slug, license_name, files=None):
    data = {
        "owner_slug": owner_slug,
        "model_slug":  model_slug,
        "body":  {
            "instance_slug": instance_slug,
            "framework": framework,
            "license_name": license_name,
            "files": files
            }
        }
    try:
        api_client = KaggleApiV1Client()
        response = api_client.post(f"/models/{owner_slug}/{model_slug}/create/instance", data)
        print(response)
        logger.info("Model Instance Created.")
    except requests.exceptions.HTTPError as e:
        print(e)
        logger.error(
                "Unable to create model instance at this time."
            )

def create_model_instance_or_version(owner_slug, model_slug, framework, instance_slug, license_name, files=None):
    instance_exists = True
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/models/{owner_slug}/{model_slug}/{framework}/{instance_slug}/get")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == DOES_NOT_EXIST_ERROR:
            instance_exists = False
            logger.error(
                f"Model instance for framework '{framework}' does not exist for model '{model_slug}'. Creating..."
            )
            create_model_instance(owner_slug, model_slug, framework, instance_slug, license_name, files)
        else:
            logger.warning("Unable to validate model instance exists at this time.")
            return

    if instance_exists == True:
        data = {
        "owner_slug": owner_slug,
        "model_slug":  model_slug,
        "instance_slug": instance_slug,
        "framework": framework,
        "body":  {
            "version_notes": "hey"
            }
        }
        try:
            api_client = KaggleApiV1Client()
            api_client.post(f"/models/{owner_slug}/{model_slug}/{framework}/{instance_slug}/create/version", data)
            logger.info("Model Instance Version Created.")
        except requests.exceptions.HTTPError as e:
            logger.error(
                    "Unable to create model instance version at this time."
                )
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
#create_model_instance("/usr/local/google/home/aminmohamed/labeled_test_impression.csv"])