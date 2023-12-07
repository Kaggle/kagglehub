import datetime
import os
import shutil
import tempfile
from typing import Optional
import logging
import requests
from http import HTTPStatus

import urllib


from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import BackendError, KaggleApiHTTPError
from kagglehub.handle import ModelHandle

logger = logging.getLogger(__name__)

DATASET_METADATA_FILE = 'dataset-metadata.json'
OLD_DATASET_METADATA_FILE = 'datapackage.json'
KERNEL_METADATA_FILE = 'kernel-metadata.json'
MODEL_METADATA_FILE = 'model-metadata.json'
MODEL_INSTANCE_METADATA_FILE = 'model-instance-metadata.json'


def parse(string):
    time_formats = [
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%fZ'
    ]
    for t in time_formats:
        try:
            result = datetime.strptime(string[:26], t).replace(microsecond=0)
            return result
        except:
            pass
    return string

class DirectoryArchive(object):
    def __init__(self, fullpath, format):
        self._fullpath = fullpath
        self._format = format
        self.name = None
        self.path = None

    def __enter__(self):
        self._temp_dir = tempfile.mkdtemp()
        _, dir_name = os.path.split(self._fullpath)
        self.path = shutil.make_archive(
            os.path.join(self._temp_dir, dir_name), self._format,
            self._fullpath)
        _, self.name = os.path.split(self.path)
        return self

    def __exit__(self, *args):
        shutil.rmtree(self._temp_dir)

class File(object):
    def __init__(self, init_dict):
        parsed_dict = {k: parse(v) for k, v in init_dict.items()}
        self.__dict__.update(parsed_dict)
        self.size = File.get_size(self.totalBytes)

    def __repr__(self):
        return self.ref

    @staticmethod
    def get_size(size, precision=0):
        suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
        suffix_index = 0
        while size >= 1024 and suffix_index < 4:
            suffix_index += 1
            size /= 1024.0
        return '%.*f%s' % (precision, size, suffixes[suffix_index])

def _upload_blob(file_path):
    data = {
        "type": "model",
        "name": os.path.basename(file_path),
        "contentLength": os.path.getsize(file_path),
        "lastModifiedEpochSeconds": int(os.path.getmtime(file_path))
    }
    api_client = KaggleApiV1Client()
    response = api_client.post("/blobs/upload", data=data)
    print(response)
    if 'error' in response and response['error'] != "":
        raise BackendError(response['error'])

    with open(file_path, "rb") as f:
        file_data = f.read()

    headers = {"Content-Type": "application/octet-stream"}
    signed_response = requests.put(response['createUrl'], data=file_data, headers=headers)

    if 'error' in signed_response and signed_response['error'] != "":
        raise BackendError(signed_response['error'])

    return response['token']

def upload_files(folder, quiet=False, dir_mode='skip'):
    """ upload files in a folder
            Parameters
        ==========
        request: the prepared request
        folder: the folder to upload from
        quiet: suppress verbose output (default is False)
    """
    tokens = []
    for file_name in os.listdir(folder):
        if (file_name in [
                DATASET_METADATA_FILE, OLD_DATASET_METADATA_FILE,
                KERNEL_METADATA_FILE, MODEL_METADATA_FILE,
                MODEL_INSTANCE_METADATA_FILE
        ]):
            continue
        tokens.append(_upload_file_or_folder(
            folder, file_name, dir_mode, quiet))
    return tokens

def _upload_file_or_folder(parent_path,
                            file_or_folder_name,
                            dir_mode,
                            quiet=False):
    full_path = os.path.join(parent_path, file_or_folder_name)
    if os.path.isfile(full_path):
        return _upload_file(file_or_folder_name, full_path, quiet)

    elif os.path.isdir(full_path):
        if dir_mode in ['zip', 'tar']:
            with DirectoryArchive(full_path, dir_mode) as archive:
                return _upload_file(archive.name, archive.path,
                                            quiet)
        elif not quiet:
            print("Skipping folder: " + file_or_folder_name +
                    "; use '--dir-mode' to upload folders")
    else:
        if not quiet:
            print('Skipping: ' + file_or_folder_name)
    return None

def _upload_file(file_name, full_path, quiet):
    """ Helper function to upload a single file
        Parameters
        ==========
        file_name: name of the file to upload
        full_path: path to the file to upload
        quiet: suppress verbose output
        :return: None - upload unsuccessful; instance of UploadFile - upload successful
    """

    if not quiet:
        print('Starting upload for file ' + file_name)

    content_length = os.path.getsize(full_path)
    token = _upload_blob(full_path)
    if token is None:
        if not quiet:
            print('Upload unsuccessful: ' + file_name)
        return None
    if not quiet:
        print('Upload successful: ' + file_name + ' (' +
                File.get_size(content_length) + ')')
    return token

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

def create_model_instance(model_handle: ModelHandle, license_name: str, files: list[str]):
    data = {
        "instanceSlug": model_handle.variation,
        "framework": model_handle.framework,
        "licenseName": license_name,
        "overview": "test",
        "usage": "test",
        "trainingData": ["test"],
        "files": [{"token": files[0]}]
    }
    api_client = KaggleApiV1Client()
    response = api_client.post(f"/models/{model_handle.owner}/{model_handle.model}/create/instance", data)
    # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
    if 'error' in response and response['error'] != "":
        raise BackendError(response['error'])
    logger.info("Model Instance Created.")

def create_model_instance_version(model_handle: ModelHandle, files: list[str], version_notes=""):
    data = {
        "versionNotes": version_notes,
        "files": [{"token": files[0]}]
    }
    api_client = KaggleApiV1Client()
    response = api_client.post(f"/models/{model_handle.owner}/{model_handle.model}/{model_handle.framework}/{model_handle.variation}/create/version", data)
    # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
    if 'error' in response and response['error'] != "":
        raise BackendError(response['error'])
    logger.info("Model Instance Version Created.")

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
    except KaggleApiHTTPError as e:
        if e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.error(
                f"Model '{model_slug}' does not exist for user '{owner_slug}'. Creating Model..."
            )
            create_model(owner_slug, model_slug)
            return
        raise(e)
