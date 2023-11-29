import os
import shutil
import tempfile
from typing import Optional
import logging
import requests
import json
import io
import time
import datetime


from kagglehub.clients import KaggleApiV1Client

logger = logging.getLogger(__name__)

DOES_NOT_EXIST_ERROR = 401

DATASET_METADATA_FILE = 'dataset-metadata.json'
OLD_DATASET_METADATA_FILE = 'datapackage.json'
KERNEL_METADATA_FILE = 'kernel-metadata.json'
MODEL_METADATA_FILE = 'model-metadata.json'
MODEL_INSTANCE_METADATA_FILE = 'model-instance-metadata.json'

API_URL = "https://api.example.com/v1/blobs/upload"


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

def upload_blob(file_path, quiet, blob_type, upload_context):
    with open(file_path, 'rb') as f:
        file_data = f.read()

    request_body = {
        "blob_type": blob_type,
        "file_name": os.path.basename(file_path),
        "content_length": os.path.getsize(file_data),
        "last_modified_epoch_seconds": int(os.path.getmtime(file_path))
    }

    response = requests.post(API_URL, json=request_body)
    if response.status_code != 200:
        raise Exception('Failed to initiate upload: ' + response.text)

    upload_url = response.json()['upload_url']

    # Split the file into chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    chunks = [file_data[i:i+chunk_size] for i in range(0, len(file_data), chunk_size)]

    # Upload the chunks
    for i, chunk in enumerate(chunks):
        chunk_offset = i * chunk_size
        chunk_uri = upload_url + '&upload_offset=' + str(chunk_offset)

        headers = {
            'Content-Range': 'bytes {}-{}/{}'.format(chunk_offset, chunk_offset + len(chunk) - 1, len(file_data)),
        }

        chunk_response = requests.put(chunk_uri, headers=headers, data=chunk)
        if chunk_response.status_code != 200:
            raise Exception('Failed to upload chunk: ' + chunk_response.text)

    # Complete the upload
    complete_upload_request = {
        'upload_id': response.json()['upload_id']
    }

    complete_response = requests.post(API_URL, json=complete_upload_request)
    if complete_response.status_code != 200:
        raise Exception('Failed to complete upload: ' + complete_response.text)

    print('File uploaded successfully!')

def upload_files(self,
                    request,
                    resources,
                    folder,
                    blob_type,
                    upload_context,
                    quiet=False,
                    dir_mode='skip'):
    """ upload files in a folder
            Parameters
        ==========
        request: the prepared request
        resources: the files to upload
        folder: the folder to upload from
        blob_type (ApiBlobType): To which entity the file/blob refers
        upload_context (ResumableUploadContext): Context for resumable uploads
        quiet: suppress verbose output (default is False)
    """
    for file_name in os.listdir(folder):
        if (file_name in [
                self.DATASET_METADATA_FILE, self.OLD_DATASET_METADATA_FILE,
                self.KERNEL_METADATA_FILE, self.MODEL_METADATA_FILE,
                self.MODEL_INSTANCE_METADATA_FILE
        ]):
            continue
        upload_file = self._upload_file_or_folder(
            folder, file_name, blob_type, upload_context, dir_mode, quiet,
            resources)
        if upload_file is not None:
            request.files.append(upload_file)

def _upload_file_or_folder(self,
                            parent_path,
                            file_or_folder_name,
                            blob_type,
                            upload_context,
                            dir_mode,
                            quiet=False,
                            resources=None):
    full_path = os.path.join(parent_path, file_or_folder_name)
    if os.path.isfile(full_path):
        return self._upload_file(file_or_folder_name, full_path, blob_type,
                                    upload_context, quiet, resources)

    elif os.path.isdir(full_path):
        if dir_mode in ['zip', 'tar']:
            with DirectoryArchive(full_path, dir_mode) as archive:
                return self._upload_file(archive.name, archive.path,
                                            blob_type, upload_context, quiet,
                                            resources)
        elif not quiet:
            print("Skipping folder: " + file_or_folder_name +
                    "; use '--dir-mode' to upload folders")
    else:
        if not quiet:
            print('Skipping: ' + file_or_folder_name)
    return None

def _upload_file(self, file_name, full_path, blob_type, upload_context,
                    quiet, resources):
    """ Helper function to upload a single file
        Parameters
        ==========
        file_name: name of the file to upload
        full_path: path to the file to upload
        blob_type (ApiBlobType): To which entity the file/blob refers
        upload_context (ResumableUploadContext): Context for resumable uploads
        quiet: suppress verbose output
        resources: optional file metadata
        :return: None - upload unsuccessful; instance of UploadFile - upload successful
    """

    if not quiet:
        print('Starting upload for file ' + file_name)

    content_length = os.path.getsize(full_path)
    token = self._upload_blob(full_path, quiet, blob_type, upload_context)
    if token is None:
        if not quiet:
            print('Upload unsuccessful: ' + file_name)
        return None
    if not quiet:
        print('Upload successful: ' + file_name + ' (' +
                File.get_size(content_length) + ')')
    return

def create_model(data=None):
    try:
        data = {
        "framework": "tensorflow",
        "version": "1.0",
        "description": "A pre-trained model for image classification",
        "tags": ["image-classification", "pretrained"]
        }
        api_client = KaggleApiV1Client()
        api_client.post("/models/create/new", data)
        logger.info("Model Created.")
    except requests.exceptions.HTTPError as e:
        logger.error(
                "Unable to create model at this time."
            )

def create_model_instance(owner_slug, model_slug, data=None):
    try:
        api_client = KaggleApiV1Client()
        api_client.post(f"/models/{owner_slug}/{model_slug}/create/instance", data)
        logger.info("Model Instance Created.")
    except requests.exceptions.HTTPError as e:
        logger.error(
                "Unable to create model instance at this time."
            )


def create_model_instance_or_version(owner_slug, model_slug, framework, instance_slug, data=None):
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
            create_model_instance(owner_slug, model_slug)
        else:
            logger.warning("Unable to validate model instance exists at this time.")
            return

    if instance_exists == True:
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
            create_model()
        else:
            logger.warning("Unable to validate model exists at this time.")