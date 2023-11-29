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
from kagglehub.models_helpers import get_or_create_model, create_model_instance_or_version
from kagglehub.blob_service import StartBlobUploadRequest, StartBlobUploadResponse

logger = logging.getLogger(__name__)

DOES_NOT_EXIST_ERROR = 404

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

class ResumableUploadResult(object):
    # Upload was complete, i.e., all bytes were received by the server.
    COMPLETE = 1

    # There was a non-transient error during the upload or the upload expired.
    # The upload cannot be resumed so it should be restarted from scratch
    # (i.e., call /api/v1/files/upload to initiate the upload and get the
    # create/upload url and token).
    FAILED = 2

    # Upload was interrupted due to some (transient) failure but it can be
    # safely resumed.
    INCOMPLETE = 3

class ResumableUploadContext(object):
    def __init__(self, no_resume=False):
        self.no_resume = no_resume
        self._temp_dir = os.path.join(tempfile.gettempdir(), '.kaggle/uploads')
        self._file_uploads = []

    def __enter__(self):
        if self.no_resume:
            return
        self._create_temp_dir()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.no_resume:
            return
        if exc_type is not None:
            # Don't delete the upload file info when there is an error
            # to give it a chance to retry/resume on the next invocation.
            return
        for file_upload in self._file_uploads:
            file_upload.cleanup()

    def get_upload_info_file_path(self, path):
        return os.path.join(
            self._temp_dir,
            '%s.json' % path.replace(os.path.sep, '_').replace(':', '_'))

    def new_resumable_file_upload(self, path, start_blob_upload_request):
        file_upload = ResumableFileUpload(path, start_blob_upload_request,
                                          self)
        self._file_uploads.append(file_upload)
        file_upload.load()
        return file_upload

    def _create_temp_dir(self):
        try:
            os.makedirs(self._temp_dir)
        except FileExistsError:
            pass

class ResumableFileUpload(object):
    # Reference: https://cloud.google.com/storage/docs/resumable-uploads
    # A resumable upload must be completed within a week of being initiated
    RESUMABLE_UPLOAD_EXPIRY_SECONDS = 6 * 24 * 3600

    def __init__(self, path, start_blob_upload_request, context):
        self.path = path
        self.start_blob_upload_request = start_blob_upload_request
        self.context = context
        self.timestamp = int(time.time())
        self.start_blob_upload_response = None
        self.can_resume = False
        self.upload_complete = False
        if self.context.no_resume:
            return
        self._upload_info_file_path = self.context.get_upload_info_file_path(
            path)

    def get_token(self):
        if self.upload_complete:
            return self.start_blob_upload_response.token
        return None

    def load(self):
        if self.context.no_resume:
            return
        self._load_previous_if_any()

    def _load_previous_if_any(self):
        if not os.path.exists(self._upload_info_file_path):
            return False

        try:
            with io.open(self._upload_info_file_path, 'r') as f:
                previous = ResumableFileUpload.from_dict(
                    json.load(f), self.context)
                if self._is_previous_valid(previous):
                    self.start_blob_upload_response = previous.start_blob_upload_response
                    self.timestamp = previous.timestamp
                    self.can_resume = True
        except Exception as e:
            print('Error while trying to load upload info:', e)

    def _is_previous_valid(self, previous):
        return previous.path == self.path and \
               previous.start_blob_upload_request == self.start_blob_upload_request and \
               previous.timestamp > time.time() - ResumableFileUpload.RESUMABLE_UPLOAD_EXPIRY_SECONDS

    def upload_initiated(self, start_blob_upload_response):
        if self.context.no_resume:
            return

        self.start_blob_upload_response = start_blob_upload_response
        with io.open(self._upload_info_file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=True)

    def upload_completed(self):
        if self.context.no_resume:
            return

        self.upload_complete = True
        self._save()

    def _save(self):
        with io.open(self._upload_info_file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=True)

    def cleanup(self):
        if self.context.no_resume:
            return

        try:
            os.remove(self._upload_info_file_path)
        except OSError:
            pass

    def to_dict(self):
        return {
            'path':
            self.path,
            'start_blob_upload_request':
            self.start_blob_upload_request.to_dict(),
            'timestamp':
            self.timestamp,
            'start_blob_upload_response':
            self.start_blob_upload_response.to_dict()
            if self.start_blob_upload_response is not None else None,
            'upload_complete':
            self.upload_complete,
        }

    def from_dict(other, context):
        new = ResumableFileUpload(
            other['path'],
            StartBlobUploadRequest(**other['start_blob_upload_request']),
            context)
        new.timestamp = other.get('timestamp')
        start_blob_upload_response = other.get('start_blob_upload_response')
        if start_blob_upload_response is not None:
            new.start_blob_upload_response = StartBlobUploadResponse(
                **start_blob_upload_response)
            new.upload_complete = other.get('upload_complete') or False
        return new

    def to_str(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.to_str()


def _upload_blob(self, path, quiet, blob_type, upload_context):
        """ upload a file

            Parameters
            ==========
            path: the complete path to upload
            quiet: suppress verbose output (default is False)
            blob_type (ApiBlobType): To which entity the file/blob refers
            upload_context (ResumableUploadContext): Context for resumable uploads
        """
        file_name = os.path.basename(path)
        content_length = os.path.getsize(path)
        last_modified_epoch_seconds = int(os.path.getmtime(path))

        start_blob_upload_request = StartBlobUploadRequest(
            blob_type,
            file_name,
            content_length,
            last_modified_epoch_seconds=last_modified_epoch_seconds)

        file_upload = upload_context.new_resumable_file_upload(
            path, start_blob_upload_request)

        for i in range(0, self.MAX_UPLOAD_RESUME_ATTEMPTS):
            if file_upload.upload_complete:
                return file_upload

            if not file_upload.can_resume:
                # Initiate upload on Kaggle backend to get the url and token.
                start_blob_upload_response = self.process_response(
                    self.with_retry(self.upload_file_with_http_info)(
                        file_upload.start_blob_upload_request))
                file_upload.upload_initiated(start_blob_upload_response)

            upload_result = self.upload_complete(
                path,
                file_upload.start_blob_upload_response.create_url,
                quiet,
                resume=file_upload.can_resume)
            if upload_result == ResumableUploadResult.INCOMPLETE:
                continue  # Continue (i.e., retry/resume) only if the upload is incomplete.

            if upload_result == ResumableUploadResult.COMPLETE:
                file_upload.upload_completed()
            break

        return file_upload.get_token()

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

def create_model():
    try:
        api_client = KaggleApiV1Client()
        api_client.post("/models/create/new")
        logger.info("Model Created.")
    except requests.exceptions.HTTPError as e:
        logger.error(
                "Unable to create model at this time."
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


def create_model_instance(owner_slug, model_slug):
    try:
        api_client = KaggleApiV1Client()
        api_client.post(f"/models/{owner_slug}/{model_slug}/create/instance")
        logger.info("Model Instance Created.")
    except requests.exceptions.HTTPError as e:
        logger.error(
                "Unable to create model instance at this time."
            )


def create_model_instance_or_version(owner_slug, model_slug, framework, instance_slug):
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
            api_client.post(f"/models/{owner_slug}/{model_slug}/{framework}/{instance_slug}/create/version")
            logger.info("Model Instance Version Created.")
        except requests.exceptions.HTTPError as e:
            logger.error(
                    "Unable to create model instance version at this time."
                )