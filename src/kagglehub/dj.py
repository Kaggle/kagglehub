import contextvars
import importlib
import inspect
import os
import pathlib
import sys
from typing import Callable
import urllib.request

from kagglehub.clients import KaggleApiV1Client

scoped_pipeline_dir_ctx = contextvars.ContextVar('kagglehub_scoped_pipeline_dir', default=None)

# Decorator which scopes a function within an exported Pipeline module to
# 1. Change working directory to the module's base directory so files can be read easily.
# 2. Set a `kagglehub` ContextVar so datasource references will use the versions used
#    by the original saved version of the Pipeline.
def pipeline_scoped(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        module = inspect.getmodule(func)
        pipeline_dir = (
            pathlib.Path(os.path.dirname(module.__file__)).parent
            if hasattr(module, '__file__')
            # No __file__ means we weren't imported, so assume we're in an interactive session
            else '/kaggle/working'
        )

        original_ctx = scoped_pipeline_dir_ctx.get()
        try:
            scoped_pipeline_dir_ctx.set(pipeline_dir)
            return_value = func(*args, **kwargs)
        finally:
            scoped_pipeline_dir_ctx.set(original_ctx)
        return return_value
    return inner

def get_asset_path(path):
    pipeline_dir = scoped_pipeline_dir_ctx.get() or '/kaggle/working'
    assets_dir = os.path.join(pipeline_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    return os.path.join(assets_dir, path)

def download_notebook_output_files(user_name, kernel_slug, out_dir=''):
    client = KaggleApiV1Client();
    response = client.get(f'kernels/output?user_name={user_name}&kernel_slug={kernel_slug}&page_size=500')
    for f in response['files']:
        outfile_path = os.path.join(out_dir, f['fileName'])
        outfile_dir = os.path.dirname(outfile_path)
        if len(outfile_dir) > 0:
            os.makedirs(outfile_dir, exist_ok=True)
        urllib.request.urlretrieve(f['url'], outfile_path)
        print(f'downloaded {outfile_path}')

def import_pipeline_module(user_name, kernel_slug):
    download_notebook_output_files(user_name, kernel_slug, f'/kaggle/tmp/code/{user_name}_{kernel_slug}')
    # TODO: try to import without appending to path?
    sys.path.append(f'/kaggle/tmp/code')
    return importlib.import_module(f'{user_name}_{kernel_slug}.module')
