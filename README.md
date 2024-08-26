# Kaggle Hub Client Library

## Installation

Install the `kagglehub` package with pip:

```
pip install kagglehub
```

## Usage

### Authenticate

Authenticating is **only** needed to access public resources requiring user consent or private resources.

First, you will need a Kaggle account. You can sign up [here](https://www.kaggle.com/account/login).

After login, you can download your Kaggle API credentials at https://www.kaggle.com/settings by clicking on the "Create New Token" button under the "API" section.

You have 3 different options to authenticate.

#### Option 1: Calling kagglehub.login()

This will prompt you to enter your username and token:

```python
import kagglehub

kagglehub.login()
```

#### Option 2: Read credentials from environment variables

You can also choose to export your Kaggle username and token to the environment:

```sh
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx
```

#### Option 3: Read credentials from `kaggle.json`

Store your `kaggle.json` credentials file at `~/.kaggle/kaggle.json`.

Alternatively, you can set the `KAGGLE_CONFIG_DIR` environment variable to change this location to `$KAGGLE_CONFIG_DIR/kaggle.json`.

Note for Windows users: The default directory is `%HOMEPATH%/kaggle.json`.

### Download Model

The following examples download the `answer-equivalence-bem` variation of this Kaggle model: https://www.kaggle.com/models/google/bert/tensorFlow2/answer-equivalence-bem

```python
import kagglehub

# Download the latest version.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem')

# Download a specific version.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem/1')

# Download a single file.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem', path='variables/variables.index')

# Download a model or file, even if previously downloaded to cache. Only available outside Kaggle notebooks.
kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem', force_download=True)
```

### Upload Model
Uploads a new variation (or a new variation's version if it already exists).

```python
import kagglehub

# For example, to upload a new variation to this model:
# - https://www.kaggle.com/models/google/bert/tensorFlow2/answer-equivalence-bem
# 
# You would use the following handle: `google/bert/tensorFlow2/answer-equivalence-bem`
handle = '<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>'
local_model_dir = 'path/to/local/model/dir'

kagglehub.model_upload(handle, local_model_dir)

# You can also specify some version notes (optional)
kagglehub.model_upload(handle, local_model_dir, version_notes='improved accuracy')

# You can also specify a license (optional)
kagglehub.model_upload(handle, local_model_dir, license_name='Apache 2.0')

# You can also specify a list of patterns for files/dirs to ignore.
# These patterns are combined with `kagglehub.models.DEFAULT_IGNORE_PATTERNS` 
# to determine which files and directories to exclude. 
# To ignore entire directories, include a trailing slash (/) in the pattern.
kagglehub.model_upload(handle, local_model_dir, ignore_patterns=["original/", "*.tmp"])
```

### Download Dataset

The following examples download the `Spotify Recommendation` Kaggle dataset: https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation

```python
import kagglehub

# Download the latest version.
kagglehub.dataset_download('bricevergnou/spotify-recommendation')

# Download a specific version.
kagglehub.dataset_download('bricevergnou/spotify-recommendation/versions/1')

# Download a single file
kagglehub.dataset_download('bricevergnou/spotify-recommendation', path='data.csv')

# Download a dataset or file, even if previously downloaded to cache. Only available outside Kaggle notebooks.
kagglehub.dataset_download('bricevergnou/spotify-recommendation', force_download=True)
```

### Upload Dataset

Uploads a new dataset (or a new version if it already exists).

```python
import kagglehub

# For example, to upload a new dataset (or version) at:
# - https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation
# 
# You would use the following handle: `bricevergnou/spotify-recommendation`
handle = '<KAGGLE_USERNAME>/<DATASET>
local_dataset_dir = 'path/to/local/dataset/dir'

# Create a new dataset
kagglehub.dataset_upload(handle, local_dataset_dir)

# You can then create a new version of this existing dataset and include version notes (optional).
kagglehub.dataset_upload(handle, local_dataset_dir, version_notes='improved data')

# You can also specify a list of patterns for files/dirs to ignore.
# These patterns are combined with `kagglehub.datasets.DEFAULT_IGNORE_PATTERNS` 
# to determine which files and directories to exclude. 
# To ignore entire directories, include a trailing slash (/) in the pattern.
kagglehub.dataset_upload(handle, local_dataset_dir, ignore_patterns=["original/", "*.tmp"])
```

## Development

### Prequisites

We use [hatch](https://hatch.pypa.io) to manage this project.

Follow these [instructions](https://hatch.pypa.io/latest/install/) to install it.

### Tests

```sh
# Run all tests for current Python version.
hatch test

# Run all tests for all Python versions.
hatch test --all

# Run all tests for a specific Python version.
hatch test -py 3.11

# Run a single test file
hatch test tests/test_<SOME_FILE>.py
```

### Integration Tests

To run integration tests on your local machine, you need to set up your Kaggle API credentials. You can do this in one of these two ways described in the earlier sections of this document. Refer to the sections: 
- [Using environment variables](#option-2-read-credentials-from-environment-variables)
- [Using credentials file](#option-3-read-credentials-from-kagglejson)

After setting up your credentials by any of these methods, you can run the integration tests as follows:

```sh
# Run all tests
hatch test integration_tests
```


### Run `kagglehub` from source

```sh
# Download a model & print the path
hatch run python -c "import kagglehub; print('path: ', kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem'))"
```

### Lint / Format

```sh
# Lint check
hatch run lint:style
hatch run lint:typing
hatch run lint:all     # for both

# Format
hatch run lint:fmt
```

### Coverage report

```sh
hatch test --cover
```

### Build

```sh
hatch build
```

### Running `hatch` commands inside Docker

This is useful to run in a consistent environment and easily switch between Python versions.

The following shows how to run `hatch run lint:all` but this also works for any other hatch commands:

```
# Use default Python version
./docker-hatch run lint:all

# Use specific Python version (Must be a valid tag from: https://hub.docker.com/_/python)
./docker-hatch -v 3.9 run lint:all

# Run test in docker with specific Python version
./docker-hatch -v 3.9 test
```

## VS Code setup

### Prerequisites
Install the recommended extensions.

### Instructions

Configure hatch to create virtual env in project folder.
```
hatch config set dirs.env.virtual .env
```

After, create all the python environments needed by running `hatch -e all run tests`.

Finally, configure vscode to use one of the selected environments:
`cmd + shift + p` -> `python: Select Interpreter` -> Pick one of the folders in `./.env`

## Support

The kagglehub library has configured automatic logging which is stored in a log folder. The log destination is resolved via the [os.path.expanduser](https://docs.python.org/3/library/os.path.html#os.path.expanduser)

The table below contains possible locations:
| os      | log path                                         |
|---------|--------------------------------------------------|
| osx     | /user/$USERNAME/.kaggle/logs/kagglehub.log       |
| linux   | ~/.kaggle/logs/kagglehub.log                     |
| windows | C:\Users\\%USERNAME%\\.kaggle\logs\kagglehub.log |

Please include the log to help troubleshoot issues.
