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

After login in, you can download your Kaggle API credentials at https://www.kaggle.com/settings by clicking on the "Create New Token" button under the "API" section.

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
```

## Development

### Prequisites

We use [hatch](https://hatch.pypa.io) to manage this project.

Follow these [instructions](https://hatch.pypa.io/latest/install/) to install it.

### Tests

```sh
# Run all tests
hatch run test

# Run a single test file
hatch run test tests/test_<SOME_FILE>.py
```

### Integration Tests

To run integration tests on your local machine, you need to set up your Kaggle API credentials. You can do this in one of these two ways described in the earlier sections of this document. Refer to the sections: 
- [Using environment variables](#option-2-read-credentials-from-environment-variables)
- [Using credentials file](#option-3-read-credentials-from-kagglejson)

After setting up your credentials by any of these methods, you can run the integration tests as follows:

```sh
# Run all tests
hatch run integration-test
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
hatch cov
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
```
