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

The following examples download the `answer-equivalence-bem` variation of this Kaggle model: https://www.kaggle.com/models/google/bert/frameworks/tensorFlow2/variations/answer-equivalence-bem

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