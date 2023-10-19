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

You have 3 different options to authenticate

#### Option 1: Calling kagglehub.login()

```
# TODO(b/305808471): Not yet implemented.
kagglehub.login()
```

#### Option 2: Read credentials from environment variables

You can also choose to export your Kaggle username and token to the environment:

```
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx
```

#### Option 3: Read credentials from `kaggle.json`

Store your `kaggle.json` credentials file at `~/.kaggle/kaggle.json`.

Alternatively, you can set the `KAGGLE_CONFIG_DIR` environment variable to change this location to `$KAGGLE_CONFIG_DIR/kaggle.json`.

Note for Windows users: The default directory is `%HOMEPATH%/kaggle.json`.

## Development

### Prequisites

We use [hatch](https://hatch.pypa.io) to manage this project.

Follow these [instructions](https://hatch.pypa.io/latest/install/) to install it.

### Tests

```
# Run all tests
hatch run test

# Run a single test file
hatch run test tests/test_<SOME_FILE>.py
```

### Lint / Format

```
# Lint check
hatch run lint:style
hatch run lint:typing
hatch run lint:all     # for both

# Format
hatch run lint:fmt
```

### Coverage report

```
hatch cov
```

### Build

```
hatch build
```