# Kaggle Hub Client Library

## Installation

Install the `kagglehub` package with pip:

```
pip install kagglehub
```

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