ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}

RUN pip install hatch

# Add only the minimal files required to be able to pre-create the hatch environments.
# If any of these files changes, a new Docker build is necessary. This is why we need
# to only include the minimal set of files.
ADD pyproject.toml /working/pyproject.toml
ADD LICENSE /working/LICENSE
ADD README.md /working/README.md
ADD src/kagglehub/__init__.py /working/src/kagglehub/__init__.py
WORKDIR /working

# Pre-create the hatch environments.
# This drastically cut the time to run commands with the `docker-hatch` wrapper
#  since the creation of the environments (including syncing dependencies) is
# only done once when building this image and is skipped later.
RUN hatch env create default
RUN hatch env create lint

ENTRYPOINT ["hatch"]