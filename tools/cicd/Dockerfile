# This Docker image is a modified version of the python image that
# includes [hatch](https://hatch.pypa.io/latest/)

ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}

RUN python -m pip install hatch

ENTRYPOINT ["hatch"]