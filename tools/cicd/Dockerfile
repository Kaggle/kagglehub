# This Docker image is a modified version of the python image that
# includes [hatch](https://hatch.pypa.io/latest/)
# and [twine](https://pypi.org/project/twine/)

ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}

RUN python -m pip install --upgrade pip

RUN python -m pip install hatch==1.14.0 twine==6.0.1

ENTRYPOINT ["hatch"]