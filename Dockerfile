ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}

RUN pip install hatch

ENTRYPOINT ["hatch"]