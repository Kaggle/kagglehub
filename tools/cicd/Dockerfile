# This Docker image is a modified version of the python image that
# includes [hatch](https://hatch.pypa.io/latest/)
# and [twine](https://pypi.org/project/twine/)

ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}

# Pick up the emSign/eMudhra CA removal (GHSA-6q48-6c7w-9gw2) in the OS trust store and Python tooling.
# hatch uses uv to manage its environments; floor it for GHSA-hq85-5cpj-mr62.
RUN apt-get update && apt-get install -y --only-upgrade ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade "pip>=26.2.2" "certifi>=2026.10.19" hatch==1.16.5 "uv>=0.13.4" twine==6.0.1

ENTRYPOINT ["hatch"]