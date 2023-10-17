# This Docker image is a modified version of the python image that
# includes [hatch](https://hatch.pypa.io/latest/)

FROM python:3.9.18

RUN python -m pip install hatch

ENTRYPOINT ["hatch"]