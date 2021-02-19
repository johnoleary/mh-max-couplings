
FROM python:3.9.1-buster as builder

RUN apt-get update && apt-get install -y --no-install-recommends --yes python3-venv libpython3-dev && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip

FROM builder as builder-venv

COPY requirements.txt /requirements.txt
RUN /venv/bin/pip3 install -r /requirements.txt

FROM builder-venv as runner

COPY . /app
WORKDIR /app

USER 1001
