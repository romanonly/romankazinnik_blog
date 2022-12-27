FROM python:3.6.10
WORKDIR /usr/app

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip
RUN cat /tmp/requirements.txt | xargs -n 1 pip install
COPY metrics/*.py ./metrics/

