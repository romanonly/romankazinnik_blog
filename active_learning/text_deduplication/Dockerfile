FROM python:3.6.10
WORKDIR /usr/app

COPY requirements.txt /tmp/requirements.txt
#RUN pip install --upgrade pip &&
#RUN pip install -r /tmp/requirements_features.txt
RUN python3 -m pip install --upgrade pip
RUN cat /tmp/requirements.txt | xargs -n 1 pip install
COPY metrics/*.py ./metrics/
# default tnrypoint in python image python3
# ENTRYPOINT [ "python3"]
#ENTRYPOINT ["python3", "/src/cpu_gpu.py"]
#ENTRYPOINT ["python3", "/src/lstm_cpu_gpu_cuda_imbd_v0.py"]
