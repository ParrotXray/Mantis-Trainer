FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

LABEL authors="ParrotXray"
LABEL description="NetGuardia for training models"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

ENV DATASET=""
ENV DATAPATH=""
ENV ALL="false"
ENV DATAPREPROCESS="false"
ENV DEEPAUTOENCODER="false"
ENV CLASSIFIER="false"
ENV EXPORT="false"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    python3.12-dev \
    build-essential \
    git \
    tini \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir --break-system-packages \
    -i https://pypi.org/simple -r requirements.txt

COPY src/ ./src/
COPY entrypoint.sh .

RUN chmod +x ./entrypoint.sh ./src/main.py

ENTRYPOINT ["/usr/bin/tini", "--", "./entrypoint.sh"]
CMD []
