FROM ubuntu:24.04

LABEL authors="ParrotXray"
LABEL description="Mantis for training models"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

ENV DATAPATH=""
ENV ALL="false"
ENV DATAPREPROCESS="false"
ENV DEEPAUTOENCODER="false"
ENV EXPORT="false"
ENV FEATUREANALYSIS="false"
ENV FINETUNE="false"
ENV SCORE="false"
ENV ARGS=""
ENV API="false"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"
ENV NODE_ID=""

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    tini \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv "$VIRTUAL_ENV"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chmod=755 entrypoint.sh .
COPY src/ ./src/
RUN chmod 755 ./src/main.py ./src/server.py

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "./entrypoint.sh"]
CMD []
