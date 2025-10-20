# Base image: Ubuntu LTS con utilidades de devcontainers
FROM mcr.microsoft.com/devcontainers/base:ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    lsb-release \
    gnupg \
    wget \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-distutils \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN mkdir -p /home/vscode && chown -R vscode:0 /home/vscode

ENV PATH=/home/vscode/.local/bin:$PATH
WORKDIR /workspaces
