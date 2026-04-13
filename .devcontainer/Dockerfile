FROM ghcr.io/anthropics/claude-code:latest

# Python + ML tooling for data prep and local testing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    cmake \
    clang \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

# Terraform
RUN curl -fsSL https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_linux_amd64.zip -o /tmp/terraform.zip \
    && unzip -q /tmp/terraform.zip -d /usr/local/bin \
    && rm /tmp/terraform.zip

# Python ML dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    transformers \
    peft \
    datasets \
    accelerate \
    bitsandbytes \
    scipy \
    fastapi \
    uvicorn[standard] \
    httpx \
    anthropic

WORKDIR /workspace
