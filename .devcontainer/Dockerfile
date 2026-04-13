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

# GitHub CLI (for gh issue, gh pr, etc.)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y gh \
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

# Claude Code agent personas and skills
# Copied from host at build time — mount updated versions at runtime if needed
COPY claude-config/ /root/.claude/

WORKDIR /workspace
