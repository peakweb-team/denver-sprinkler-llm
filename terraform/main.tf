###############################################################################
# Denver Sprinkler LLM — Terraform Root Configuration
# POC-grade AWS infrastructure for training and inference
###############################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state only — appropriate for POC
  # For production, migrate to S3 backend with DynamoDB locking
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "denver-sprinkler-llm"
      Environment = "poc"
      ManagedBy   = "terraform"
    }
  }
}

# Latest Amazon Linux 2023 AMI for the target region
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# AWS Deep Learning AMI for GPU training instances (NVIDIA drivers + CUDA pre-installed)
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver AMI (Amazon Linux 2) Version *"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# Ubuntu 24.04 LTS AMI for inference instance
data "aws_ami" "ubuntu_2404" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# Current AWS account and region info
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
