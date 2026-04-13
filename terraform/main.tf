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

# Current AWS account and region info
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
