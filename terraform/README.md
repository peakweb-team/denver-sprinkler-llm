# Terraform — Denver Sprinkler LLM AWS Infrastructure

POC-grade AWS infrastructure for training and deploying the Denver Sprinkler LLM.

## Architecture

- **VPC** with a single public subnet (no NAT gateway — POC simplicity)
- **S3 bucket** with versioning for model artifacts
- **GPU spot instance** for training (count-gated, off by default)
- **CPU on-demand instance** for inference with Elastic IP (count-gated, off by default)
- **IAM role** scoped to S3, CloudWatch, and EC2 describe
- **Budget alarm** at $50/month with email notifications

## Prerequisites

1. [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5.0
2. AWS credentials configured via one of:
   - Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - AWS CLI profile: `aws configure`
   - IAM instance profile (if running inside AWS)

## Quick Start

```bash
cd terraform/

# 1. Initialize Terraform
terraform init

# 2. Copy and edit the variables file
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values (bucket name, email, etc.)

# 3. Preview changes
terraform plan

# 4. Apply (creates VPC, S3, IAM, budget alarm — no instances by default)
terraform apply
```

## Launching Instances

Instances are **off by default** to avoid costs. Enable them by setting variables:

```bash
# Launch the inference server
terraform apply -var="launch_inference=true"

# Launch a training spot instance
terraform apply -var="launch_training=true"

# Tear down training after completion (keep inference running)
terraform apply -var="launch_training=false" -var="launch_inference=true"
```

## Tear Down Everything

```bash
terraform destroy
```

## Cost Estimates

| Resource | Estimated Monthly Cost |
|----------|----------------------|
| VPC + networking | Free |
| S3 bucket (small usage) | < $1 |
| Budget alarm + SNS | Free |
| t3.medium inference (on-demand, 24/7) | ~$30 |
| g5.xlarge training (spot, ~10 hrs) | ~$5-8 |
| Elastic IP (while attached) | Free |
| **Total (inference running)** | **~$35** |

## Files

| File | Purpose |
|------|---------|
| `main.tf` | Provider config, terraform block, data sources |
| `variables.tf` | All input variables with descriptions and defaults |
| `outputs.tf` | Resource identifiers exposed for other components |
| `terraform.tfvars.example` | Documented example variables file |
| `networking.tf` | VPC, subnet, internet gateway, route table, security groups |
| `storage.tf` | S3 bucket with versioning and encryption |
| `iam.tf` | IAM role + instance profile with least-privilege policies |
| `compute_training.tf` | GPU spot instance for model training |
| `compute_inference.tf` | CPU on-demand instance + Elastic IP for inference |
| `budgets.tf` | $50/month budget alarm with SNS email notification |

## Security Notes

- SSH access is controlled by `allowed_ssh_cidr` — restrict to your IP
- S3 bucket has public access blocked and server-side encryption
- IAM policies follow least-privilege (scoped to specific bucket and resources)
- Never commit `terraform.tfvars` or `.tfstate` files to git
