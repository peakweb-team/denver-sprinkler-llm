###############################################################################
# Input Variables
###############################################################################

# --- Region & Naming ---

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name used as prefix for resource naming"
  type        = string
  default     = "denver-sprinkler-llm"
}

# --- Networking ---

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for the public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "availability_zone" {
  description = "Availability zone for the public subnet"
  type        = string
  default     = "us-west-2a"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into instances (set to your IP/32)"
  type        = string
  default     = "0.0.0.0/0"
}

# --- Storage ---

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for model artifacts (must be globally unique)"
  type        = string
}

# --- Compute: Training ---

variable "launch_training" {
  description = "Set to true to launch the GPU training spot instance"
  type        = bool
  default     = false
}

variable "training_instance_type" {
  description = "EC2 instance type for training (GPU instance)"
  type        = string
  default     = "g5.xlarge"
}

variable "training_spot_max_price" {
  description = "Maximum hourly price for the training spot instance (USD)"
  type        = string
  default     = "0.50"
}

# --- Compute: Inference ---

variable "launch_inference" {
  description = "Set to true to launch the always-on inference instance"
  type        = bool
  default     = false
}

variable "inference_instance_type" {
  description = "EC2 instance type for inference (CPU instance)"
  type        = string
  default     = "t3.medium"
}

# --- SSH ---

variable "enable_ssh_access" {
  description = "Attach SSH security group to instances (set to true and configure allowed_ssh_cidr for SSH access)"
  type        = bool
  default     = false
}

variable "ssh_key_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
  default     = ""
}

# --- Budget ---

variable "budget_limit" {
  description = "Monthly budget limit in USD"
  type        = string
  default     = "50.0"
}

variable "alert_email" {
  description = "Email address for budget and alarm notifications"
  type        = string
}
