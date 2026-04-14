###############################################################################
# Outputs
###############################################################################

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "iam_instance_profile_name" {
  description = "Name of the IAM instance profile for EC2 instances"
  value       = aws_iam_instance_profile.ec2_profile.name
}

output "training_spot_request_id" {
  description = "ID of the training spot instance request (empty if not launched)"
  value       = var.launch_training ? aws_spot_instance_request.training[0].id : ""
}

output "inference_instance_id" {
  description = "ID of the inference instance (empty if not launched)"
  value       = var.launch_inference ? aws_instance.inference[0].id : ""
}

output "inference_elastic_ip" {
  description = "Elastic IP of the inference instance (empty if not launched)"
  value       = var.launch_inference ? aws_eip.inference[0].public_ip : ""
}

output "budget_sns_topic_arn" {
  description = "ARN of the SNS topic for budget alerts"
  value       = aws_sns_topic.budget_alerts.arn
}

output "health_check_url" {
  description = "URL for the inference server health check endpoint"
  value       = var.launch_inference ? "http://${aws_eip.inference[0].public_ip}/health" : ""
}
