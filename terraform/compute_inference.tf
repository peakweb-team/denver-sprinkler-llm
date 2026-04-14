###############################################################################
# Compute: Inference — Always-on CPU instance with Elastic IP (count-gated)
# Set launch_inference = true to create; false (default) to skip
###############################################################################

resource "aws_instance" "inference" {
  count = var.launch_inference ? 1 : 0

  ami                    = data.aws_ami.ubuntu_2404.id
  instance_type          = var.inference_instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = var.enable_ssh_access ? [aws_security_group.ssh.id, aws_security_group.inference.id] : [aws_security_group.inference.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  metadata_options {
    http_tokens                 = "required"
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-inference"
    Role = "inference"
  }

  user_data = templatefile("${path.module}/templates/inference-userdata.sh.tpl", {
    s3_bucket_name   = var.s3_bucket_name
    s3_model_prefix  = "models/denver-sprinkler-3b-1bit"
    inference_domain = var.inference_domain
    repo_url         = "peakweb-team/denver-sprinkler-llm"
    alert_email      = var.alert_email
  })
}

# --- Elastic IP for stable public address ---

resource "aws_eip" "inference" {
  count  = var.launch_inference ? 1 : 0
  domain = "vpc"

  tags = {
    Name = "${var.project_name}-inference-eip"
  }
}

resource "aws_eip_association" "inference" {
  count         = var.launch_inference ? 1 : 0
  instance_id   = aws_instance.inference[0].id
  allocation_id = aws_eip.inference[0].id
}

# --- CloudWatch Alarms ---

resource "aws_cloudwatch_metric_alarm" "inference_high_cpu" {
  count = var.launch_inference ? 1 : 0

  alarm_name          = "${var.project_name}-inference-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Inference instance CPU utilization > 80% for 10 minutes"
  alarm_actions       = [aws_sns_topic.budget_alerts.arn]
  ok_actions          = [aws_sns_topic.budget_alerts.arn]

  dimensions = {
    InstanceId = aws_instance.inference[0].id
  }

  tags = {
    Name = "${var.project_name}-inference-high-cpu"
  }
}

resource "aws_cloudwatch_metric_alarm" "inference_high_disk" {
  count = var.launch_inference ? 1 : 0

  alarm_name          = "${var.project_name}-inference-high-disk"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "disk_used_percent"
  namespace           = "DenverSprinklerLLM"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "Inference instance disk utilization > 85%"
  alarm_actions       = [aws_sns_topic.budget_alerts.arn]
  ok_actions          = [aws_sns_topic.budget_alerts.arn]

  dimensions = {
    InstanceId = aws_instance.inference[0].id
    path       = "/"
    fstype     = "ext4"
  }

  tags = {
    Name = "${var.project_name}-inference-high-disk"
  }
}
