###############################################################################
# Compute: Training — GPU spot instance (count-gated)
# Set launch_training = true to create; false (default) to skip
###############################################################################

resource "aws_spot_instance_request" "training" {
  count = var.launch_training ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.training_instance_type
  spot_price             = var.training_spot_max_price
  wait_for_fulfillment   = false
  spot_type              = "one-time"
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = var.enable_ssh_access ? [aws_security_group.training.id, aws_security_group.ssh.id] : [aws_security_group.training.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  metadata_options {
    http_tokens                 = "required"
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-training"
    Role = "training"
  }

  timeouts {
    create = "30m"
  }

  # User data — Deep Learning AMI has NVIDIA drivers pre-installed
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    echo "Denver Sprinkler LLM — Training Instance Bootstrap"
    echo "Instance started at $(date -u)" >> /var/log/training-bootstrap.log
    # Deep Learning AMI includes NVIDIA drivers and CUDA
    # TODO: Pull training scripts from S3 or git
    # TODO: Start training job
    # TODO: Upload model artifacts to S3 and signal completion
  EOF
}
