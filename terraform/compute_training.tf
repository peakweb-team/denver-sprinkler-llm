###############################################################################
# Compute: Training - GPU on-demand instance (count-gated)
# Set launch_training = true to create; false (default) to skip
###############################################################################

resource "aws_instance" "training" {
  count = var.launch_training ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.training_instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = var.enable_ssh_access ? [aws_security_group.training.id, aws_security_group.ssh.id] : [aws_security_group.training.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = local.ssh_key_name

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

  # User data - Deep Learning AMI has NVIDIA drivers pre-installed
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    echo "Denver Sprinkler LLM - Training Instance Bootstrap"
    echo "Instance started at $(date -u)" >> /var/log/training-bootstrap.log
    # Deep Learning AMI includes NVIDIA drivers and CUDA
  EOF
}
