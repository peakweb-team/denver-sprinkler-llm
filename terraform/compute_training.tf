###############################################################################
# Compute: Training — GPU spot instance (count-gated)
# Set launch_training = true to create; false (default) to skip
###############################################################################

resource "aws_spot_instance_request" "training" {
  count = var.launch_training ? 1 : 0

  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.training_instance_type
  spot_price             = var.training_spot_max_price
  wait_for_fulfillment   = true
  spot_type              = "one-time"
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ssh.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-training"
    Role = "training"
  }

  # User data placeholder — install NVIDIA drivers, pull training scripts, etc.
  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -euo pipefail
    echo "Denver Sprinkler LLM — Training Instance Bootstrap"
    echo "Instance started at $(date -u)" >> /var/log/training-bootstrap.log
    # TODO: Install NVIDIA drivers, CUDA, Python, training dependencies
    # TODO: Pull training scripts from S3 or git
    # TODO: Start training job
  EOF
  )
}
