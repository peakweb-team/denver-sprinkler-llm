###############################################################################
# Compute: Inference — Always-on CPU instance with Elastic IP (count-gated)
# Set launch_inference = true to create; false (default) to skip
###############################################################################

resource "aws_instance" "inference" {
  count = var.launch_inference ? 1 : 0

  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.inference_instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ssh.id, aws_security_group.inference.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-inference"
    Role = "inference"
  }

  # User data placeholder — install bitnet.cpp, FastAPI server, pull model from S3
  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -euo pipefail
    echo "Denver Sprinkler LLM — Inference Instance Bootstrap"
    echo "Instance started at $(date -u)" >> /var/log/inference-bootstrap.log
    # TODO: Install bitnet.cpp, Python, FastAPI dependencies
    # TODO: Pull model artifact from S3
    # TODO: Start FastAPI inference server on port 8000
  EOF
  )
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
