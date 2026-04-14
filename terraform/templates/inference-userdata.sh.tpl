#!/bin/bash
# =============================================================================
# Denver Sprinkler LLM — Inference Instance Bootstrap (user-data)
# Installs Docker, CloudWatch agent, nginx, builds and runs inference container
# =============================================================================
set -euo pipefail
exec > >(tee /var/log/inference-bootstrap.log) 2>&1

echo "=== Inference bootstrap started at $(date -u) ==="

# ---------- System updates ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

# ---------- Install Docker CE ----------
apt-get install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
systemctl enable docker
systemctl start docker

# ---------- Install CloudWatch Agent ----------
wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb
rm -f amazon-cloudwatch-agent.deb

# CloudWatch agent configuration for memory, disk, and CPU
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'CWCONFIG'
{
  "metrics": {
    "namespace": "DenverSprinklerLLM",
    "metrics_collected": {
      "mem": {
        "measurement": ["mem_used_percent"],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": ["disk_used_percent"],
        "resources": ["/"],
        "metrics_collection_interval": 60
      },
      "cpu": {
        "measurement": ["cpu_usage_active"],
        "metrics_collection_interval": 60,
        "totalcpu": true
      }
    },
    "append_dimensions": {
      "InstanceId": "$${aws:InstanceId}"
    }
  }
}
CWCONFIG

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
  -s

# ---------- Install nginx and certbot ----------
apt-get install -y nginx certbot python3-certbot-nginx

# ---------- Clone repo and build Docker image ----------
apt-get install -y git
cd /opt
git clone https://github.com/${repo_url}.git denver-sprinkler-llm
cd /opt/denver-sprinkler-llm
docker build -t denver-sprinkler-inference:latest -f server/Dockerfile .

# ---------- Create model directory ----------
mkdir -p /opt/models
chmod 755 /opt/models

# ---------- Create systemd service for inference container ----------
cat > /etc/systemd/system/inference.service <<'SVCEOF'
[Unit]
Description=Denver Sprinkler LLM Inference Server
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker stop inference
ExecStartPre=-/usr/bin/docker rm inference
ExecStart=/usr/bin/docker run --name inference \
  --mount type=bind,source=/opt/models,target=/models \
  -e S3_BUCKET_NAME=${s3_bucket_name} \
  -e S3_MODEL_PREFIX=${s3_model_prefix} \
  -p 127.0.0.1:8000:8000 \
  denver-sprinkler-inference:latest
ExecStop=/usr/bin/docker stop inference

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable inference.service
systemctl start inference.service

# ---------- Configure nginx reverse proxy ----------
INFERENCE_DOMAIN="${inference_domain}"

if [ -n "$INFERENCE_DOMAIN" ]; then
  # HTTPS mode with certbot
  cat > /etc/nginx/sites-available/inference <<NGINXEOF
server {
    listen 80;
    server_name $INFERENCE_DOMAIN;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name $INFERENCE_DOMAIN;

    # certbot will fill in ssl_certificate and ssl_certificate_key
    ssl_certificate /etc/letsencrypt/live/$INFERENCE_DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$INFERENCE_DOMAIN/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
    }
}
NGINXEOF

  rm -f /etc/nginx/sites-enabled/default
  ln -sf /etc/nginx/sites-available/inference /etc/nginx/sites-enabled/inference

  # Start nginx first so certbot can use the HTTP challenge
  systemctl restart nginx

  # Obtain SSL certificate
  certbot --nginx -d "$INFERENCE_DOMAIN" --non-interactive --agree-tos -m "${alert_email}" --redirect

  # Certbot auto-renewal is installed by default via systemd timer
  systemctl enable certbot.timer
  systemctl start certbot.timer

else
  # HTTP-only mode — nginx reverse proxy on port 80
  cat > /etc/nginx/sites-available/inference <<'NGINXEOF'
server {
    listen 80 default_server;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
NGINXEOF

  rm -f /etc/nginx/sites-enabled/default
  ln -sf /etc/nginx/sites-available/inference /etc/nginx/sites-enabled/inference
  systemctl restart nginx
fi

# ---------- Wait for health check ----------
echo "Waiting for inference server to become healthy..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "=== Health check passed at $(date -u) ==="
    exit 0
  fi
  echo "Health check attempt $i/30 — waiting 10s..."
  sleep 10
done

echo "WARNING: Health check did not pass within 5 minutes. Check container logs."
echo "=== Bootstrap completed at $(date -u) ==="
