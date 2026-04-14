#!/usr/bin/env bash
# =============================================================================
# Denver Sprinkler LLM — Deploy Inference Server
# Wraps terraform init/plan/apply with safety checks and health verification
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$SCRIPT_DIR/../terraform" && pwd)"
HEALTH_TIMEOUT=300  # 5 minutes
HEALTH_INTERVAL=10  # seconds between checks

echo "=== Denver Sprinkler LLM — Inference Deployment ==="
echo ""

# ---------- Terraform init ----------
echo ">> Running terraform init..."
cd "$TF_DIR"
terraform init

# ---------- Terraform plan ----------
echo ""
echo ">> Running terraform plan..."
terraform plan -var="launch_inference=true" -out=inference.plan

echo ""
echo "============================================="
echo "  Estimated monthly cost: ~\$34/month"
echo "  Budget limit: \$50/month"
echo "============================================="
echo ""
read -rp "Apply this plan? (y/N) " confirm

if [[ "${confirm,,}" != "y" ]]; then
  echo "Aborted. No changes applied."
  rm -f inference.plan
  exit 0
fi

# ---------- Terraform apply ----------
echo ""
echo ">> Running terraform apply..."
terraform apply inference.plan
rm -f inference.plan

# ---------- Get outputs ----------
ELASTIC_IP=$(terraform output -raw inference_elastic_ip)
HEALTH_URL="http://${ELASTIC_IP}/health"

echo ""
echo ">> Inference instance deployed."
echo "   Elastic IP: ${ELASTIC_IP}"
echo "   Health URL: ${HEALTH_URL}"

# ---------- Wait for instance + health check ----------
echo ""
echo ">> Waiting for health check to pass (timeout: ${HEALTH_TIMEOUT}s)..."

elapsed=0
while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
  if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    echo ""
    echo "=== Health check PASSED ==="
    echo ""
    echo "Inference server is live at: http://${ELASTIC_IP}"
    echo "Health endpoint: ${HEALTH_URL}"
    echo ""
    echo "To run smoke tests:  ./scripts/smoke-test.sh"
    exit 0
  fi

  echo "  Health check attempt $((elapsed / HEALTH_INTERVAL + 1)) — waiting ${HEALTH_INTERVAL}s..."
  sleep "$HEALTH_INTERVAL"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

echo ""
echo "WARNING: Health check did not pass within ${HEALTH_TIMEOUT} seconds."
echo "The instance may still be bootstrapping (Docker build takes ~10-15 min)."
echo "Check manually: curl ${HEALTH_URL}"
echo "SSH in and check: journalctl -u inference.service -f"
exit 1
