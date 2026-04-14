#!/usr/bin/env bash
# =============================================================================
# Denver Sprinkler LLM — Smoke Test
# Verifies inference server health and chat endpoints after deployment
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$SCRIPT_DIR/../terraform" && pwd)"

# ---------- Get endpoint from Terraform or argument ----------
if [ -n "${1:-}" ]; then
  ENDPOINT="$1"
else
  cd "$TF_DIR"
  ELASTIC_IP=$(terraform output -raw inference_elastic_ip 2>/dev/null || true)
  if [ -z "$ELASTIC_IP" ]; then
    echo "ERROR: Could not determine inference IP. Pass endpoint as argument:"
    echo "  $0 http://<ip-or-domain>"
    exit 1
  fi
  ENDPOINT="http://${ELASTIC_IP}"
fi

echo "=== Denver Sprinkler LLM — Smoke Test ==="
echo "Endpoint: ${ENDPOINT}"
echo ""

PASS=0
FAIL=0

# ---------- Test 1: Health check ----------
echo ">> Test 1: GET /health"
HTTP_CODE=$(curl -sf -o /tmp/health_response.json -w "%{http_code}" "${ENDPOINT}/health" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  echo "   PASS — HTTP 200"
  echo "   Response:"
  cat /tmp/health_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/health_response.json
  PASS=$((PASS + 1))
else
  echo "   FAIL — HTTP ${HTTP_CODE}"
  FAIL=$((FAIL + 1))
fi
echo ""

# ---------- Test 2: Chat endpoint ----------
echo ">> Test 2: POST /v1/chat"
HTTP_CODE=$(curl -sf -o /tmp/chat_response.json -w "%{http_code}" \
  -X POST "${ENDPOINT}/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What services do you offer?"}' 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  RESPONSE_LENGTH=$(wc -c < /tmp/chat_response.json)
  if [ "$RESPONSE_LENGTH" -gt 2 ]; then
    echo "   PASS — HTTP 200, response length: ${RESPONSE_LENGTH} bytes"
    echo "   Response:"
    cat /tmp/chat_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/chat_response.json
    PASS=$((PASS + 1))
  else
    echo "   FAIL — HTTP 200 but empty response"
    FAIL=$((FAIL + 1))
  fi
else
  echo "   FAIL — HTTP ${HTTP_CODE}"
  FAIL=$((FAIL + 1))
fi
echo ""

# ---------- Summary ----------
echo "============================================="
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "============================================="

# Clean up temp files
rm -f /tmp/health_response.json /tmp/chat_response.json

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
exit 0
