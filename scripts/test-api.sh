#!/usr/bin/env bash
# scripts/test-api.sh — Quick smoke test for the live API
#
# Usage:
#   bash scripts/test-api.sh                                    # defaults
#   bash scripts/test-api.sh https://my-url.onrender.com mykey  # custom
#
# On Windows (PowerShell):
#   bash scripts/test-api.sh
#   # or directly with curl.exe:
#   curl.exe -s https://github-summarizer-api.onrender.com/api/v1/health

set -euo pipefail

BASE_URL="${1:-https://github-summarizer-api.onrender.com}"
API_KEY="${2:-gitSum123}"

echo "══════════════════════════════════════════"
echo "  Testing: $BASE_URL"
echo "══════════════════════════════════════════"

# ── Health ────────────────────────────────────
echo ""
echo "▸ GET /api/v1/health"
curl -s "$BASE_URL/api/v1/health" | python -m json.tool 2>/dev/null || curl -s "$BASE_URL/api/v1/health"

# ── Summarize (small repo for speed) ──────────
echo ""
echo "▸ POST /api/v1/summarize  (expressjs/express)"
echo "  (this may take 30-120s for LLM processing...)"
curl -s -X POST "$BASE_URL/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"repo_url": "https://github.com/expressjs/express"}' \
  | python -m json.tool 2>/dev/null || echo "(raw response above)"

echo ""
echo "══════════════════════════════════════════"
echo "  Done"
echo "══════════════════════════════════════════"
