# scripts/test-api.ps1 — Quick smoke test for the live API (PowerShell)
#
# Usage:
#   .\scripts\test-api.ps1                                          # defaults
#   .\scripts\test-api.ps1 -BaseUrl https://my-url.onrender.com     # custom URL
#   .\scripts\test-api.ps1 -ApiKey mykey                            # custom key

param(
    [string]$BaseUrl = "https://github-summarizer-api.onrender.com",
    [string]$ApiKey  = "gitSum123"
)

Write-Host "`n══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Testing: $BaseUrl" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════`n" -ForegroundColor Cyan

# ── Health ────────────────────────────────────
Write-Host "▸ GET /api/v1/health" -ForegroundColor Yellow
$health = curl.exe -s "$BaseUrl/api/v1/health"
$health | ConvertFrom-Json | ConvertTo-Json -Depth 5
Write-Host ""

# ── Summarize ─────────────────────────────────
Write-Host "▸ POST /api/v1/summarize  (expressjs/express)" -ForegroundColor Yellow
Write-Host "  (this may take 30-120s for LLM processing...)" -ForegroundColor DarkGray

$body = '{"repo_url": "https://github.com/expressjs/express"}'
$result = curl.exe -s -X POST "$BaseUrl/api/v1/summarize" `
    -H "Content-Type: application/json" `
    -H "Authorization: Bearer $ApiKey" `
    -d $body

try {
    $result | ConvertFrom-Json | ConvertTo-Json -Depth 10
} catch {
    Write-Host $result
}

Write-Host "`n══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Done" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
