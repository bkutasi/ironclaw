#!/bin/bash
# Start IronClaw with Telegram Polling Mode (FIXED)
# Clears tunnel URL from database to force polling mode

set -e

echo "=== Starting IronClaw with Telegram Polling Mode (Fixed) ==="
echo ""

# Set environment variables
export LLM_BACKEND=openai_compatible
export LLM_BASE_URL=https://integrate.api.nvidia.com/v1
export LLM_API_KEY=$NGC_KEY
export LLM_MODEL=z-ai/glm5
export GATEWAY_PORT=3004
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN must be set}"
export HTTP_PORT=8081

# DO NOT set TUNNEL_URL - this prevents webhook mode detection

echo "Environment configured:"
echo "  - GATEWAY_PORT: $GATEWAY_PORT (Web UI)"
echo "  - HTTP_PORT: $HTTP_PORT (Webhook server)"
echo "  - TUNNEL_URL: (not set - forces polling mode)"
echo "  - LLM_MODEL: $LLM_MODEL"
echo ""

# Delete webhook first
echo "Step 1: Deleting Telegram webhook..."
curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook" \
    -H "Content-Type: application/json" \
    -d '{}' > /dev/null
echo "✓ Webhook deleted"

# Clear tunnel URL from database
echo ""
echo "Step 2: Clearing tunnel URL from database..."

DB_URL="postgres://postgres:yourpass@localhost:5433/ironclaw"

# Clear tunnel_url configs
psql "$DB_URL" -c "
DELETE FROM workspace_kv 
WHERE key IN (
    'channels/telegram/state/tunnel_url',
    'channels/telegram/tunnel_url',
    'telegram/tunnel_url'
);
" > /dev/null 2>&1 && echo "✓ Cleared tunnel_url from database" || echo "⚠️  Could not clear (may already be cleared)"

# Wait for changes
sleep 1

# Verify webhook status
echo ""
echo "Step 3: Verifying webhook status..."
WEBHOOK_INFO=$(curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo")
WEBHOOK_URL=$(echo "$WEBHOOK_INFO" | jq -r '.result.url // "empty"')
PENDING_COUNT=$(echo "$WEBHOOK_INFO" | jq -r '.result.pending_update_count // 0')

if [ "$WEBHOOK_URL" == "" ] || [ "$WEBHOOK_URL" == "null" ]; then
    echo "✓ Webhook: DELETED (polling mode ready)"
else
    echo "⚠️  Webhook still set: $WEBHOOK_URL"
fi
echo "  Pending updates: $PENDING_COUNT"
echo ""

# Start IronClaw
echo "Step 4: Starting IronClaw..."
echo "Logs: /tmp/ironclaw.log"
echo ""
echo "EXPECTED LOG MESSAGE:"
echo "  'DEBUG Polling mode enabled (no tunnel configured)'"
echo ""
echo "WRONG MESSAGE (webhook mode):"
echo "  'DEBUG Webhook mode enabled (tunnel configured)'"
echo ""
echo "---"
echo ""

cd /media/nvme/projects/ironclaw
./target/release/ironclaw 2>&1 | tee /tmp/ironclaw.log
