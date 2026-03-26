#!/bin/bash
# Start IronClaw with correct port configuration for Telegram webhook

set -e

echo "=== Starting IronClaw with Telegram Webhook Support ==="
echo ""

# Set environment variables
export LLM_BACKEND=openai_compatible
export LLM_BASE_URL=https://integrate.api.nvidia.com/v1
export LLM_API_KEY=$NGC_KEY
export LLM_MODEL=z-ai/glm5
export GATEWAY_PORT=3004
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN must be set}"
export TUNNEL_URL=https://ironclaw.kutasi.dev
export HTTP_PORT=8081
export HTTP_WEBHOOK_SECRET="f728634e611a3bc25df6c8790bd80b15b5b9d381a1740aa492a02031b9ee3bcc"

echo "Environment configured:"
echo "  - GATEWAY_PORT: $GATEWAY_PORT (Web UI)"
echo "  - HTTP_PORT: $HTTP_PORT (Webhook server)"
echo "  - TUNNEL_URL: $TUNNEL_URL"
echo "  - LLM_MODEL: $LLM_MODEL"
echo ""

# Check if cloudflared tunnel is running
if ! pgrep -f "cloudflared.*${TUNNEL_URL}" > /dev/null; then
    echo "⚠️  Cloudflare tunnel not running!"
    echo "Starting tunnel..."
    
    CONFIG_FILE="/home/bkutasi/.cloudflared/ironclaw-config.yml"
    if [ -f "$CONFIG_FILE" ]; then
        nohup cloudflared tunnel --config "$CONFIG_FILE" run > /tmp/cloudflared-ironclaw-stdout.log 2>&1 &
        echo "✓ Tunnel started"
        sleep 3
    else
        echo "❌ Tunnel config not found: $CONFIG_FILE"
        echo "Run fix-telegram-tunnel.sh first"
        exit 1
    fi
else
    echo "✓ Cloudflare tunnel already running"
fi

# Verify tunnel is connected
echo ""
echo "Checking tunnel connectivity..."
for i in {1..10}; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://ironclaw.kutasi.dev/" 2>&1 || echo "000")
    if [ "$HTTP_CODE" != "000" ]; then
        echo "✓ Tunnel is connected (HTTP $HTTP_CODE)"
        break
    fi
    sleep 1
done

if [ "$HTTP_CODE" = "000" ]; then
    echo "⚠️  Tunnel may not be fully connected yet"
fi

# Start IronClaw
echo ""
echo "Starting IronClaw..."
echo "Logs will be written to: /tmp/ironclaw.log"
echo ""

cd /media/nvme/projects/ironclaw
./target/release/ironclaw 2>&1 | tee /tmp/ironclaw.log
