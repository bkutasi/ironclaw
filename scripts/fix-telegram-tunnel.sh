#!/bin/bash
# Fix Telegram webhook by properly configuring Cloudflare tunnel
# This creates a proper config file and restarts the tunnel

set -e

TUNNEL_ID="aa56b30c-d046-4e01-9ad3-4ec110d696a0"
TUNNEL_NAME="ironclaw"
CREDENTIALS_FILE="/home/bkutasi/.cloudflared/${TUNNEL_ID}.json"
CONFIG_FILE="/home/bkutasi/.cloudflared/${TUNNEL_NAME}-config.yml"

echo "=== Fixing Cloudflare Tunnel Configuration ==="
echo ""

# Check if credentials file exists
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "❌ Tunnel credentials not found: $CREDENTIALS_FILE"
    exit 1
fi

echo "✓ Tunnel credentials found"

# Create the config file
cat > "$CONFIG_FILE" << EOF
# Cloudflare Tunnel Configuration for IronClaw
# Routes ironclaw.kutasi.dev to local webhook server

tunnel: ${TUNNEL_ID}
credentials-file: ${CREDENTIALS_FILE}

# Logging
logfile: /tmp/cloudflared-ironclaw.log
loglevel: info

# Ingress rules - route traffic to local services
ingress:
  # Route Telegram webhook to local HTTP server
  - hostname: ironclaw.kutasi.dev
    path: /webhook/telegram
    service: http://localhost:8081
    
  # Route HTTP channel (if needed)
  - hostname: ironclaw.kutasi.dev
    path: /webhook/*
    service: http://localhost:8081
    
  # Route gateway/web UI
  - hostname: ironclaw.kutasi.dev
    service: http://localhost:3004
    
  # Default: return 404 for unmatched routes
  - service: http_status:404
EOF

echo "✓ Created config file: $CONFIG_FILE"
echo ""
echo "Config contents:"
echo "---"
cat "$CONFIG_FILE"
echo "---"
echo ""

# Kill any existing cloudflared processes for this tunnel
echo "Stopping any existing cloudflared processes..."
pkill -f "cloudflared.*${TUNNEL_ID}" 2>/dev/null || true
pkill -f "cloudflared.*tunnel.*run.*ironclaw" 2>/dev/null || true
sleep 2

# Verify no processes are running
if pgrep -f "cloudflared.*${TUNNEL_ID}" > /dev/null; then
    echo "⚠️  Warning: Some cloudflared processes still running"
    sleep 2
fi

echo "✓ Stopped existing tunnel processes"
echo ""

# Start the tunnel with the new config
echo "Starting tunnel with new configuration..."
nohup cloudflared tunnel --config "$CONFIG_FILE" run > /tmp/cloudflared-ironclaw-stdout.log 2>&1 &
TUNNEL_PID=$!

echo "✓ Tunnel started with PID: $TUNNEL_PID"
echo ""

# Wait for tunnel to connect
echo "Waiting for tunnel to connect (up to 15 seconds)..."
for i in {1..15}; do
    if curl -s -o /dev/null -w "%{http_code}" "https://ironclaw.kutasi.dev/webhook/telegram" 2>/dev/null | grep -q "405\|400\|404"; then
        echo "✓ Tunnel is responding!"
        break
    fi
    sleep 1
done

# Test the webhook endpoint
echo ""
echo "=== Testing Webhook Endpoint ==="
echo ""
echo "Testing POST to /webhook/telegram..."
HTTP_CODE=$(curl -s -o /tmp/webhook-test-response.txt -w "%{http_code}" \
    -X POST "https://ironclaw.kutasi.dev/webhook/telegram" \
    -H "Content-Type: application/json" \
    -d '{"test":true}' \
    2>&1)

echo "HTTP Response Code: $HTTP_CODE"

if [ "$HTTP_CODE" = "405" ] || [ "$HTTP_CODE" = "400" ]; then
    echo "✓ SUCCESS! Webhook endpoint is reachable"
    echo "  (405/400 is expected - Telegram sends different payload)"
elif [ "$HTTP_CODE" = "502" ]; then
    echo "❌ Still getting 502 - tunnel may not be fully connected"
    echo "Check logs: tail -f /tmp/cloudflared-ironclaw.log"
elif [ "$HTTP_CODE" = "000" ]; then
    echo "❌ Connection failed - tunnel not reachable"
    echo "Check logs: tail -f /tmp/cloudflared-ironclaw.log"
else
    echo "? Unexpected response code: $HTTP_CODE"
fi

echo ""
echo "=== Next Steps ==="
echo ""
echo "1. Restart IronClaw:"
echo "   ./target/release/ironclaw"
echo ""
echo "2. Watch for these log messages:"
echo "   - 'has_webhook_secret=false' (OK for now)"
echo "   - 'Webhook registered successfully: https://ironclaw.kutasi.dev/webhook/telegram'"
echo ""
echo "3. Test by sending a message to your Telegram bot"
echo ""
echo "4. (Optional) To make tunnel persistent, add to systemd:"
echo "   sudo systemctl edit cloudflared-ironclaw"
echo ""
echo "Tunnel logs: tail -f /tmp/cloudflared-ironclaw.log"
echo "IronClaw logs: ./target/release/ironclaw 2>&1 | tee ironclaw.log"
echo ""

# Save the config file location
echo "$CONFIG_FILE" > /tmp/ironclaw-tunnel-config.txt
echo "Config file path saved to /tmp/ironclaw-tunnel-config.txt"
