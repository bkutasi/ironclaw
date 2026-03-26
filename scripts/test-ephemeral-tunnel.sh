#!/bin/bash
# Quick test to verify ephemeral tunnel setup works

set -e

echo "=== Testing Ephemeral Tunnel Setup ==="
echo ""

# Kill any existing cloudflared on port 8081
echo "Cleaning up existing processes..."
pkill -f "cloudflared.*localhost:8081" 2>/dev/null || true
sleep 1

# Start ephemeral tunnel
echo "Starting ephemeral tunnel..."
cloudflared tunnel --url http://localhost:8081 > /tmp/test-ephemeral.log 2>&1 &
TUNNEL_PID=$!
echo "✓ Cloudflared started (PID: $TUNNEL_PID)"

# Wait for URL
echo "Waiting for tunnel URL..."
for i in {1..15}; do
    TUNNEL_URL=$(grep -oP 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' /tmp/test-ephemeral.log | head -1 || true)
    if [ -n "$TUNNEL_URL" ]; then
        echo "✓ Tunnel connected: $TUNNEL_URL"
        break
    fi
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ Failed to get tunnel URL"
    cat /tmp/test-ephemeral.log
    kill $TUNNEL_PID
    exit 1
fi

# Test the tunnel endpoint
echo ""
echo "Testing tunnel endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$TUNNEL_URL/webhook/telegram" \
    -H "Content-Type: application/json" \
    -d '{"update_id":999}' 2>&1 || echo "000")

echo "Response: HTTP $HTTP_CODE"

if [ "$HTTP_CODE" == "200" ] || [ "$HTTP_CODE" == "404" ]; then
    echo "✅ Tunnel is working! Endpoint is reachable"
else
    echo "⚠️  Unexpected response code: $HTTP_CODE"
    echo "This is expected if IronClaw is not running"
fi

# Cleanup
echo ""
echo "Cleaning up..."
kill $TUNNEL_PID 2>/dev/null || true
rm -f /tmp/test-ephemeral.log

echo "✓ Test complete"
echo ""
echo "To use ephemeral tunnel for real:"
echo "  export NGC_KEY='your-key'"
echo "  ./.tmp/start-ironclaw-ephemeral.sh"
