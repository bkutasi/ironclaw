#!/bin/bash
# Fix Telegram webhook secret configuration
# This script saves the webhook secret to the encrypted secrets store

set -e

WEBHOOK_SECRET="f728634e611a3bc25df6c8790bd80b15b5b9d381a1740aa492a02031b9ee3bcc"

echo "=== Fixing Telegram Webhook Configuration ==="
echo ""

# Check if ironclaw CLI is available
if ! command -v ironclaw &> /dev/null; then
    echo "❌ ironclaw CLI not found in PATH"
    echo "Please run this from the project directory or add target/release to PATH"
    exit 1
fi

echo "✓ ironclaw CLI found"
echo ""

# Save the webhook secret
echo "Saving webhook secret to encrypted store..."
ironclaw secret save telegram_webhook_secret "$WEBHOOK_SECRET"

if [ $? -eq 0 ]; then
    echo "✓ Webhook secret saved successfully"
else
    echo "❌ Failed to save webhook secret"
    exit 1
fi

echo ""
echo "=== Next Steps ==="
echo "1. Restart ironclaw: ./target/release/ironclaw"
echo "2. Watch for: 'has_webhook_secret=true' in startup logs"
echo "3. Test webhook: curl -X POST https://ironclaw.kutasi.dev/webhook/telegram"
echo ""
echo "To verify the secret was saved:"
echo "  ironclaw secret list"
