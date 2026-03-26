#!/bin/bash
# Fix: Remove trailing newline from cloudflared URL file

URL_FILE="/tmp/cloudflared-ephemeral-url.txt"

if [ -f "$URL_FILE" ]; then
    # Read URL and strip all whitespace (including newlines)
    TUNNEL_URL=$(cat "$URL_FILE" | tr -d '[:space:]')
    echo -n "$TUNNEL_URL" > "$URL_FILE"
    echo "Fixed: Removed trailing newline from $URL_FILE"
    echo "URL is now: '$TUNNEL_URL'"
else
    echo "URL file not found: $URL_FILE"
fi
