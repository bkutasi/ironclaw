#!/bin/bash
# =============================================================================
# IronClaw Ephemeral Cloudflare Tunnel Starter
# =============================================================================
# 
# USAGE:
#   export NGC_KEY="your-nvidia-api-key"
#   export TELEGRAM_BOT_TOKEN="your-bot-token"
#   ./.tmp/start-ironclaw-ephemeral.sh
#
# MODE: Ephemeral Tunnel Mode - Temporary public URL for webhook testing
#
# REQUIRED: NGC_KEY, TELEGRAM_BOT_TOKEN
# OPTIONAL: LLM_MODEL, GATEWAY_PORT, HTTP_PORT, HTTP_WEBHOOK_SECRET
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

LOG_FILE="/tmp/ironclaw.log"
CLOUDFLARED_LOG="/tmp/cloudflared-ephemeral.log"

cleanup() {
    echo ""
    echo -e "${YELLOW}=== Shutting down Ephemeral Tunnel ===${NC}"
    if [ -f "$CLOUDFLARED_PID_FILE" ]; then
        CLOUDFLARED_PID=$(cat "$CLOUDFLARED_PID_FILE" 2>/dev/null || true)
        if [ -n "$CLOUDFLARED_PID" ] && kill -0 "$CLOUDFLARED_PID" 2>/dev/null; then
            echo "Stopping cloudflared (PID: $CLOUDFLARED_PID)..."
            kill "$CLOUDFLARED_PID" 2>/dev/null || true
            sleep 1
            kill -9 "$CLOUDFLARED_PID" 2>/dev/null || true
        fi
        rm -f "$CLOUDFLARED_PID_FILE"
    fi
    pkill -f "cloudflared.*localhost:${HTTP_PORT}" 2>/dev/null || true
    rm -f "$TUNNEL_URL_FILE"
    rm -f "$CLOUDFLARED_LOG"
    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠️${NC} $1"; }
log_error() { echo -e "${RED}❌${NC} $1"; }
log_step() { echo -e "${BLUE}=== $1 ===${NC}"; }

check_rate_limit() {
    log_step "Checking for Previous Rate Limit Errors"
    if [ -f "$LOG_FILE" ] && grep -q "429" "$LOG_FILE" 2>/dev/null; then
        log_warn "Detected previous 429 (Too Many Requests) error in logs"
        log_info "Waiting 60 seconds to avoid rate limiting..."
        echo ""
        for i in {60..1}; do
            printf "\r  Waiting: %3d seconds remaining..." $i
            sleep 1
        done
        echo ""
        log_info "Rate limit cooldown complete. Proceeding..."
    else
        log_info "No previous rate limit errors detected"
    fi
    echo ""
}

validate_environment() {
    log_step "Validating Environment"
    VALIDATION_FAILED=false
    
    if [ -z "$NGC_KEY" ]; then
        log_error "NGC_KEY is not set"
        echo "   FIX: export NGC_KEY=\"your-nvidia-api-key\""
        VALIDATION_FAILED=true
    else
        log_info "NGC_KEY is configured"
    fi
    
    if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
        log_error "TELEGRAM_BOT_TOKEN is not set"
        echo "   FIX: export TELEGRAM_BOT_TOKEN=\"your-bot-token\""
        VALIDATION_FAILED=true
    else
        log_info "TELEGRAM_BOT_TOKEN is configured"
    fi
    
    if [ "$VALIDATION_FAILED" = true ]; then
        log_error "Environment validation failed"
        exit 1
    fi
    echo ""
}

setup_webhook_secret() {
    log_step "Configuring Webhook Security"
    if [ -z "$HTTP_WEBHOOK_SECRET" ]; then
        log_warn "HTTP_WEBHOOK_SECRET not set - auto-generating secure secret"
        export HTTP_WEBHOOK_SECRET=$(openssl rand -hex 32)
        log_info "Generated webhook secret (64-character hex string)"
        echo "   To persist across restarts, add to your environment:"
        echo "   export HTTP_WEBHOOK_SECRET=\"$HTTP_WEBHOOK_SECRET\""
    else
        log_info "HTTP_WEBHOOK_SECRET is configured"
    fi
    echo ""
}

set_defaults() {
    export GATEWAY_PORT="${GATEWAY_PORT:-3004}"
    export HTTP_PORT="${HTTP_PORT:-8081}"
    export LLM_MODEL="${LLM_MODEL:-z-ai/glm5}"
    export LLM_BACKEND="${LLM_BACKEND:-openai_compatible}"
    export LLM_BASE_URL="${LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}"
    export CLOUDFLARED_PID_FILE="/tmp/cloudflared-ephemeral-${HTTP_PORT}.pid"
    export TUNNEL_URL_FILE="/tmp/cloudflared-ephemeral-url.txt"
}

check_cloudflared() {
    log_step "Checking Cloudflared Installation"
    if ! command -v cloudflared &> /dev/null; then
        log_error "cloudflared binary not found"
        echo "   FIX: Install from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        exit 1
    fi
    log_info "cloudflared found: $(cloudflared --version 2>&1 | head -1)"
    echo ""
}

validate_database() {
    log_step "Validating Database Connection"
    
    if [ ! -f "./target/release/ironclaw" ]; then
        log_error "IronClaw binary not found at ./target/release/ironclaw"
        echo "   FIX: cargo build --release"
        exit 1
    fi
    log_info "IronClaw binary found"
    
    # Try Docker PostgreSQL first, then direct
    DB_URL="postgres://postgres:yourpass@localhost:5433/ironclaw"
    if ! psql "$DB_URL" -c "SELECT 1" > /dev/null 2>&1; then
        DB_URL="postgres://postgres:yourpass@localhost:5432/ironclaw"
        if ! psql "$DB_URL" -c "SELECT 1" > /dev/null 2>&1; then
            log_error "Cannot connect to PostgreSQL database"
            echo "   FIX: docker start ironclaw-db || sudo systemctl start postgresql"
            exit 1
        fi
    fi
    log_info "Database connection verified ($DB_URL)"
    echo ""
}

start_ephemeral_tunnel() {
    log_step "Starting Ephemeral Cloudflare Tunnel"
    
    if pkill -f "cloudflared.*localhost:${HTTP_PORT}" 2>/dev/null; then
        log_info "Killed existing cloudflared processes"
        sleep 1
    fi
    
    rm -f "$CLOUDFLARED_LOG"
    cloudflared tunnel --url "http://localhost:${HTTP_PORT}" > "$CLOUDFLARED_LOG" 2>&1 &
    CLOUDFLARED_PID=$!
    echo "$CLOUDFLARED_PID" > "$CLOUDFLARED_PID_FILE"
    log_info "Cloudflared started (PID: $CLOUDFLARED_PID)"
    
    log_info "Waiting for tunnel connection..."
    TUNNEL_URL=""
    MAX_RETRIES=30
    
    for ((i=1; i<=MAX_RETRIES; i++)); do
        if ! kill -0 "$CLOUDFLARED_PID" 2>/dev/null; then
            log_error "Cloudflared died unexpectedly"
            cat "$CLOUDFLARED_LOG"
            exit 1
        fi
        
        if [ -f "$CLOUDFLARED_LOG" ]; then
            TUNNEL_URL=$(grep -oP 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$CLOUDFLARED_LOG" 2>/dev/null | head -1 || true)
        fi
        
        if [ -n "$TUNNEL_URL" ]; then
            log_info "Tunnel connected: $TUNNEL_URL"
            echo -n "$TUNNEL_URL" > "$TUNNEL_URL_FILE"
            export TUNNEL_URL="$TUNNEL_URL"
            break
        fi
        
        [ $((i % 5)) -eq 0 ] && echo "   Waiting for tunnel... ($i/$MAX_RETRIES)"
        sleep 1
    done
    
    if [ -z "$TUNNEL_URL" ]; then
        log_error "Failed to establish tunnel after $MAX_RETRIES seconds"
        cat "$CLOUDFLARED_LOG"
        exit 1
    fi
    echo ""
}

wait_for_dns() {
    log_step "Waiting for DNS Propagation"
    log_info "Waiting for trycloudflare.com DNS to propagate (30 seconds)..."
    echo "   This is required for Telegram to resolve the tunnel URL"
    echo ""
    
    for i in {30..1}; do
        printf "\r  DNS propagation: %3d seconds remaining..." $i
        sleep 1
    done
    echo ""
    log_info "DNS propagation wait complete"
    echo ""
}

verify_tunnel() {
    log_step "Verifying Tunnel Endpoint"
    echo "   Testing: POST $TUNNEL_URL/webhook/telegram"
    
    TUNNEL_TEST_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$TUNNEL_URL/webhook/telegram" -d '{}' -m 10 2>&1 || echo "000")
    
    case $TUNNEL_TEST_RESPONSE in
        200|201|202|204|404)
            log_info "Tunnel endpoint accessible (HTTP $TUNNEL_TEST_RESPONSE)"
            ;;
        000|530|502|503|504)
            log_warn "Tunnel endpoint not ready yet (HTTP $TUNNEL_TEST_RESPONSE)"
            echo "   This is normal - DNS may still be propagating"
            echo "   IronClaw will retry webhook registration automatically"
            ;;
        *)
            log_warn "Unexpected response: HTTP $TUNNEL_TEST_RESPONSE"
            ;;
    esac
    echo ""
}

show_configuration() {
    log_step "Configuration Summary"
    echo "  ${CYAN}Mode:${NC}           Ephemeral Tunnel (temporary URL)"
    echo "  ${CYAN}Tunnel URL:${NC}     $TUNNEL_URL"
    echo "  ${CYAN}Gateway Port:${NC}   $GATEWAY_PORT (Web UI)"
    echo "  ${CYAN}HTTP Port:${NC}      $HTTP_PORT (Webhook server)"
    echo "  ${CYAN}LLM Model:${NC}      $LLM_MODEL"
    echo "  ${CYAN}Webhook Secret:${NC} ✓ Auto-generated"
    echo ""
    log_info "IronClaw will register webhook on startup"
    echo "   If DNS hasn't propagated, it will retry automatically"
    echo ""
}

start_ironclaw() {
    log_step "Starting IronClaw"
    echo "Log file: $LOG_FILE"
    echo ""
    log_info "Ready to receive Telegram messages"
    echo ""
    echo -e "${BLUE}=== Expected Startup Sequence ===${NC}"
    echo "  1. IronClaw initializes database"
    echo "  2. Webhook server starts on 0.0.0.0:$HTTP_PORT"
    echo "  3. IronClaw registers webhook with Telegram"
    echo "  4. Ready to receive messages"
    echo ""
    echo -e "${CYAN}=== What to Monitor ===${NC}"
    echo "  - 'Webhook server listening on 0.0.0.0:$HTTP_PORT'"
    echo "  - 'Webhook registered successfully'"
    echo "  - 'Webhook request received' (on incoming messages)"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    cd /media/nvme/projects/ironclaw
    ./target/release/ironclaw 2>&1 | tee "$LOG_FILE"
}

main() {
    echo ""
    log_step "IronClaw Ephemeral Tunnel Mode Starter"
    echo ""
    
    set_defaults
    check_rate_limit
    validate_environment
    check_cloudflared
    setup_webhook_secret
    validate_database
    start_ephemeral_tunnel
    wait_for_dns  # CRITICAL: Wait for DNS propagation
    verify_tunnel
    show_configuration
    start_ironclaw
}

main
