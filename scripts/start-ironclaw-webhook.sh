#!/bin/bash
# =============================================================================
# IronClaw Telegram Webhook Mode Starter
# =============================================================================
# 
# USAGE:
#   export NGC_KEY="your-nvidia-api-key"
#   export TELEGRAM_BOT_TOKEN="your-bot-token"
#   ./.tmp/start-ironclaw-webhook.sh
#
# MODE:
#   Webhook Mode: Telegram pushes updates to your server instantly
#   - Instant message delivery (no 30s delay)
#   - Fewer API calls (only on messages)
#   - Cleaner logs (no constant polling)
#
# REQUIRED ENVIRONMENT VARIABLES:
#   NGC_KEY              - NVIDIA NGC API key for LLM backend
#   TELEGRAM_BOT_TOKEN   - Telegram bot token from @BotFather
#   TUNNEL_URL           - Your public URL (default: https://ironclaw.kutasi.dev)
#
# OPTIONAL ENVIRONMENT VARIABLES:
#   LLM_MODEL            - Model to use (default: z-ai/glm5)
#   GATEWAY_PORT         - Web UI port (default: 3004)
#   HTTP_PORT            - Webhook server port (default: 8081)
#   HTTP_WEBHOOK_SECRET  - Secret for webhook validation (auto-generated if not set)
#
# CLOUDFLARE SETUP (IMPORTANT!):
#   1. Cloudflare Dashboard → Security → Bots
#   2. Add exception: URI Path contains "/webhook/telegram" → Skip Bot Fight Mode
#   3. OR temporarily disable Bot Fight Mode for testing
#
# TROUBLESHOOTING:
#   1. "403 Forbidden" → Check Cloudflare Bot Fight Mode exception
#   2. "Webhook not working" → Verify tunnel is connected
#   3. "Messages delayed" → Check Cloudflare tunnel connection
#   4. "429 Too Many Requests" → Wait 60 seconds, script handles this automatically
#
# EXPECTED LOG MESSAGES:
#   - "Webhook server listening on 0.0.0.0:8081" - Server started
#   - "Webhook registered successfully" - Telegram configured (by IronClaw)
#   - "Webhook request received" - Message received (on incoming messages)
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Log file for error tracking
LOG_FILE="/tmp/ironclaw.log"

# =============================================================================
# Cleanup Handler
# =============================================================================
cleanup() {
    echo ""
    echo -e "${YELLOW}=== Shutting down IronClaw ===${NC}"
    echo "IronClaw will be stopped. Webhook registration persists for quick restart."
    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# =============================================================================
# Helper Functions
# =============================================================================
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}❌${NC} $1"
}

log_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# =============================================================================
# Rate Limit Detection and Recovery
# =============================================================================
check_rate_limit() {
    log_step "Checking for Previous Rate Limit Errors"
    
    if [ -f "$LOG_FILE" ]; then
        if grep -q "429" "$LOG_FILE" 2>/dev/null; then
            log_warn "Detected previous 429 (Too Many Requests) error in logs"
            log_info "Waiting 60 seconds to avoid rate limiting..."
            echo ""
            
            # Show countdown
            for i in {60..1}; do
                printf "\r  Waiting: %3d seconds remaining..." $i
                sleep 1
            done
            echo ""
            log_info "Rate limit cooldown complete. Proceeding..."
        else
            log_info "No previous rate limit errors detected"
        fi
    else
        log_info "No previous log file found (first run or log cleared)"
    fi
    echo ""
}

# =============================================================================
# Environment Validation
# =============================================================================
validate_environment() {
    log_step "Validating Environment"
    
    VALIDATION_FAILED=false
    
    if [ -z "$NGC_KEY" ]; then
        log_error "NGC_KEY is not set"
        echo "   Fix: export NGC_KEY=\"your-nvidia-api-key\""
        echo "   Get your key from: https://org.ngc.nvidia.com/setup/personal-keys"
        VALIDATION_FAILED=true
    else
        log_info "NGC_KEY is configured"
    fi
    
    if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
        log_error "TELEGRAM_BOT_TOKEN is not set"
        echo "   Fix: export TELEGRAM_BOT_TOKEN=\"your-bot-token\""
        echo "   Get your token from: @BotFather on Telegram"
        VALIDATION_FAILED=true
    else
        log_info "TELEGRAM_BOT_TOKEN is configured"
    fi
    
    if [ "$VALIDATION_FAILED" = true ]; then
        echo ""
        log_error "Environment validation failed"
        exit 1
    fi
    echo ""
}

# =============================================================================
# Auto-generate Webhook Secret
# =============================================================================
setup_webhook_secret() {
    log_step "Configuring Webhook Security"
    
    if [ -z "$HTTP_WEBHOOK_SECRET" ]; then
        log_warn "HTTP_WEBHOOK_SECRET not set - auto-generating secure secret"
        export HTTP_WEBHOOK_SECRET=$(openssl rand -hex 32)
        log_info "Generated webhook secret (64-character hex string)"
        echo "   This secret will be used by IronClaw to validate incoming webhook requests"
        echo "   To persist across restarts, add to your environment:"
        echo "   export HTTP_WEBHOOK_SECRET=\"$HTTP_WEBHOOK_SECRET\""
    else
        log_info "HTTP_WEBHOOK_SECRET is configured (secret length: ${#HTTP_WEBHOOK_SECRET} chars)"
    fi
    echo ""
}

# =============================================================================
# Set Defaults
# =============================================================================
set_defaults() {
    export TUNNEL_URL="${TUNNEL_URL:-https://ironclaw.kutasi.dev}"
    export GATEWAY_PORT="${GATEWAY_PORT:-3004}"
    export HTTP_PORT="${HTTP_PORT:-8081}"
    export LLM_MODEL="${LLM_MODEL:-z-ai/glm5}"
    export LLM_BACKEND="${LLM_BACKEND:-openai_compatible}"
    export LLM_BASE_URL="${LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}"
}

# =============================================================================
# Binary and Database Validation
# =============================================================================
validate_binary_and_db() {
    log_step "Validating Binary and Database"
    
    # Check if IronClaw binary exists
    if [ ! -f "./target/release/ironclaw" ]; then
        log_error "IronClaw binary not found at ./target/release/ironclaw"
        echo "   Fix: Build the project with: cargo build --release"
        exit 1
    fi
    log_info "IronClaw binary found"
    
    # Check database connection
    DB_URL="postgres://postgres:yourpass@localhost:5433/ironclaw"
    if ! psql "$DB_URL" -c "SELECT 1" > /dev/null 2>&1; then
        log_error "Cannot connect to PostgreSQL database"
        echo "   Fix: Ensure PostgreSQL is running and database 'ironclaw' exists"
        echo "   Start PostgreSQL: sudo systemctl start postgresql"
        echo "   Create database: createdb ironclaw"
        exit 1
    fi
    log_info "Database connection verified"
    echo ""
}

# =============================================================================
# Cloudflare Tunnel Verification
# =============================================================================
verify_tunnel() {
    log_step "Verifying Cloudflare Tunnel Accessibility"
    
    # Check if tunnel process is running
    if pgrep -f "cloudflared.*run" > /dev/null; then
        log_info "Cloudflare tunnel process is running"
    else
        log_warn "Cloudflare tunnel process not detected"
        echo "   Attempting to start tunnel..."
        
        CONFIG_FILE="/home/bkutasi/.cloudflared/ironclaw-config.yml"
        if [ -f "$CONFIG_FILE" ]; then
            nohup cloudflared tunnel --config "$CONFIG_FILE" run > /tmp/cloudflared-webhook.log 2>&1 &
            sleep 3
            if pgrep -f "cloudflared.*run" > /dev/null; then
                log_info "Tunnel started successfully"
            else
                log_warn "Tunnel start attempted - verify manually"
            fi
        else
            log_warn "Tunnel config not found at $CONFIG_FILE"
            echo "   Assuming tunnel is managed externally or manually"
        fi
    fi
    
    # Test tunnel accessibility - CRITICAL: Test webhook endpoint specifically
    echo ""
    log_info "Testing tunnel endpoint accessibility..."
    echo "   Testing: POST $TUNNEL_URL/webhook/telegram"
    
    # Test the actual webhook endpoint
    TUNNEL_TEST_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$TUNNEL_URL/webhook/telegram" -d '{}' -m 10 2>&1 || echo "000")
    
    case $TUNNEL_TEST_RESPONSE in
        200|201|202|204)
            log_info "Webhook endpoint is accessible (HTTP $TUNNEL_TEST_RESPONSE)"
            ;;
        404)
            log_warn "Webhook endpoint returned 404 (expected - IronClaw not running yet)"
            echo "   This is normal - endpoint exists but IronClaw will register on startup"
            ;;
        403)
            log_error "Webhook endpoint blocked (HTTP 403)"
            echo ""
            echo "   FIX: Cloudflare Bot Fight Mode is blocking webhook requests"
            echo "   1. Go to Cloudflare Dashboard → Security → Bots"
            echo "   2. Add exception: URI Path contains \"/webhook/telegram\""
            echo "   3. Action: Skip Bot Fight Mode"
            echo ""
            echo "   See: .tmp/CLOUDFLARE_FIX_STEPS.md for detailed instructions"
            exit 1
            ;;
        502|503|504)
            log_error "Webhook endpoint unreachable (HTTP $TUNNEL_TEST_RESPONSE)"
            echo ""
            echo "   FIX: Cloudflare tunnel is not properly connected"
            echo "   1. Check tunnel status: pgrep -f cloudflared"
            echo "   2. Restart tunnel if needed"
            echo "   3. Verify tunnel config: cat $CONFIG_FILE"
            exit 1
            ;;
        530)
            log_error "Origin DNS error (HTTP 530)"
            echo ""
            echo "   FIX: Cloudflare cannot resolve your origin server"
            echo "   1. Check tunnel is connected to correct hostname"
            echo "   2. Verify DNS records in Cloudflare dashboard"
            exit 1
            ;;
        000)
            log_error "Cannot connect to tunnel URL"
            echo ""
            echo "   FIX: Network connectivity issue or tunnel not running"
            echo "   1. Check internet connection"
            echo "   2. Verify tunnel is running: pgrep -f cloudflared"
            echo "   3. Test URL manually: curl -I $TUNNEL_URL"
            exit 1
            ;;
        *)
            log_warn "Unexpected response code: HTTP $TUNNEL_TEST_RESPONSE"
            echo "   Proceeding with caution - monitor logs for issues"
            ;;
    esac
    echo ""
}

# =============================================================================
# Environment Summary
# =============================================================================
show_configuration() {
    log_step "Configuration Summary"
    
    echo "  ${CYAN}Mode:${NC}           Webhook (instant delivery)"
    echo "  ${CYAN}Tunnel URL:${NC}     $TUNNEL_URL"
    echo "  ${CYAN}Gateway Port:${NC}   $GATEWAY_PORT (Web UI)"
    echo "  ${CYAN}HTTP Port:${NC}      $HTTP_PORT (Webhook server)"
    echo "  ${CYAN}LLM Model:${NC}      $LLM_MODEL"
    echo "  ${CYAN}LLM Backend:${NC}    $LLM_BACKEND"
    echo "  ${CYAN}Webhook Secret:${NC} ✓ Configured (auto-generated or provided)"
    echo ""
    
    log_info "Webhook registration will be handled by IronClaw on startup"
    echo "   The script verifies tunnel accessibility but does NOT register the webhook"
    echo "   This prevents duplicate registration and HTTP 429 rate limit errors"
    echo ""
}

# =============================================================================
# Cloudflare Bot Fight Mode Warning
# =============================================================================
show_cloudflare_warning() {
    log_step "Cloudflare Bot Fight Mode"
    
    echo -e "${YELLOW}IMPORTANT:${NC} If you encounter 403 Forbidden errors:"
    echo ""
    echo "   1. Go to Cloudflare Dashboard → Security → Bots"
    echo "   2. Click 'Create Rule' or add exception"
    echo "   3. Configure:"
    echo "      - Field: URI Path"
    echo "      - Operator: contains"
    echo "      - Value: /webhook/telegram"
    echo "      - Action: Skip Bot Fight Mode"
    echo ""
    echo "   See: .tmp/CLOUDFLARE_FIX_STEPS.md for detailed instructions"
    echo ""
}

# =============================================================================
# Start IronClaw
# =============================================================================
start_ironclaw() {
    log_step "Starting IronClaw"
    
    echo "Log file: $LOG_FILE"
    echo ""
    log_info "Ready to receive Telegram messages"
    echo ""
    echo -e "${BLUE}=== Expected Startup Sequence ===${NC}"
    echo "  1. IronClaw initializes database connections"
    echo "  2. Webhook server starts on 0.0.0.0:$HTTP_PORT"
    echo "  3. IronClaw registers webhook with Telegram automatically"
    echo "  4. Ready to receive messages"
    echo ""
    echo -e "${CYAN}=== What to Monitor ===${NC}"
    echo "  - 'Webhook server listening on 0.0.0.0:$HTTP_PORT'"
    echo "  - 'Webhook registered successfully'"
    echo "  - 'Webhook request received' (on incoming messages)"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Change to project directory and start IronClaw
    cd /media/nvme/projects/ironclaw
    ./target/release/ironclaw 2>&1 | tee "$LOG_FILE"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo ""
    log_step "IronClaw Telegram Webhook Mode Starter"
    echo ""
    
    # Set defaults first
    set_defaults
    
    # Check for rate limit from previous run
    check_rate_limit
    
    # Validate environment
    validate_environment
    
    # Auto-generate webhook secret if not set
    setup_webhook_secret
    
    # Validate binary and database
    validate_binary_and_db
    
    # Verify tunnel accessibility
    verify_tunnel
    
    # Show configuration
    show_configuration
    
    # Show Cloudflare warning
    show_cloudflare_warning
    
    # Start IronClaw
    start_ironclaw
}

# Run main function
main
