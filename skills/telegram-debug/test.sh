#!/bin/bash
# Test script for telegram-debug skill
# Run this to verify all diagnostic tools work correctly

set -e

echo "==================================="
echo "Telegram Debug Skill - Test Suite"
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((TESTS_PASSED++))
    ((TESTS_RUN++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((TESTS_FAILED++))
    ((TESTS_RUN++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

info() {
    echo -e "ℹ️  INFO: $1"
}

# Test 1: Check skill files exist
echo "Test 1: Skill Files Exist"
echo "-------------------------"

if [ -f "SKILL.md" ]; then
    pass "SKILL.md exists"
else
    fail "SKILL.md not found"
fi

if [ -f "README.md" ]; then
    pass "README.md exists"
else
    fail "README.md not found"
fi

if [ -f "telegram-debug.capabilities.json" ]; then
    pass "telegram-debug.capabilities.json exists"
else
    fail "telegram-debug.capabilities.json not found"
fi

echo ""

# Test 2: Validate JSON syntax
echo "Test 2: JSON Syntax Validation"
echo "-------------------------------"

if command -v jq &> /dev/null; then
    if jq empty telegram-debug.capabilities.json 2>/dev/null; then
        pass "Capabilities JSON is valid"
    else
        fail "Capabilities JSON has syntax errors"
    fi
else
    warn "jq not installed, skipping JSON validation"
fi

echo ""

# Test 3: Check required tools
echo "Test 3: Required Tools Available"
echo "---------------------------------"

REQUIRED_TOOLS=("curl" "pgrep")

for tool in "${REQUIRED_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        pass "$tool is available"
    else
        fail "$tool is not installed"
    fi
done

# Optional tools
OPTIONAL_TOOLS=("jq" "dig" "nslookup" "psql")

for tool in "${OPTIONAL_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        pass "$tool is available (optional)"
    else
        warn "$tool not installed (optional)"
    fi
done

echo ""

# Test 4: Check documentation structure
echo "Test 4: Documentation Structure"
echo "--------------------------------"

# Check SKILL.md has required sections
REQUIRED_SECTIONS=(
    "telegram_check_webhook"
    "telegram_check_tunnel"
    "telegram_check_config"
    "telegram_fix_common_issues"
    "telegram_full_report"
)

for section in "${REQUIRED_SECTIONS[@]}"; do
    if grep -q "$section" SKILL.md; then
        pass "SKILL.md contains $section"
    else
        fail "SKILL.md missing $section section"
    fi
done

echo ""

# Test 5: Simulate webhook check (if token available)
echo "Test 5: Webhook Check Simulation"
echo "---------------------------------"

if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    info "TELEGRAM_BOT_TOKEN found, running actual check"
    
    RESPONSE=$(curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo")
    
    if echo "$RESPONSE" | jq -e '.ok' > /dev/null 2>&1; then
        pass "Telegram API accessible"
        
        WEBHOOK_URL=$(echo "$RESPONSE" | jq -r '.result.url')
        if [ "$WEBHOOK_URL" != "null" ] && [ -n "$WEBHOOK_URL" ]; then
            pass "Webhook URL is set: $WEBHOOK_URL"
        else
            warn "Webhook URL not set (this is OK for testing)"
        fi
    else
        fail "Telegram API error: $RESPONSE"
    fi
else
    warn "TELEGRAM_BOT_TOKEN not set, skipping live API test"
    info "To test: export TELEGRAM_BOT_TOKEN=your_token"
fi

echo ""

# Test 6: Simulate tunnel check
echo "Test 6: Tunnel Check Simulation"
echo "--------------------------------"

if pgrep -f cloudflared > /dev/null; then
    pass "cloudflared process is running"
    
    # Try to get tunnel URL from process args
    TUNNEL_URL=$(pgrep -af cloudflared | grep -oP 'https?://[^\s]+' | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        pass "Tunnel URL detected: $TUNNEL_URL"
    else
        warn "Could not extract tunnel URL from process args"
    fi
else
    warn "cloudflared not running (this is OK for testing)"
fi

echo ""

# Test 7: Check config file structure
echo "Test 7: Config File Structure"
echo "------------------------------"

if [ -f "$HOME/.ironclaw/.env" ]; then
    pass "~/.ironclaw/.env exists"
    
    # Check for required variables
    if grep -q "TELEGRAM_BOT_TOKEN" "$HOME/.ironclaw/.env"; then
        pass "TELEGRAM_BOT_TOKEN configured"
    else
        warn "TELEGRAM_BOT_TOKEN not configured"
    fi
    
    if grep -q "LLM_MODEL" "$HOME/.ironclaw/.env"; then
        LLM_MODEL=$(grep "LLM_MODEL" "$HOME/.ironclaw/.env" | cut -d'=' -f2 | tr -d '"')
        if [ "$LLM_MODEL" != "stepfun-ai/step-3.5-flash" ]; then
            pass "LLM_MODEL is valid: $LLM_MODEL"
        else
            fail "LLM_MODEL is invalid: $LLM_MODEL (returns 404)"
        fi
    else
        warn "LLM_MODEL not configured"
    fi
    
    if grep -q "DATABASE_URL" "$HOME/.ironclaw/.env"; then
        pass "DATABASE_URL configured"
    else
        warn "DATABASE_URL not configured"
    fi
else
    warn "~/.ironclaw/.env not found (this is OK for fresh install)"
fi

echo ""

# Test 8: Validate auto-fix logic
echo "Test 8: Auto-Fix Logic Validation"
echo "----------------------------------"

# Check that auto-fix section exists in SKILL.md
if grep -q "Auto-Fix" SKILL.md; then
    pass "Auto-fix documentation exists"
else
    fail "Auto-fix documentation missing"
fi

# Check for backup mechanism
if grep -q "backup" SKILL.md; then
    pass "Backup mechanism documented"
else
    warn "Backup mechanism not documented"
fi

echo ""

# Test 9: Check error handling documentation
echo "Test 9: Error Handling Documentation"
echo "-------------------------------------"

ERROR_DOCS=(
    "integrations/errors/tunnel-errors.md"
    "integrations/errors/bot-no-response.md"
    "architecture/errors/llm-model-mismatch.md"
)

for doc in "${ERROR_DOCS[@]}"; do
    if [ -f "../../$doc" ]; then
        pass "$doc exists"
    else
        # Try alternate path
        if [ -f "/media/nvme/projects/ironclaw/$doc" ]; then
            pass "$doc exists (full path)"
        else
            warn "$doc not found (optional)"
        fi
    fi
done

echo ""

# Test 10: Check mode selection guide
echo "Test 10: Mode Selection Guide"
echo "------------------------------"

if [ -f "../../telecom/guides/telegram-mode-selection.md" ]; then
    pass "telegram-mode-selection.md exists"
elif [ -f "/media/nvme/projects/ironclaw/telecom/guides/telegram-mode-selection.md" ]; then
    pass "telegram-mode-selection.md exists (full path)"
else
    warn "telegram-mode-selection.md not found (optional)"
fi

echo ""

# Summary
echo "==================================="
echo "Test Summary"
echo "==================================="
echo "Tests Run:    $TESTS_RUN"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    echo ""
    echo "Note: Some failures may be OK depending on your setup."
    echo "Critical failures: Missing SKILL.md, invalid JSON, missing tools"
    echo "Non-critical: Missing optional tools, config files not set up yet"
    exit 1
fi
