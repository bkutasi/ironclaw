#!/bin/bash
echo "Telegram Debug Skill - Verification"
echo "===================================="
echo ""

# Check files
echo "Files:"
for file in SKILL.md README.md telegram-debug.capabilities.json; do
  if [ -f "$file" ]; then
    echo "  ✅ $file ($(wc -l < "$file") lines)"
  else
    echo "  ❌ $file MISSING"
  fi
done

echo ""
echo "Documentation files:"
for doc in ../../integrations/errors/tunnel-errors.md ../../integrations/errors/bot-no-response.md ../../architecture/errors/llm-model-mismatch.md ../../telecom/guides/telegram-mode-selection.md; do
  if [ -f "$doc" ]; then
    echo "  ✅ $doc"
  else
    echo "  ⚠️  $doc (optional)"
  fi
done

echo ""
echo "JSON Validation:"
if jq empty telegram-debug.capabilities.json 2>/dev/null; then
  echo "  ✅ telegram-debug.capabilities.json is valid JSON"
else
  echo "  ❌ telegram-debug.capabilities.json is INVALID"
fi

echo ""
echo "Tool definitions in capabilities:"
jq -r '.tools[].name' telegram-debug.capabilities.json | while read tool; do
  echo "  - $tool"
done

echo ""
echo "✅ Skill verification complete!"
