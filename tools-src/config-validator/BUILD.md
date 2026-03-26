# Building the Config Validator Tool

## Prerequisites

```bash
# Ensure you have Rust installed
rustup update

# Add WASM target
rustup target add wasm32-unknown-unknown
```

## Build Commands

```bash
cd tools-src/config-validator

# Build release version
cargo build --release --target wasm32-unknown-unknown

# The WASM module will be at:
# target/wasm32-unknown-unknown/release/config_validator_tool.wasm
```

## Installation

```bash
# Copy to IronClaw tools directory
cp target/wasm32-unknown-unknown/release/config_validator_tool.wasm \
   ~/.ironclaw/tools/

# Or register via CLI
ironclaw tool install \
  target/wasm32-unknown-unknown/release/config_validator_tool.wasm \
  --name config_validator
```

## Troubleshooting Build Issues

### Error: package requires rustc 1.70 or newer

```bash
rustup update
```

### Error: target not found

```bash
rustup target add wasm32-unknown-unknown
```

### Error: mio/uuid compilation failures

This can happen if dependencies have updated. Try:

```bash
# Clean build
cargo clean
cargo build --release --target wasm32-unknown-unknown

# Or update Cargo.lock
rm Cargo.lock
cargo build --release --target wasm32-unknown-unknown
```

## Testing

```bash
# Run unit tests
cargo test

# Test with sample input
echo '{"env_vars": {"NGC_KEY": "nvapi-test", "LLM_MODEL": "z-ai/glm5"}, "check": "env"}' | \
  cargo run --release
```

## Usage Example

Once installed, use from IronClaw:

```json
{
  "tool": "config_validator",
  "params": {
    "env_vars": {
      "NGC_KEY": "nvapi-your-key",
      "LLM_MODEL": "z-ai/glm5",
      "DATABASE_URL": "postgres://postgres:yourpass@localhost:5433/ironclaw"
    },
    "check": "full_report"
  }
}
```

## Verification

Check the built WASM module:

```bash
file target/wasm32-unknown-unknown/release/config_validator_tool.wasm
# Should output: WebAssembly (wasm) binary module

wasm2wat target/wasm32-unknown-unknown/release/config_validator_tool.wasm | head -20
# Should show WASM module structure
```
