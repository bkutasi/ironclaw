<!-- Context: architecture/concepts | Priority: critical | Version: 1.0 | Updated: 2026-03-19 -->
# Concept: WASM Channels

**Purpose**: IronClaw's plugin architecture using WebAssembly for communication channels

**Last Updated**: 2026-03-19

## Core Idea

IronClaw uses WASM modules as isolated, sandboxed communication channels (Telegram, HTTP, etc.). Each channel is a compiled Rust WASM file that handles protocol-specific logic, loaded dynamically at runtime.

## Key Points

- **Isolation**: Each channel runs in sandboxed WASM environment
- **Dynamic loading**: Channels loaded from `~/.ironclaw/channels/` at runtime
- **Host functions**: WASM modules call Rust host functions for HTTP, logging, storage
- **Type safety**: Rust ensures type-safe communication between host and WASM
- **Hot reload**: Rebuild WASM → restart IronClaw → new channel version loaded

## Quick Example

```bash
# Channel location
~/.ironclaw/channels/
├── telegram.wasm    # Telegram bot channel
├── http.wasm        # HTTP webhook channel
└── discord.wasm     # Discord channel (future)

# Rebuild WASM channel
cd channels/telegram && cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/telegram.wasm ~/.ironclaw/channels/
```

## 📂 Codebase References

**WASM Runtime**:
- `src/wasm/runtime.rs` - WASM runtime initialization
- `src/wasm/host_functions.rs` - Host functions exposed to WASM

**Channel Interface**:
- `src/channels/mod.rs` - Channel trait and management
- `src/channels/telegram.rs` - Telegram channel wrapper

**WASM Modules**:
- `channels/telegram/src/lib.rs` - Telegram WASM implementation
- `channels/telegram/Cargo.toml` - WASM build configuration

**Build Scripts**:
- `scripts/build-wasm.sh` - WASM compilation script

## Deep Dive

**Reference**: See `guides/build-process.md` for WASM build workflow

## Related

- concepts/docker-postgres.md
- guides/build-process.md
- errors/common-build-issues.md
