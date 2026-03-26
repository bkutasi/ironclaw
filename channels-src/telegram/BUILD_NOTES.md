# Telegram Channel WASM Build Notes

## Issue (March 2026)

Building the Telegram channel WASM component with `wasm32-wasip2` target fails due to dependency incompatibilities:

```
error[E0658]: use of unstable library feature `wasip2`
  --> io-lifetimes-2.0.4/src/lib.rs:52:29
   |
52 | pub use std::os::wasi::io::{AsFd, BorrowedFd, OwnedFd};
```

### Root Cause

The `io-lifetimes` crate v2.0.4 (pulled in by `wasmtime-wasi` v28.0.1) uses unstable `wasip2` features that were not yet stabilized in Rust 1.92.0.

### Workaround

A pre-built WASM component is available in the source directory:
- `telegram.wasm` (367KB, component model v0x1000d)

This was built successfully on Mar 18 2026 and can be copied directly to `~/.ironclaw/channels/`.

## Build Requirements (for future fixes)

To build from source, you need:

1. **Rust** with WASI target:
   ```bash
   rustup target add wasm32-wasip2
   ```

2. **wasm-tools** for component creation:
   ```bash
   cargo install wasm-tools
   ```

3. **Clang** or **WASI SDK** for C dependencies:
   ```bash
   # Option A: Install clang
   apt-get install clang
   
   # Option B: Install WASI SDK
   # Download from https://github.com/WebAssembly/wasi-sdk/releases
   export WASI_SDK_PATH=/opt/wasi-sdk
   ```

4. **Build command**:
   ```bash
   cargo build --release --target wasm32-wasip2
   wasm-tools component new target/wasm32-wasip2/release/telegram_channel.wasm -o telegram.wasm
   wasm-tools strip telegram.wasm -o telegram.wasm
   ```

## Installation

```bash
mkdir -p ~/.ironclaw/channels
cp telegram.wasm telegram.capabilities.json ~/.ironclaw/channels/
```

## Validation

```bash
wasm-tools validate ~/.ironclaw/channels/telegram.wasm
# Should output nothing if valid
```
