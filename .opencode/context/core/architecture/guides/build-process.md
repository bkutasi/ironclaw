<!-- Context: architecture/guides | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Guide: Build Process

**Purpose**: Build IronClaw Rust binary and WASM channels

**Last Updated**: 2026-03-19

## Prerequisites

- Rust toolchain (`rustup`)
- WASM target (`wasm32-unknown-unknown`)
- PostgreSQL (Docker or native)
- Node.js (for some tooling)

**Estimated time**: 5 min

## Steps

### 1. Install Rust Toolchain
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```
**Expected**: `rustc --version` shows version

### 2. Add WASM Target
```bash
rustup target add wasm32-unknown-unknown
```
**Expected**: Target added successfully

### 3. Build Main Binary
```bash
cargo build --release
```
**Expected**: `target/release/ironclaw` binary created

### 4. Build WASM Channels
```bash
cd channels/telegram
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/telegram.wasm ~/.ironclaw/channels/
```
**Expected**: `telegram.wasm` copied to channels directory

### 5. Verify Build
```bash
./target/release/ironclaw --version
ls -lh ~/.ironclaw/channels/telegram.wasm
```
**Expected**: Version output and WASM file exists

## Verification

```bash
# Check binary
file target/release/ironclaw
# Should show: ELF 64-bit executable

# Check WASM
file ~/.ironclaw/channels/telegram.wasm
# Should show: WebAssembly (wasm) binary module
```

## 📂 Codebase References

**Build Configuration**:
- `Cargo.toml` - Main project dependencies
- `channels/telegram/Cargo.toml` - WASM channel dependencies

**Build Scripts**:
- `scripts/build-all.sh` - Full build script
- `scripts/build-wasm.sh` - WASM-only build

**Source Code**:
- `src/main.rs` - Main entry point
- `channels/telegram/src/lib.rs` - WASM channel code

## Troubleshooting

| Issue | Solution |
|-------|----------|
| WASM target not found | `rustup target add wasm32-unknown-unknown` |
| Build fails with linker error | Install `build-essential` or `gcc` |
| WASM module too large | Build with `--release` flag |

## Related

- concepts/wasm-channels.md
- lookup/env-variables.md
- errors/common-build-issues.md
