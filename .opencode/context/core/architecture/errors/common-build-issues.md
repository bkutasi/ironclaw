<!-- Context: architecture/errors | Priority: high | Version: 1.0 | Updated: 2026-03-19 -->
# Errors: Build Issues

**Purpose**: Common build and compilation errors for IronClaw

**Last Updated**: 2026-03-19

## Error: WASM Target Not Found

**Symptom**:
```
error: package `telegram v0.1.0` cannot be built because it requires rustc 1.70
or newer, while the currently active rustc version is 1.69.0
```

**Cause**: WASM target not installed or Rust version too old.

**Solution**:
```bash
# Update Rust
rustup update

# Add WASM target
rustup target add wasm32-unknown-unknown

# Verify
rustup target list --installed | grep wasm
```

**Prevention**: Run `rustup target add wasm32-unknown-unknown` once during setup

**Frequency**: common

---

## Error: Linker Not Found

**Symptom**:
```
error: linker `cc` not found
```

**Cause**: Build tools not installed (Linux) or Xcode command line tools missing (macOS).

**Solution**:
```bash
# Linux (Debian/Ubuntu)
sudo apt-get install build-essential

# Linux (Fedora/RHEL)
sudo dnf install gcc gcc-c++ make

# macOS
xcode-select --install
```

**Prevention**: Install build tools before first build

**Frequency**: common

---

## Error: Database Connection Failed

**Symptom**:
```
Error: Failed to connect to database: connection refused
```

**Cause**: PostgreSQL not running or wrong port.

**Solution**:
```bash
# Check if PostgreSQL running
docker ps | grep postgres

# Start Docker PostgreSQL
docker run -d --name ironclaw-postgres \
  -e POSTGRES_PASSWORD=pass \
  -p 5433:5432 postgres:16-alpine

# Or check system PostgreSQL
sudo systemctl status postgresql
```

**Prevention**: Start database before running IronClaw

**Frequency**: common

---

## Error: Port Already in Use

**Symptom**: `Error: Address already in use (os error 98)`

**Cause**: Previous instance still running.

**Solution**:
```bash
pkill ironclaw && sleep 2 && ./target/release/ironclaw
```

**Prevention**: Stop previous instance first

**Frequency**: common

---

## Related

- concepts/wasm-channels.md
- guides/build-process.md
