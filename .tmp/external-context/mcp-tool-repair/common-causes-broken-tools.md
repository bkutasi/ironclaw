---
source: Tetrate.io & Stainless.com & Microsoft Learn
library: MCP (Model Context Protocol)
topic: Common causes of broken tools in MCP implementations
fetched: 2026-03-26
official_docs: https://modelcontextprotocol.io
---

# Common Causes of Broken Tools in MCP Implementations

## Five Main Categories of MCP Failures

Based on analysis of hundreds of MCP issues, failures fall into these categories:

1. **Configuration & Connection Issues** (35%)
2. **Schema & Registration Errors** (25%)
3. **Handler & Execution Failures** (20%)
4. **Authentication & Environment Problems** (15%)
5. **Timeout & Performance Issues** (5%)

---

## Category 1: Configuration & Connection Issues

### 1.1 Incorrect Command Paths

**Problem:**
```json
// WRONG - tilde doesn't expand in most GUI apps
{
  "command": "node",
  "args": ["~/projects/my-server/index.js"]
}

// WRONG - relative paths fail when app started from different directory
{
  "command": "python",
  "args": ["./server.py"]
}
```

**Fix:**
```json
// CORRECT - use absolute paths
{
  "command": "node",
  "args": ["/Users/username/projects/my-server/index.js"]
}
```

**Verification:**
```bash
# Test exact command from config
cd /exact/working/directory
node /absolute/path/to/server/index.js

# Should start without errors
```

### 1.2 Windows cmd.exe Execution Policy

**Problem:**
Windows blocks `npx` execution in subprocess contexts.

**Symptoms:**
- Server shows "Not connected"
- Works in terminal but not in VS Code/Cursor
- No error message shown

**Fix:**
```json
// BEFORE (broken on Windows)
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"]
}

// AFTER (fixed)
{
  "command": "cmd",
  "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-github"]
}
```

### 1.3 Missing Dependencies

**Problem:**
Server crashes immediately on startup.

**Symptoms:**
- `Error: Cannot find module '...'`
- `ModuleNotFoundError: No module named '...'`

**Fix:**
```bash
# For Node.js servers
cd /path/to/server
npm install

# For Python servers
cd /path/to/server
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1.4 Version Incompatibility

**Problem:**
Server uses features not available in installed runtime.

**Symptoms:**
```
SyntaxError: Unexpected token '?'  // Node < 14
ImportError: cannot import name 'TypeAlias'  // Python < 3.10
```

**Fix:**
```bash
# Check versions
node --version  # Need 18+ for MCP SDK
python --version  # Need 3.10+

# Use specific version in config
{
  "command": "/usr/local/bin/node18",
  "args": ["server.js"]
}
```

---

## Category 2: Schema & Registration Errors

### 2.1 Invalid JSON Schema

**Problem:**
MCP clients silently drop tools with invalid schemas.

**Common Mistakes:**
```javascript
// WRONG - type as array at top level
{
  type: ["string", "null"],
  description: "Optional field"
}

// WRONG - missing required properties
{
  type: "object",
  properties: {
    query: { type: "string" }
  }
  // Missing "required" array
}

// WRONG - invalid property types
{
  query: {
    type: "string",
    default: 123  // Number default for string field
  }
}
```

**Fix:**
```javascript
// CORRECT - valid JSON Schema
{
  type: "object",
  properties: {
    query: {
      type: "string",
      description: "Search query"
    },
    limit: {
      type: "number",
      description: "Max results",
      default: 5
    }
  },
  required: ["query"]
}
```

**Validation:**
```bash
# Test schema with validator
echo '{"type": "object", ...}' | jq . > schema.json
# Open https://jsonschema.net/validator
# Paste schema and validate
```

### 2.2 Tools Not Registered Before Connection

**Problem:**
Tools defined after `server.connect()` are never exposed.

**Symptoms:**
- Server starts successfully
- tools/list returns empty array
- No errors logged

**Fix:**
```javascript
// WRONG - tools registered after connect
const server = new McpServer({...});
const transport = new StdioServerTransport();
await server.connect(transport);

server.tool("my-tool", ...);  // Too late!

// CORRECT - register before connect
const server = new McpServer({...});

server.tool("my-tool", ...);  // Register first

const transport = new StdioServerTransport();
await server.connect(transport);  // Then connect
```

### 2.3 Missing Tool Decorators (Python/FastMCP)

**Problem:**
Tools defined without proper decorators aren't registered.

**Fix:**
```python
# WRONG - missing decorator
from fastmcp import FastMCP
mcp = FastMCP("my-server")

def my_tool(arg: str) -> str:
    return f"Result: {arg}"

# CORRECT - use @mcp.tool() decorator
@mcp.tool()
def my_tool(arg: str) -> str:
    """Tool description here."""  # Docstring required!
    return f"Result: {arg}"
```

---

## Category 3: Handler & Execution Failures

### 3.1 Unhandled Exceptions

**Problem:**
Tool handlers throw exceptions instead of returning structured errors.

**Symptoms:**
- Server process dies silently
- No error response returned
- Host app shows spinner then nothing

**Fix:**
```javascript
// WRONG - throws on error
server.tool("fetch-data", { url: z.string() }, async ({ url }) => {
  const data = await fetch(url).then(r => r.json());  // Throws!
  return { content: [{ type: "text", text: JSON.stringify(data) }] };
});

// CORRECT - always return, never throw
server.tool("fetch-data", { url: z.string() }, async ({ url }) => {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return {
        isError: true,
        content: [{ type: "text", text: `HTTP ${response.status}` }]
      };
    }
    const data = await response.json();
    return { content: [{ type: "text", text: JSON.stringify(data) }] };
  } catch (err) {
    return {
      isError: true,
      content: [{ type: "text", text: `Failed: ${err.message}` }]
    };
  }
});
```

### 3.2 Incorrect Return Shape

**Problem:**
Tool returns don't match MCP content specification.

**Symptoms:**
- Tool appears to succeed but no output shown
- Silent failure with no error message

**Fix:**
```javascript
// WRONG - plain string
return { content: "result" };

// WRONG - missing type field
return { content: [{ text: "result" }] };

// WRONG - wrong content type
return { content: [{ type: "markdown", text: "result" }] };  // No markdown type!

// CORRECT
return {
  content: [
    { type: "text", text: "result" }
  ]
};

// CORRECT - multiple content blocks
return {
  content: [
    { type: "text", text: "Summary:" },
    { type: "text", text: JSON.stringify(data, null, 2) }
  ]
};
```

### 3.3 Tool Name Mismatch

**Problem:**
Tool name in definition doesn't match name used in calls.

**Symptoms:**
- Tool not found error
- Works in Inspector but not in host app

**Fix:**
```javascript
// Ensure exact match
server.tool("search_web", ...);  // Definition

// Client call
{
  "method": "tools/call",
  "params": {
    "name": "search_web",  // Must match exactly
    "arguments": {...}
  }
}
```

---

## Category 4: Authentication & Environment Problems

### 4.1 Missing Environment Variables

**Problem:**
GUI apps don't inherit shell environment.

**Symptoms:**
- Works in terminal
- Fails in Claude Desktop/Cursor
- Error: "Missing API key" or "undefined"

**Fix:**
```json
// Explicitly pass env vars in config
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/server.js"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "DATABASE_URL": "postgresql://...",
        "NODE_ENV": "production",
        "PATH": "/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```

### 4.2 Invalid API Keys

**Problem:**
API keys expired, revoked, or incorrect.

**Symptoms:**
- Authentication failed errors
- 401/403 responses from APIs
- Tools return empty results

**Fix:**
```javascript
// Validate at startup, not at call time
const requiredEnv = ["OPENAI_API_KEY", "BRAVE_API_KEY"];

for (const key of requiredEnv) {
  if (!process.env[key]) {
    process.stderr.write(`FATAL: Missing ${key}\n`);
    process.exit(1);
  }
  
  // Optional: validate format
  if (key === "OPENAI_API_KEY" && !process.env[key].startsWith("sk-")) {
    process.stderr.write(`WARNING: ${key} format looks incorrect\n`);
  }
}
```

### 4.3 Credential Rotation Issues

**Problem:**
Credentials rotated but server not restarted.

**Symptoms:**
- Suddenly stops working
- Logs show "invalid token"
- Server still running

**Fix:**
1. Update credentials in config
2. Fully restart host app (not just window)
3. Verify server process restarted:
   ```bash
   ps aux | grep mcp
   # Kill old processes
   kill -9 <pid>
   ```

---

## Category 5: Timeout & Performance Issues

### 5.1 Long-Running Operations

**Problem:**
Tools exceed client timeout (typically 30-60 seconds).

**Symptoms:**
- Tool starts but never completes
- Host app shows pending indefinitely
- No error returned

**Fix:**
```javascript
// Use progress notifications for long operations
server.tool(
  "process-large-file",
  { filePath: z.string() },
  async ({ filePath }, { sendNotification }) => {
    const lines = await readLines(filePath);
    
    for (let i = 0; i < lines.length; i++) {
      await processLine(lines[i]);
      
      // Signal liveness every 10 seconds
      if (i % 100 === 0) {
        await sendNotification({
          method: "notifications/progress",
          params: {
            progressToken: "processing",
            progress: i,
            total: lines.length
          }
        });
      }
    }
  }
);
```

### 5.2 Monolithic Tool Design

**Problem:**
Single tool tries to do too much work.

**Fix:**
```javascript
// WRONG - times out
server.tool("index-all-documents", async () => {
  return await indexEntireDirectory();  // 5 minutes
});

// CORRECT - paginated approach
server.tool("index-batch", {
  offset: z.number().default(0),
  batchSize: z.number().default(50)
}, async ({ offset, batchSize }) => {
  const batch = await getBatch(offset, batchSize);
  const indexed = await indexFiles(batch);
  const hasMore = batch.length === batchSize;
  
  return {
    content: [{
      type: "text",
      text: JSON.stringify({
        indexed: indexed.length,
        nextOffset: offset + batchSize,
        hasMore
      })
    }]
  };
});
```

### 5.3 Resource Exhaustion

**Problem:**
Server runs out of memory or file descriptors.

**Symptoms:**
- Crashes after multiple calls
- "Too many open files" errors
- Memory usage grows unbounded

**Fix:**
```javascript
// Implement cleanup
import { EventEmitter } from 'events';

class ResourceManager extends EventEmitter {
  constructor(maxOpen = 100) {
    super();
    this.openResources = new Set();
    this.maxOpen = maxOpen;
  }
  
  async acquire(resource) {
    if (this.openResources.size >= this.maxOpen) {
      // Close oldest resource
      const oldest = this.openResources.values().next().value;
      await this.release(oldest);
    }
    this.openResources.add(resource);
    return resource;
  }
  
  async release(resource) {
    await resource.close();
    this.openResources.delete(resource);
  }
}
```

---

## Systematic Troubleshooting Workflow

### Step 1: Reproduce the Error
```bash
# Document exact steps
# 1. What input causes the problem?
# 2. What environment (host app, OS)?
# 3. Can you reproduce consistently?
```

### Step 2: Check Server Startup
```bash
# Run server manually
cd /path/to/server
node index.js  # or python main.py

# Should see:
# - No immediate crashes
# - Server stays running
# - Listening for connections
```

### Step 3: Test with MCP Inspector
```bash
# Install inspector
npm install -g @modelcontextprotocol/inspector

# Run and connect
mcp-inspector
# Open http://localhost:5173
# Connect to server
# List tools
# Call tools manually
```

### Step 4: Check Logs
```bash
# Claude Desktop (macOS)
tail -n 20 -F ~/Library/Logs/Claude/mcp*.log

# Cursor
# View → Output → MCP Logs

# Custom logging
tail -f /tmp/mcp-server.log
```

### Step 5: Isolate the Problem
```
Server starts?
├─ NO → Check config syntax, dependencies, paths
│
└─ YES → Tools appear in Inspector?
   ├─ NO → Check schema, registration, decorators
   │
   └─ YES → Tool calls succeed in Inspector?
      ├─ NO → Check handler code, return shape, auth
      │
      └─ YES → Problem is in host app integration
         └─ Check host app logs, version compatibility
```

### Step 6: Fix and Verify
```bash
# After fixing, run verification
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | node index.js

# Should see valid JSON-RPC response with tools array
```

---

## Prevention Best Practices

### Development
1. **Use TypeScript/Pyright** for schema validation at compile time
2. **Add unit tests** for tool handlers
3. **Test with Inspector** before host app integration
4. **Log to stderr** not stdout
5. **Validate inputs** before processing

### Configuration
1. **Use absolute paths** in all configs
2. **Explicitly declare env vars** in `env` section
3. **Version pin dependencies** (package.json, requirements.txt)
4. **Document required env vars** in README

### Error Handling
1. **Wrap all handlers** in try/catch
2. **Return structured errors** with `isError: true`
3. **Include error context** in responses
4. **Log errors to file** for debugging

### Performance
1. **Use progress notifications** for operations >10s
2. **Paginate large result sets**
3. **Implement caching** where appropriate
4. **Set reasonable timeouts** on external calls

### Security
1. **Validate all inputs** (URLs, file paths, queries)
2. **Sanitize outputs** to prevent injection
3. **Use least-privilege credentials**
4. **Rotate API keys** regularly
5. **Monitor usage patterns** for anomalies

---

## Quick Reference: Error Codes

| Error Pattern | Most Likely Cause | Quick Fix |
|--------------|-------------------|-----------|
| "Not connected" | Windows cmd.exe policy | Use `"command": "cmd"` with `/c` |
| Empty tools list | Schema validation failure | Validate with jsonschema.net |
| "Module not found" | Missing dependencies | Run `npm install` or `pip install` |
| Silent crash | Unhandled exception | Wrap in try/catch, return isError |
| Timeout | Long operation | Add progress notifications |
| "Missing API key" | Env var not inherited | Add to `env` in config |
| "Session not found" | HTTP session expired | Implement re-initialization |
| "Invalid params" (-32602) | Capability mismatch | Check initialize exchange |

---

## Getting Help

When stuck:

1. **Check documentation**: https://modelcontextprotocol.io
2. **Test with Inspector**: Isolate server vs client issue
3. **Review logs**: Server stderr and host app logs
4. **Search GitHub issues**: https://github.com/modelcontextprotocol/servers/issues
5. **Provide diagnostics**:
   - Config file (sanitized)
   - Log excerpts
   - Steps to reproduce
   - Environment details (OS, versions)
