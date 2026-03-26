---
source: GitHub Issues & Microsoft Q&A
library: MCP (Model Context Protocol)
topic: Fix "Builder not available" and "Not connected" errors
fetched: 2026-03-26
official_docs: https://github.com/modelcontextprotocol/servers/issues/1082
---

# How to Fix "Builder not available for repairing tool" and "Not connected" Errors

## Error: "Not connected" - Most Common MCP Tool Failure

This is the most frequently reported MCP error across multiple platforms (Cursor, Claude Desktop, Cline, Roo Code, Windsurf).

### Root Causes

1. **Windows cmd.exe execution policy** - `npx` cannot execute in subprocess context
2. **Session management issues** - MCP session IDs expiring or being deleted prematurely
3. **Server initialization failures** - Server crashes before establishing connection
4. **Environment variable inheritance** - GUI apps don't inherit shell environment

---

## Solution 1: Windows cmd.exe Fix (Most Common)

### Problem
On Windows, running `npx` directly as a command fails because Windows policies don't allow temporary execution in subprocess contexts.

### Fix: Use `cmd /c` Prefix

**BEFORE (Broken):**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

**AFTER (Fixed):**
```json
{
  "mcpServers": {
    "github": {
      "command": "cmd",
      "args": [
        "/c",
        "npx",
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

**The Key Change:**
- Changed `"command": "npx"` to `"command": "cmd"`
- Added `"/c"` as first argument
- Moved `npx` command into args array

This allows Windows to interpret the command correctly and launch the server.

### Alternative: Use Absolute Path to npx

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "C:\\nvm4w\\nodejs\\npx.ps1",
      "args": ["-y", "chrome-devtools-mcp@latest"]
    }
  }
}
```

---

## Solution 2: Install Package Globally (Windows Workaround)

If `cmd /c` doesn't work, install the MCP server globally:

```bash
npm install -g @modelcontextprotocol/server-github
```

Then update config:

```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": ["C:/Users/YourName/AppData/Roaming/npm/node_modules/@modelcontextprotocol/server-github/dist/index.js"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

**Downside:** Takes up local storage and requires manual updates.

---

## Solution 3: Fix Session Management (Azure AI Foundry / Streamable HTTP)

### Problem
When using Streamable HTTP transport, the client may send a `DELETE /` request before calling tools, destroying the session.

**Error from server:**
```json
{"error":{"code":-32001,"message":"Session not found"},"id":"","jsonrpc":"2.0"}
```

### Fix: Intercept DELETE Requests (Server-Side)

Add middleware to ignore session deletion:

```csharp
// ASP.NET Core MCP Server
app.Use(async (context, next) =>
{
    if (context.Request.Method == HttpMethods.Delete)
    {
        Console.WriteLine($"Intercepted MCP DELETE session call at {context.Request.Path}");
        var sessionId = context.Request.Headers["Mcp-Session-Id"].ToString();
        Console.WriteLine($"Session ID: {sessionId}");
        
        context.Response.StatusCode = StatusCodes.Status200OK;
        context.Response.ContentType = "application/json";
        await context.Response.WriteAsync("{\"result\":\"session ignored\"}");
        return; // Don't call next middleware
    }
    
    await next();
});
```

### Alternative: Configure Session Timeout

```csharp
builder.Services.AddMcpServer()
    .WithHttpTransport(options =>
    {
        options.IdleTimeout = TimeSpan.FromMinutes(30);
        options.MaxIdleSessionCount = 100;
    });
```

---

## Solution 4: Fix rmcp Session Re-initialization (OpenAI Codex Bug)

### Problem
When MCP server session expires (HTTP 404), rmcp client doesn't re-initialize as required by MCP spec.

**MCP Spec Requirement:**
> When a client receives HTTP 404 in response to a request containing an `Mcp-Session-Id`, it **MUST** start a new session by sending a new `InitializeRequest` without a session ID attached.

### Workaround: Detect and Reconnect

```python
# Client-side session recovery
class MCPClientWithRecovery:
    async def call_tool(self, tool_name: str, args: dict):
        try:
            return await self._call_tool_impl(tool_name, args)
        except SessionExpiredError:
            # Re-initialize session
            await self.initialize()
            # Retry the call
            return await self._call_tool_impl(tool_name, args)
```

---

## Solution 5: Fix FastMCP Context Parameter Error

### Problem
IAM MCP server tools fail with validation error requiring `ctx` parameter.

**Error:**
```
Error executing tool list_users: 1 validation error for list_usersArguments
ctx Field required [type=missing, input_value={'max_items': 5}, input_type=dict]
```

### Root Cause
Using `CallToolResult` instead of `Context` type for ctx parameter.

### Fix: Use Correct Context Type

**BEFORE (Incorrect):**
```python
from mcp.types import CallToolResult

@mcp.tool()
async def list_users(
    ctx: CallToolResult,  # Wrong type - gets exposed in schema
    path_prefix: Optional[str] = None,
    max_items: int = 100,
) -> UsersListResponse:
```

**AFTER (Fixed):**
```python
from mcp.server.fastmcp import Context

@mcp.tool()
async def list_users(
    ctx: Context,  # Correct type - automatically excluded from schema
    path_prefix: Optional[str] = None,
    max_items: int = 100,
) -> UsersListResponse:
```

**Why this works:** FastMCP framework automatically excludes parameters typed as `Context` from tool schemas.

---

## Solution 6: Fix Tool Response Shape Errors

### Problem
Tool call response error where type and content are absent.

**Error:**
```
fix tool call response error which is type and content are absent
```

### Fix: Return Proper Content Structure

**BEFORE (Broken):**
```typescript
return {
  content: "plain text result"  // Wrong - should be array
};
```

**AFTER (Fixed):**
```typescript
return {
  content: [
    {
      type: "text",
      text: "properly structured result"
    }
  ]
};
```

---

## Diagnostic Checklist

When encountering "Builder not available" or "Not connected" errors:

### 1. Check Server Startup
```bash
# Run server command manually
cd /path/to/server
node index.js  # or python main.py

# Should see no errors and server should stay running
```

### 2. Verify Configuration
```bash
# Validate JSON syntax
cat ~/.cursor/mcp.json | jq .

# Check for common issues:
# - Absolute paths (not ~/)
# - cmd /c on Windows
# - Environment variables in "env" section
```

### 3. Test with MCP Inspector
```bash
# Install inspector
npm install -g @modelcontextprotocol/inspector

# Run inspector
mcp-inspector

# Connect to your server at http://localhost:5173
# Test tools manually
```

### 4. Check Client Logs
```bash
# Claude Desktop (macOS)
tail -n 20 -F ~/Library/Logs/Claude/mcp*.log

# Cursor
# View → Output → MCP Logs

# VS Code (Cline extension)
# View → Output → Cline channel
```

### 5. Verify Environment Variables
```bash
# Test that env vars are accessible
echo $GITHUB_PERSONAL_ACCESS_TOKEN

# In config, explicitly pass env vars:
{
  "env": {
    "API_KEY": "value"
  }
}
```

---

## Common Platform-Specific Fixes

### Windows
- Use `"command": "cmd"` with `"/c"` prefix
- Install packages globally if npx fails
- Use forward slashes or escaped backslashes in paths

### macOS
- Use absolute paths (tilde ~ doesn't expand)
- Check SIP restrictions on process execution
- Verify Node.js is in PATH for GUI apps

### Linux
- Check file permissions on server scripts
- Ensure virtualenv is activated
- Verify systemd/user service restrictions

### Azure AI Foundry
- Intercept DELETE requests
- Configure session timeouts
- Add `x-mcp-client-tenant-id` header if needed

---

## Prevention Best Practices

1. **Always use absolute paths** in configuration files
2. **Explicitly declare environment variables** in `env` section
3. **Wrap tool handlers in try/catch** - never throw unhandled exceptions
4. **Log to stderr, not stdout** - stdout is for JSON-RPC only
5. **Test with MCP Inspector** before integrating with host app
6. **Use structured error responses** with `isError: true`
7. **Implement session recovery** for HTTP transport
8. **Validate JSON schemas** before deploying tools
