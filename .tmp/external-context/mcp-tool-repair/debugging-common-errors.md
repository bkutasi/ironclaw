---
source: Markaicode & Model Context Protocol Official Docs
library: MCP (Model Context Protocol)
topic: Debugging common errors and repair mechanisms
fetched: 2026-03-26
official_docs: https://modelcontextprotocol.io/legacy/tools/debugging
---

# MCP Debugging: Common Errors and How to Fix Them

## Why MCP Errors Are Hard to Debug

MCP (Model Context Protocol) runs over stdio or SSE. When something goes wrong, the error often lives in a subprocess that the host app swallows.

The protocol has three layers where things break:

```
Host App (Claude Desktop / Cursor)
    │
    ▼
MCP Client (JSON-RPC 2.0 over stdio or SSE)
    │
    ▼
MCP Server (your tool implementation)
```

A failure at any layer looks the same from the top: the tool doesn't appear, or it appears and silently fails.

---

## Error 1: Server Fails to Start

**Symptoms:**
- Tool list is empty in the host app
- No error shown — server just isn't there
- `claude_desktop_config.json` looks correct

### Step 1: Run the Server Manually

```bash
# Run the exact command from your config, in the exact working directory
node /path/to/your/server/index.js

# Or for Python servers
python /path/to/server/main.py
```

If it crashes here, you'll see the real error. Common causes:

- **`Error: Cannot find module '...'`** → `npm install` wasn't run in the server directory
- **`ModuleNotFoundError`** → virtualenv not activated, or `pip install -r requirements.txt` was skipped
- **`ENOENT: no such file or directory`** → path in config is wrong (use absolute paths, not `~/`)

### Step 2: Verify Your Config Path

```json
// claude_desktop_config.json — WRONG (tilde doesn't expand)
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["~/projects/my-server/index.js"]
    }
  }
}

// CORRECT — use the full absolute path
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/Users/mark/projects/my-server/index.js"]
    }
  }
}
```

### Step 3: Check Node / Python Version

```bash
# MCP SDK requires Node 18+
node --version

# Python SDK requires 3.10+
python --version

# If wrong version is resolving, use full path in config
which node   # use this path in "command"
```

**If it fails:**
- **`SyntaxError: Unexpected token '?'`** (optional chaining) → Node < 14. Upgrade to 18+.
- **`ImportError: cannot import name 'TypeAlias'`** → Python < 3.10. Use `python3.11` explicitly.

---

## Error 2: Server Starts But Tools Don't Appear

**Symptoms:**
- Server process is running
- Host app shows the server as connected
- Tool list is empty or missing specific tools

### Step 1: Validate Your Tool Schema

The most common cause is an invalid JSON Schema in your tool definition. MCP clients silently drop tools with schema errors.

```javascript
// ❌ WRONG — "type" must be a string, not an array at the top level
server.tool("search", {
  query: z.string(),
  limit: z.number().optional()  // optional() generates {"anyOf": [...]} — fine
}, async ({ query, limit }) => { ... });

// ✅ CORRECT — check that your Zod schema serializes to valid JSON Schema
import { zodToJsonSchema } from "zod-to-json-schema";
const schema = z.object({ query: z.string(), limit: z.number().optional() });
console.log(JSON.stringify(zodToJsonSchema(schema), null, 2));
// Paste output into https://jsonschema.net/validator to check
```

### Step 2: Log the tools/list Response

```bash
# Send a raw tools/list request over stdio to see exactly what your server returns
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | node index.js
```

**Expected output:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "search",
        "description": "...",
        "inputSchema": { "type": "object", "properties": { ... } }
      }
    ]
  }
}
```

If `tools` is an empty array, your server isn't registering the tool. If the response is malformed JSON, that's your bug.

### Step 3: Check Tool Registration

```javascript
// ✅ Tool must be registered BEFORE server.connect()
const server = new McpServer({ name: "my-server", version: "1.0.0" });

server.tool("search", "Search documents", {
  query: z.string().describe("Search query"),
}, async ({ query }) => {
  return { content: [{ type: "text", text: await doSearch(query) }] };
});

// connect() AFTER all tools are registered
const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## Error 3: Tool Is Called But Returns an Error

**Symptoms:**
- Tool appears in the list
- Model attempts to call it
- Response contains `isError: true` or the call hangs

### Step 1: Wrap Your Handler in Try/Catch

MCP servers should never throw — they must return a structured error response. An unhandled exception kills the server process silently.

```javascript
// ❌ WRONG — unhandled exception kills the server
server.tool("fetch-data", { url: z.string() }, async ({ url }) => {
  const data = await fetch(url).then(r => r.json()); // throws on network error
  return { content: [{ type: "text", text: JSON.stringify(data) }] };
});

// ✅ CORRECT — always return, never throw
server.tool("fetch-data", { url: z.string() }, async ({ url }) => {
  try {
    const data = await fetch(url).then(r => r.json());
    return { content: [{ type: "text", text: JSON.stringify(data) }] };
  } catch (err) {
    return {
      isError: true,
      content: [{ type: "text", text: `Failed to fetch ${url}: ${err.message}` }]
    };
  }
});
```

### Step 2: Check Return Shape

MCP has an exact content shape. Any deviation causes silent failure.

```javascript
// ❌ WRONG — returning a plain string
return { content: "result here" };

// ❌ WRONG — missing "type" field
return { content: [{ text: "result here" }] };

// ✅ CORRECT
return {
  content: [{ type: "text", text: "result here" }]
};

// ✅ CORRECT — multiple content blocks
return {
  content: [
    { type: "text", text: "Here's what I found:" },
    { type: "text", text: JSON.stringify(results, null, 2) }
  ]
};
```

---

## Error 4: Authentication / Environment Variable Failures

**Symptoms:**
- Server works when you run it manually in your terminal
- Fails inside Claude Desktop or Cursor
- Error references a missing API key or undefined env var

This happens because GUI apps on macOS and Windows don't inherit your shell's environment. The `PATH`, `API_KEY`, and other variables you set in `.zshrc` or `.bashrc` aren't available.

### Step 1: Pass Env Vars Explicitly in Config

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/Users/mark/projects/my-server/index.js"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "DATABASE_URL": "postgresql://...",
        "NODE_ENV": "production"
      }
    }
  }
}
```

### Step 2: Validate Env at Startup

Add an explicit check when your server starts, not when the tool is called:

```javascript
// Fail loudly at startup, not silently at call time
const requiredEnv = ["OPENAI_API_KEY", "DATABASE_URL"];

for (const key of requiredEnv) {
  if (!process.env[key]) {
    // Write to stderr — MCP clients capture stdout for JSON-RPC
    process.stderr.write(`FATAL: Missing required env var: ${key}\n`);
    process.exit(1);
  }
}
```

**Why `stderr`?** MCP uses `stdout` for JSON-RPC messages. Writing errors to `stdout` corrupts the protocol stream and produces confusing parse errors. Always write debug output and errors to `stderr`.

---

## Error 5: Timeouts on Long-Running Tools

**Symptoms:**
- Tool starts executing
- No response after 30–60 seconds
- Host app shows tool as pending indefinitely

MCP clients have a default request timeout (typically 30s for Claude Desktop). Long-running operations — database queries, web scraping, large file processing — will hit this.

### Step 1: Stream Progress Updates

For operations over ~10 seconds, use MCP's progress notification to signal liveness:

```javascript
server.tool(
  "process-large-file",
  { filePath: z.string() },
  async ({ filePath }, { sendNotification }) => {
    const lines = await readLines(filePath);
    const results = [];

    for (let i = 0; i < lines.length; i++) {
      results.push(await processLine(lines[i]));

      // Send progress every 100 lines so the client knows you're alive
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

    return { content: [{ type: "text", text: `Processed ${results.length} lines` }] };
  }
);
```

### Step 2: Break Large Tasks Into Chunks

If a single tool call genuinely takes more than 30s, redesign it:

```javascript
// ❌ One monolithic tool call that times out
server.tool("index-all-documents", { path: z.string() }, async ({ path }) => {
  return await indexEntireDirectory(path); // 5 minutes, always times out
});

// ✅ Paginated tool — the model calls it repeatedly
server.tool("index-documents-batch", {
  path: z.string(),
  offset: z.number().default(0),
  batchSize: z.number().default(50)
}, async ({ path, offset, batchSize }) => {
  const files = await getFileBatch(path, offset, batchSize);
  const indexed = await indexFiles(files);
  const hasMore = indexed.length === batchSize;

  return {
    content: [{
      type: "text",
      text: JSON.stringify({ indexed: indexed.length, nextOffset: offset + batchSize, hasMore })
    }]
  };
});
```

---

## Add Structured Logging (Stop Guessing)

The single most useful thing you can do is log every incoming request and outgoing response to a file. Since `stdout` is reserved for JSON-RPC, write logs to a file:

```javascript
import fs from "fs";

const logFile = fs.createWriteStream("/tmp/mcp-server.log", { flags: "a" });

function log(level: string, data: unknown) {
  const entry = JSON.stringify({ ts: new Date().toISOString(), level, ...data as object });
  logFile.write(entry + "\n");
}

// Log every tool call
server.tool("my-tool", schema, async (params) => {
  log("info", { event: "tool_call", tool: "my-tool", params });
  try {
    const result = await doWork(params);
    log("info", { event: "tool_success", tool: "my-tool" });
    return { content: [{ type: "text", text: result }] };
  } catch (err) {
    log("error", { event: "tool_error", tool: "my-tool", error: err.message });
    return { isError: true, content: [{ type: "text", text: err.message }] };
  }
});
```

Then tail the log while you reproduce the issue:

```bash
tail -f /tmp/mcp-server.log
```

---

## Verification

After applying fixes, run this end-to-end check:

```bash
# 1. Verify server starts and lists tools
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | node index.js

# 2. Call a specific tool directly
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"your-tool-name","arguments":{"key":"value"}}}' | node index.js
```

**You should see:** A valid JSON-RPC response with `result.content` containing your tool's output — no `isError: true`, no crash.

---

## What You Learned

- MCP errors are almost always in one of five categories: startup, schema, handler, auth, or timeout
- Always write debug output to `stderr` — `stdout` is the JSON-RPC wire
- Pass env vars explicitly in the host app config; GUI apps don't inherit shell env
- Wrap every tool handler in try/catch and return structured errors instead of throwing
- Use paginated tools or progress notifications for anything that takes over 10 seconds

**Limitation:** These patterns cover stdio transport. SSE transport has additional failure modes around HTTP connection handling and reconnection logic.
