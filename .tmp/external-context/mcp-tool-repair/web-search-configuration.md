---
source: LocalLLM.in & Glukhov.org
library: MCP (Model Context Protocol)
topic: Web-search tool configuration and setup
fetched: 2026-03-26
official_docs: https://modelcontextprotocol.io
---

# Web-Search Tool Configuration in MCP Servers

## Overview

Web-search MCP servers enable AI models to access real-time information from the internet. This guide covers configuration for popular search providers and common troubleshooting.

---

## Tavily MCP Server (Recommended for Beginners)

### Setup Steps

**1. Get API Key**
- Visit https://app.tavily.com/home
- Create free account (no credit card required)
- Get 1,000 free API credits monthly
- Click "Generate MCP Link" to get configuration URL

**2. Configure in mcp.json**

```json
{
  "mcpServers": {
    "tavily-remote": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_API_KEY_HERE"
      ]
    }
  }
}
```

**Critical:** Replace `YOUR_API_KEY_HERE` with actual API key from Tavily dashboard.

**3. Available Tools**
- `tavily_search` - General web searches
- `tavily_extract` - Extract content from web pages
- `tavily_crawl` - Crawl websites for comprehensive data
- `tavily_map` - Structure search results into insights

**4. Test Query**
```
"What's the weather in London today?"
"Search for recent smartphone releases"
"Who won the Nobel Prize in Physics this year?"
```

---

## Brave Search MCP Server

### Setup Steps

**1. Get API Key**
- Visit https://brave.com/search/api/
- Create free account (payment details required, no charge)
- Get 2,000 queries per month
- Copy API key

**2. Configure in mcp.json (GUI Method)**

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key_here"
      }
    }
  }
}
```

**3. Configure for API Access (HTTP/SSE)**

Brave Search subprocess only works with GUI. For API access, deploy as HTTP server:

**Option A: Docker Deployment**
```bash
docker run -d --rm \
  -p 8080:8080 \
  -e BRAVE_API_KEY="YOUR_API_KEY_HERE" \
  -e PORT="8080" \
  --name brave-search-server \
  shoofio/brave-search-mcp-sse:latest
```

Update mcp.json:
```json
{
  "mcpServers": {
    "brave-search": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```

**Option B: Manual Installation**
```bash
git clone <repository_url>
cd brave-search-mcp-sse
npm install
npm run build

# Create .env file
echo "BRAVE_API_KEY=YOUR_KEY" > .env
echo "PORT=8080" >> .env

npm start
```

---

## DuckDuckGo MCP Server (No API Key Required)

### Python Implementation

```python
#!/usr/bin/env python3
"""MCP Server for DuckDuckGo Web Search"""

import asyncio
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

app = Server("duckduckgo-search")

async def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return formatted results"""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result in soup.select('.result')[:max_results]:
            title = result.select_one('.result__title')
            snippet = result.select_one('.result__snippet')
            link = result.select_one('.result__url')
            
            if title and snippet:
                results.append({
                    "title": title.get_text(strip=True),
                    "snippet": snippet.get_text(strip=True),
                    "url": link.get_text(strip=True) if link else ""
                })
        
        if not results:
            return "No results found."
        
        formatted = [f"Found {len(results)} results for '{query}':\n"]
        for i, r in enumerate(results, 1):
            formatted.append(f"\n{i}. **{r['title']}**")
            formatted.append(f"   {r['snippet']}")
            formatted.append(f"   {r['url']}")
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"Search error: {str(e)}"

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_web",
            description="Search the web using DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Maximum results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_web":
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        result = await search_duckduckgo(query, max_results)
        return [TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

```json
{
  "mcpServers": {
    "duckduckgo-search": {
      "command": "python",
      "args": ["/absolute/path/to/duckduckgo_mcp.py"]
    }
  }
}
```

---

## LM Studio API Integration (/v1/responses)

For developers using LM Studio 0.3.29+:

### Basic API Request

```bash
curl http://127.0.0.1:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-8b",
    "tools": [{
      "type": "mcp",
      "server_label": "tavily",
      "server_url": "https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_KEY",
      "allowed_tools": ["tavily_search"]
    }],
    "input": "What are the latest AI developments this week?"
  }'
```

### Stateful Conversations

```bash
# Initial request
curl http://127.0.0.1:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-8b",
    "tools": [{
      "type": "mcp",
      "server_label": "tavily",
      "server_url": "https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_KEY",
      "allowed_tools": ["tavily_search"]
    }],
    "input": "Search for current weather in London"
  }'

# Response includes: {"id": "resp_abc123", ...}

# Continue conversation
curl http://127.0.0.1:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-8b",
    "input": "How does that compare to Paris?",
    "previous_response_id": "resp_abc123"
  }'
```

### Streaming Responses

```bash
curl http://127.0.0.1:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-8b",
    "tools": [{
      "type": "mcp",
      "server_label": "tavily",
      "server_url": "https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_KEY",
      "allowed_tools": ["tavily_search"]
    }],
    "input": "Latest tech news today",
    "stream": true
  }'
```

Receives Server-Sent Events:
- `response.created` - When response begins
- `response.output_text.delta` - Text chunks as generated
- `response.completed` - Generation finished

---

## SearXNG Public MCP Server

### Configuration

```json
{
  "mcpServers": {
    "searxng-public": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-searxng"
      ],
      "env": {
        "SEARXNG_INSTANCE": "https://searx.be"
      }
    }
  }
}
```

### Features
- Access to multiple search engines
- Privacy-focused (no tracking)
- Configurable instances
- Supports text, images, news, videos

---

## Common Configuration Issues

### Issue 1: Server Won't Start

**Symptoms:** Server shows gray/inactive in settings

**Solutions:**
```bash
# Verify Node.js installation
node --version
npm --version

# Test server manually
npx -y @modelcontextprotocol/server-brave-search --help

# Check for errors
cd /path/to/server
npm install
node index.js
```

### Issue 2: Tools Don't Appear

**Symptoms:** Server connected but tools list empty

**Solutions:**
1. Check tool registration in server code
2. Validate JSON schema with https://jsonschema.net/validator
3. Test with MCP Inspector:
   ```bash
   npx @modelcontextprotocol/inspector
   # Open http://localhost:5173
   # Connect and list tools
   ```

### Issue 3: Model Not Using Search

**Symptoms:** Model responds from training data instead of searching

**Solutions:**
- Confirm model supports function calling (look for 🔨 icon)
- Select correct MCP server at bottom of chat
- Use explicit queries: "What is today's date?" not "Tell me about dates"
- Increase context window to 16k+ for complex queries

### Issue 4: Timeout Errors

**Symptoms:** Search requests fail with timeout

**Solutions:**
- Check internet connectivity
- Verify API key validity
- Monitor rate limits (free tiers have monthly caps)
- Implement progress notifications for long searches

### Issue 5: API-Specific Issues (LM Studio)

**Symptoms:** API requests return errors

**Solutions:**
```bash
# Verify server is running
# Developer → Server → Status should be "Running"

# Enable remote MCP
# Developer → Settings → Allow MCP → Remote: On

# Check port availability
lsof -i :1234

# Validate JSON syntax
curl http://127.0.0.1:1234/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "test"}'  # Should get proper error, not parse error
```

---

## Performance Optimization

### Query Optimization
- Be specific: "Current Microsoft stock price" vs "Microsoft stock"
- Use trigger words: "latest", "currently", "recent"
- Batch queries in single conversation

### Rate Limit Management
- **Tavily:** 1,000 credits/month ≈ 33 searches/day
- **Brave:** 2,000 queries/month ≈ 67 searches/day
- Monitor usage in provider dashboards
- Set up alerts when approaching limits

### Context Window Management
- Default LM Studio context: 4k tokens (too small)
- Recommended: 16k+ tokens for MCP
- Balance search results with conversation history
- Watch VRAM usage on GPU (KV cache grows with context)

---

## Multiple MCP Server Configuration

Run different servers for different purposes:

```json
{
  "mcpServers": {
    "tavily-general": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://mcp.tavily.com/mcp/?tavilyApiKey=KEY1"]
    },
    "brave-news": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {"BRAVE_API_KEY": "KEY2"}
    },
    "duckduckgo-backup": {
      "command": "python",
      "args": ["/path/to/duckduckgo_mcp.py"]
    }
  }
}
```

**Note:** Select only one server per conversation to avoid conflicts.

---

## Security Considerations

1. **Never hardcode API keys** in version control
2. **Use environment variables** or secret managers
3. **Implement rate limiting** to prevent abuse
4. **Validate all inputs** before passing to search engines
5. **Sanitize URLs** to prevent SSRF attacks
6. **Monitor usage** for unusual patterns

---

## Testing Checklist

Before deploying web-search MCP:

- [ ] Server starts without errors
- [ ] Tools appear in MCP Inspector
- [ ] Manual tool calls return valid responses
- [ ] API keys are valid and not expired
- [ ] Rate limits are within acceptable range
- [ ] Error handling returns structured errors
- [ ] Logging configured to stderr
- [ ] Context window sufficient for results
- [ ] Model supports function calling
- [ ] Internet connectivity verified
