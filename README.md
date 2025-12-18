# PIPO to SAP Integration Suite

## Using Python client and ts SERVER: mcp-integration-suite

## Instructions

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure Environment
- Copy `.env` and update the values as needed.
- Ensure `LLM_DEPLOYMENT_ID` is set for SAP AI Core.
- HANA credentials are now loaded from `.env` (moved from hardcoded).

### 3. Set up MCP Server
Clone and build the MCP server:
```bash
git clone https://github.com/1nbuc/mcp-integration-suite.git
cd mcp-integration-suite
npm install
npm run build
```

### 4. Run the Client
For CLI mode:
```bash
uv run python deepagentclient.py <path/to/mcp-integration-suite/dist/index.js>
```

For FastAPI server mode:
```bash
uv run python deepagentclient.py
```
Then visit http://localhost:8080

## Components
- `deepagentclient.py`: Main client with FastAPI API and CLI support
- `newclient.py`: Simpler CLI client
- MCP server: External TypeScript server from https://github.com/1nbuc/mcp-integration-suite

## Blog
https://community.sap.com/t5/technology-blog-posts/by-members/using-integration-suite-with-the-power-of-ai/ba-p/14067293
