import asyncio
import sys
import os
import json
import logging
from typing import Optional, Any, Type, Dict
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import create_model, BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    filename="pipo_client.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ==========================================================
# SAP GEN AI HUB LLM
# ==========================================================
def create_sap_llm():
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing in .env")

    return ChatOpenAI(
        deployment_id=deployment_id,
        temperature=0,
    )

# ==========================================================
# ASYNC MCP TOOL WRAPPER
# ==========================================================
class MCPAsyncTool(BaseTool):
    """
    LangChain tool that calls an MCP tool asynchronously using the SAME event loop.
    """
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Sync run is not supported")

    async def _arun(self, *args, **kwargs) -> str:
        logger.info(f"[MCP-TOOL] Executing → {self.mcp_tool_name} | Args = {kwargs}")
        result = await self.session.call_tool(self.mcp_tool_name, kwargs)

        if not result.content:
            logger.info(f"[MCP-TOOL] EMPTY result from {self.mcp_tool_name}")
            return ""

        outputs = []
        for c in result.content:
            if getattr(c, "text", None):
                outputs.append(c.text)
            elif getattr(c, "json", None):
                outputs.append(json.dumps(c.json, ensure_ascii=False))
            else:
                outputs.append(str(c))

        return "\n".join(outputs)

# ==========================================================
# Build + Update IFlow Helper
# ==========================================================
async def _build_and_update_iflow(client, iflow_id: str, files: Dict[str, str], autoDeploy: bool = True):
    """
    Sends a JSON structure to the MCP server's update-iflow tool.
    files = { "path/to/file": "content" }
    """
    assert client.session is not None, "MCP session not initialized"

    payload = {
        "id": iflow_id,
        "files": [{"filepath": path, "content": content} for path, content in files.items()],
        "autoDeploy": bool(autoDeploy),
    }

    logger.info("[LOCAL] build_and_update_iflow payload prepared")


    # ============================================
    # SAVE LOCAL COPY OF FILES  (NEW BLOCK)
    # ============================================
    local_dir = f"./_generated_iflows/{iflow_id}"
    os.makedirs(local_dir, exist_ok=True)

    for path, content in files.items():
        full_path = os.path.join(local_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.info(f"[LOCAL] Saved generated iFlow files → {local_dir}")
    print(f"\n📁 Local copy saved under: {local_dir}\n")

    result = await client.session.call_tool("update-iflow", payload)

    if not result or not result.content:
        return {"error": "empty response from update-iflow"}

    for c in result.content:
        text = getattr(c, "text", None)
        if text:
            try:
                return json.loads(text)
            except:
                return {"raw": text}

    return {"raw": str(result)}

# ==========================================================
# LangChain Tool Wrapper (Local)
# ==========================================================
from pydantic import PrivateAttr

class BuildAndUpdateIflowInput(BaseModel):
    iflow_id: str = Field(..., description="iFlow ID / name")
    files: Dict[str, str] = Field(..., description="mapping: relative path → content")
    autoDeploy: bool = Field(True, description="deploy after update")


class BuildAndUpdateIflowTool(BaseTool):
    name: str = "build_and_update_iflow"
    description: str = (
        "Build and update a full iFlow by calling the MCP update-iflow tool."
    )
    args_schema: Type[BaseModel] = BuildAndUpdateIflowInput

    _client: Any = PrivateAttr()

    def __init__(self, client: "MCPClient"):
        super().__init__()
        self._client = client

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use async")

    async def _arun(self, iflow_id: str, files: Dict[str, str], autoDeploy: bool = True):
        return await _build_and_update_iflow(self._client, iflow_id, files, autoDeploy)



# ==========================================================
# MCP CLIENT
# ==========================================================
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.llm = create_sap_llm()
        self.agent_tools: list[BaseTool] = []
        self.worker_agent: Optional[AgentExecutor] = None

    # ------------------------------------------------------
    async def connect_to_server(self, server_script: str):
        if server_script.endswith(".py"):
            cmd = "python"
        elif server_script.endswith(".js"):
            cmd = "node"
        else:
            raise ValueError("Server script must be .py or .js")

        params = StdioServerParameters(command=cmd, args=[server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        await self._build_agent_tools()

        # ADD OUR LOCAL TOOL
        self.agent_tools.append(BuildAndUpdateIflowTool(self))

        await self._build_worker_agent()

        tools = [t.name for t in (await self.session.list_tools()).tools]
        print("\nMCP Connected! Tools:", tools)
        logger.info(f"[MCP] Connected with tools: {tools}")

    # ------------------------------------------------------
    async def _build_agent_tools(self):
        assert self.session is not None
        tlist = await self.session.list_tools()

        for tdef in tlist.tools:
            if tdef.name == "update-iflow":
                continue  # Skip MCP update-iflow, use local wrapper instead
            schema = tdef.inputSchema or {}
            props = schema.get("properties", {}) or {}
            required = set(schema.get("required", []))

            fields: Dict[str, tuple[Any, Any]] = {}
            for key in props:
                if key in required:
                    fields[key] = (Any, ...)
                else:
                    fields[key] = (Any, None)

            InputModel = create_model(f"{tdef.name}_Input", **fields)

            tool = MCPAsyncTool(
                name=tdef.name,
                description=tdef.description or "",
                args_schema=InputModel,
                session=self.session,
                mcp_tool_name=tdef.name,
            )

            self.agent_tools.append(tool)

        logger.info(f"[MCP] Loaded {len(self.agent_tools)} tools")

    # ------------------------------------------------------
    async def _build_worker_agent(self):

        worker_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are SAP Integration Suite WORKER Agent.
STRICT RULES FOR CREATING OR UPDATING IFLOWS:

You must ALWAYS verify whether an iFlow exists before creating or updating it.

Allowed tool calls for verification:
- get-iflow
- package / packages (if needed to verify package exists)

1. CORRECT LOGIC:
- When user asks to create or update an iFlow:
    First call: get-iflow {{ "id": "<IFLOW_ID>" }}

- If get-iflow returns NOT FOUND (404):
    - Create the iFlow using:
        create-empty-iflow {{ package: "<PACKAGE>", name: "<IFLOW_ID>" }}

- If the iFlow exists:
    - Do NOT call create-empty-iflow.

- After creation or confirmation:
    - Generate FULL iFlow files (XML, MANIFEST.MF, .project, metadata, props).
    - Then call build_and_update_iflow.

- NEVER call build_and_update_iflow without full file dictionary.


2. Before calling build_and_update_iflow:
   YOU MUST ALWAYS generate a complete "files" dictionary containing:
      - scenarioflows XML
      - MANIFEST.MF
      - .project
      - metadata.prop
      - parameters.prop
      - parameters.propdef
      - scripts (if needed)
   This is MANDATORY.

3. NEVER call build_and_update_iflow without the "files" dictionary.
   If you don't have files, STOP and generate them first.

4. NEVER call tools in this order:
      list-iflow-examples → build_and_update_iflow
   This is ALWAYS incorrect.

5. The workflow must be:
      (A) Generate files JSON
      (B) Call build_and_update_iflow with iflow_id, files, autoDeploy=true
      (C) Wait for tool response
      (D) Report status to user


CRITICAL RULES:
1. ALWAYS generate complete iFlow file contents BEFORE calling build_and_update_iflow,
   even if the user did not ask for modifications.
   NEVER call the tool without a 'files' dict.

2. Generate files in this JSON format first:
   {{
     "src/main/resources/scenarioflows/integrationflow/IFLOW_ID.iflw": "<?xml version=...",
     "META-INF/MANIFEST.MF": "Manifest-Version: 1.0...",
     ".project": "<?xml version=...",
     "src/main/resources/metadata.prop": "...",
     "src/main/resources/parameters.prop": "...",
     "src/main/resources/parameters.propdef": "..."
   }}
3. THEN call build_and_update_iflow with iflow_id (string), files (the dict you just generated), and autoDeploy=true.
4. The files parameter is REQUIRED - do not call build_and_update_iflow without providing the complete files dict.
5. Generate realistic SAP CPI iFlow XML with proper namespaces and routing logic.
6. For HTTPS endpoints, use HTTPServer adapter.
7. For XML mapping, include proper groovy/XSLT scripts.
8. For S/4HANA OData calls, use ODataV2 adapter with correct endpoint.
9. Wait for tool response before confirming deployment success.
10. Only call build_and_update_iflow when the user requests:
- create new iflow
- change iflow
- update mapping
- add adapter
- change logic
11. Report final status to user.
"""
                ),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.agent_tools,
            prompt=worker_prompt,
        )

        self.worker_agent = AgentExecutor(
            agent=agent,
            tools=self.agent_tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    # ------------------------------------------------------
    async def process_query(self, query: str):
        logger.info(f"[LLM] User Query: {query}")

        if not self.worker_agent:
            raise RuntimeError("Worker agent not initialized")

        worker_out = await self.worker_agent.ainvoke({"input": query})
        answer = worker_out.get("output", worker_out)
        return answer

    # ------------------------------------------------------
    async def chat_loop(self):
        print("\nMCP Integration Suite Chatbot Ready. Type 'quit' to exit.\n")

        while True:
            user_input = await asyncio.to_thread(input, "Query: ")
            if user_input.lower() in ("quit", "exit"):
                break

            try:
                answer = await self.process_query(user_input)
                print("\n" + str(answer) + "\n")
            except Exception as e:
                print("ERROR:", e)
                logger.exception("ERROR")

    # ------------------------------------------------------
    async def cleanup(self):
        await self.exit_stack.aclose()


# ==========================================================
# MAIN EXECUTION
# ==========================================================
async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <server.js|server.py>")
        sys.exit(1)

    server = sys.argv[1]

    client = MCPClient()
    try:
        await client.connect_to_server(server)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
