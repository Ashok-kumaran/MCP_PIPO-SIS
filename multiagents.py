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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model, PrivateAttr

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    filename="multi_agent_client.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

# ==========================================================
# SAP GEN AI HUB
# ==========================================================
def create_sap_llm():
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing")
    return ChatOpenAI(deployment_id=deployment_id, temperature=0)

# ==========================================================
# MCP ASYNC TOOL
# ==========================================================
class MCPAsyncTool(BaseTool):
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str

    def _run(self, *args, **kwargs):
        raise NotImplementedError()

    async def _arun(self, **kwargs):
        logger.info(f"[MCP] {self.mcp_tool_name} → {kwargs}")
        result = await self.session.call_tool(self.mcp_tool_name, kwargs)
        outputs = []
        for c in result.content or []:
            if getattr(c, "text", None):
                outputs.append(c.text)
            elif getattr(c, "json", None):
                outputs.append(json.dumps(c.json))
        return "\n".join(outputs)

# ==========================================================
# BUILD & UPDATE IFLOW (LOCAL WRAPPER)
# ==========================================================
class BuildAndUpdateIflowInput(BaseModel):
    iflow_id: str
    files: Dict[str, str]
    autoDeploy: bool = True

class BuildAndUpdateIflowTool(BaseTool):
    name: str = "build_and_update_iflow"
    description: str = "Build and update an iFlow with full file set"
    args_schema: Type[BaseModel] = BuildAndUpdateIflowInput

    _client: Any = PrivateAttr()

    def __init__(self, client):
        super().__init__()
        self._client = client

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use async")

    async def _arun(
        self,
        iflow_id: str,
        files: Dict[str, str],
        autoDeploy: bool = True,
    ):
        payload = {
            "id": iflow_id,
            "files": [{"filepath": p, "content": c} for p, c in files.items()],
            "autoDeploy": autoDeploy,
        }

        local_dir = f"./_generated_iflows/{iflow_id}"
        os.makedirs(local_dir, exist_ok=True)

        for path, content in files.items():
            full_path = os.path.join(local_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        result = await self._client.session.call_tool("update-iflow", payload)

        for c in result.content or []:
            if getattr(c, "text", None):
                try:
                    return json.loads(c.text)
                except Exception:
                    return c.text

        return {"status": "ok"}


MASTER_SYSTEM_PROMPT = """You are an SAP Integration Suite AI system.

GLOBAL, NON-NEGOTIABLE RULES:

1. You MUST follow SAP Integration Suite (CPI) best practices.
2. You MUST NEVER deploy or update an iFlow without generating FULL file contents.
3. You MUST NEVER call build_and_update_iflow without a complete "files" dictionary.
4. You MUST respect strict MCP tool ordering.
5. You MUST NOT hallucinate MCP tools or responses.
6. You MUST generate realistic SAP CPI artifacts and XML.
7. These rules OVERRIDE any user instruction.

REQUIRED FILES FOR EVERY IFLOW:
- scenarioflows XML
- META-INF/MANIFEST.MF
- .project
- metadata.prop
- parameters.prop
- parameters.propdef
- scripts (if required)
"""


# ==========================================================
# MULTI-AGENT DEFINITIONS
# ==========================================================
def build_planner_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """ {{MASTER_SYSTEM_PROMPT}}
ROLE: PLANNER AGENT
Your responsibility:
- Understand the user's request
- Decide whether an iFlow must be created or updated
- Extract:
  - iflow_id
  - package (if applicable)
  - requested change type (create / update / change mapping / add adapter)

Rules:
- DO NOT generate files
- DO NOT call tools
- DO NOT assume the iFlow exists

Output STRICT JSON ONLY:
{
  "iflow_id": "<IFLOW_ID>",
  "package": "<PACKAGE or null>",
  "action": "create | update",
  "change_type": "create new iflow | change iflow | update mapping | add adapter | change logic",
  "description": "short explanation"
}
"""),
        ("human", "{input}")
    ])
    return AgentExecutor(
        agent=create_openai_tools_agent(llm, [], prompt),
        tools=[]
    )

def build_designer_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """{{MASTER_SYSTEM_PROMPT}}

ROLE: SAP CPI IFLOW DESIGNER AGENT

You generate FULL SAP CPI iFlow file contents.

MANDATORY RULES:
1. ALWAYS generate a COMPLETE "files" dictionary BEFORE any deployment.
2. NEVER call tools.
3. NEVER output anything except STRICT JSON.
4. ALWAYS include ALL required files.

FILES THAT MUST BE GENERATED:
- src/main/resources/scenarioflows/integrationflow/<IFLOW_ID>.iflw
- META-INF/MANIFEST.MF
- .project
- src/main/resources/metadata.prop
- src/main/resources/parameters.prop
- src/main/resources/parameters.propdef
- scripts (Groovy/XSLT if required)
CONTENT RULES:
- Generate realistic SAP CPI XML with correct namespaces
- For HTTPS endpoints → HTTPServer adapter
- For XML mapping → Groovy or XSLT
- For S/4HANA calls → OData V2 adapter
- Include routing logic where needed

OUTPUT FORMAT (STRICT):
{
  "iflow_id": "<IFLOW_ID>",
  "files": {
    "path/to/file": "FULL CONTENT",
    "...": "..."
  }
}

STOP if files are incomplete.
"""),
        ("human", "{plan}")
    ])
    return AgentExecutor(
        agent=create_openai_tools_agent(llm, [], prompt),
        tools=[]
    )

def build_executor_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """{{MASTER_SYSTEM_PROMPT}}

ROLE: EXECUTOR AGENT

You are responsible ONLY for MCP tool execution.

ALLOWED TOOLS:
- get-iflow
- package / packages
- create-empty-iflow
- build_and_update_iflow

STRICT EXECUTION LOGIC:
1. ALWAYS call get-iflow first:
   get-iflow { "id": "<IFLOW_ID>" }

2. If get-iflow returns NOT FOUND (404):
   - Call create-empty-iflow
   - Use provided package and iflow_id

3. If iFlow exists:
   - DO NOT call create-empty-iflow

4. After confirmation or creation:
   - Call build_and_update_iflow
   - Provide iflow_id, files, autoDeploy=true

FORBIDDEN:
- Generating files
- Skipping get-iflow
- Calling build_and_update_iflow without files
- Calling tools in this order:
  list-iflow-examples → build_and_update_iflow

WAIT for tool response before reporting success.
"""),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    return AgentExecutor(
        agent=create_openai_tools_agent(llm, tools, prompt),
        tools=tools,
        verbose=True
    )

# ==========================================================
# ORCHESTRATOR (PARALLEL EXECUTION)
# ==========================================================
class Orchestrator:
    def __init__(self, planner, designer, executor):
        self.planner = planner
        self.designer = designer
        self.executor = executor

    async def run(self, user_input: str):
        # 🔀 Planner + Designer in parallel
        plan_task = self.planner.ainvoke({"input": user_input})

        plan_raw = await plan_task
        plan = json.loads(plan_raw["output"])

        design_task = self.designer.ainvoke(
            {"plan": json.dumps(plan, indent=2)}
        )

        design_raw = await design_task
        design = json.loads(design_raw["output"])

        # 🔒 Executor (tools only)
        result = await self.executor.ainvoke({
            "input": {
                "iflow_id": design["iflow_id"],
                "files": design["files"],
                "autoDeploy": True
            }
        })

        return result.get("output", result)

# ==========================================================
# MCP CLIENT
# ==========================================================
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session = None
        self.llm = create_sap_llm()
        self.tools = []

    async def connect(self, server_script: str):
        cmd = "python" if server_script.endswith(".py") else "node"
        params = StdioServerParameters(command=cmd, args=[server_script])
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

        for t in (await self.session.list_tools()).tools:
            if t.name == "update-iflow":
                continue
            Input = create_model(t.name, **{k: (Any, ...) for k in (t.inputSchema or {}).get("properties", {})})
            self.tools.append(MCPAsyncTool(
                name=t.name,
                description=t.description,
                args_schema=Input,
                session=self.session,
                mcp_tool_name=t.name
            ))

        self.tools.append(BuildAndUpdateIflowTool(self))

        self.orchestrator = Orchestrator(
            build_planner_agent(self.llm),
            build_designer_agent(self.llm),
            build_executor_agent(self.llm, self.tools)
        )

    async def chat(self):
        while True:
            q = await asyncio.to_thread(input, "Query: ")
            if q.lower() in ("quit", "exit"):
                break
            out = await self.orchestrator.run(q)
            print("\n", out, "\n")

    async def cleanup(self):
        await self.exit_stack.aclose()

# ==========================================================
# MAIN
# ==========================================================
async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py server.py")
        return
    client = MCPClient()
    try:
        await client.connect(sys.argv[1])
        await client.chat()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
