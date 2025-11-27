import asyncio
import sys
import os
import json
import logging
from typing import Optional, Any, Type
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import create_model, BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
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
        raise NotImplementedError("Sync run is not supported; use async.")

    async def _arun(self, *args, **kwargs) -> str:
        logger.info(f"[MCP-TOOL] Executing → {self.mcp_tool_name} | Args = {kwargs}")

        result = await self.session.call_tool(self.mcp_tool_name, kwargs)

        if not result.content:
            logger.info(f"[MCP-TOOL] {self.mcp_tool_name} returned EMPTY content")
            return ""

        outputs = []
        for c in result.content:
            if getattr(c, "text", None):
                outputs.append(c.text)
            elif getattr(c, "json", None):
                outputs.append(json.dumps(c.json, ensure_ascii=False))
            else:
                outputs.append(str(c))

        logger.info(f"[MCP-TOOL] Final Output ({self.mcp_tool_name}): {outputs}")

        return "\n".join(outputs)


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
        await self._build_worker_agent()

        tools = [t.name for t in (await self.session.list_tools()).tools]
        print("\nMCP Connected! Tools:", tools)
        logger.info(f"[MCP] Connected with tools: {tools}")

    # ======================================================
    async def _build_agent_tools(self):

        assert self.session is not None
        tool_list = await self.session.list_tools()

        for tdef in tool_list.tools:
            schema = tdef.inputSchema or {}
            props = schema.get("properties", {}) or {}
            required = set(schema.get("required", []))

            fields: dict[str, tuple[Any, Any]] = {}
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

        logger.info(f"[MCP] Loaded {len(self.agent_tools)} tools into LangChain.")

    # ======================================================
    async def _build_worker_agent(self):

        worker_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are the WORKER agent for SAP CPI iFlow creation.\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER call the same tool twice with identical arguments\n"
            "2. After seeing 'successfully updated' from update-iflow, STOP immediately\n"
            "3. Do NOT keep calling update-iflow repeatedly\n"
            "4. Use the exact BPMN content provided in the context\n"
            "5. Package IDs must be alphanumeric (no spaces)\n\n"

            "CORRECT update-iflow FORMAT:\n"
            "{{{{\n"
            '  \"id\": \"IFlowID\",\n'
            '  \"files\": [\n'
            "    {{{{\n"
            '      \"filepath\": \"src/main/resources/scenarioflows/integrationflow/IFlowID.iflw\",\n'
            '      \"content\": \"<xml goes here>\"\n'
            "    }}}}\n"
            "  ],\n"
            '  \"autoDeploy\": false\n'
            "}}}}\n\n"

            "WORKFLOW YOU MUST FOLLOW:\n"
            "Step 1 → Use `package` tool to check if package exists\n"
            "  • If 404 / Not Found → create with `create-package`\n"
            "  • If exists → continue\n\n"
            "Step 2 → Create empty iFlow using `create-empty-iflow`\n\n"
            "Step 3 → Update using `update-iflow` (ONLY ONCE!)\n"
            "  • Use provided BPMN only\n"
            "  • Stop immediately after success\n\n"
            "STOPPING RULE:\n"
            "Once tool output contains: 'successfully updated', STOP and do not call any more tools.\n\n"
            "IMPORTANT:\n"
            "- NEVER invent data\n"
            "- NEVER guess file paths\n"
            "- Use only tools\n"
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

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
            max_iterations=12,
            early_stopping_method="generate",
        )


    # ======================================================
    async def process_query(self, query: str):

        logger.info(f"[LLM] User Query: {query}")

        if not self.worker_agent:
            raise RuntimeError("Worker agent not initialized")

        # Callback for logging which tool LLM selects
        def log_agent_step(step):
            if isinstance(step, dict) and "tool" in step:
                logger.info(
                    f"[LLM] Selected Tool → {step['tool']} | Args → {step.get('tool_input')}"
                )
            return step

        worker_out = await self.worker_agent.ainvoke(
            {"input": query},
            callbacks=[log_agent_step],
        )

        answer = worker_out.get("output", worker_out)
        logger.info(f"[LLM] Final Answer: {answer}")

        return answer

    # ======================================================
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

    # ======================================================
    async def cleanup(self):
        await self.exit_stack.aclose()


# ==========================================================
# MAIN
# ==========================================================
async def main():
    if len(sys.argv) < 2:
        print("Usage: python pipo_client.py <server.js|server.py>")
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

