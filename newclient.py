import asyncio
import sys
import os
import json
import logging
from typing import Optional, Any, Type, Union, Dict, List, get_args, get_origin, Literal
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

from pydantic import BaseModel, create_model
import re
from yaspin import yaspin

from langchain.memory import ConversationBufferWindowMemory



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
# LLM
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
# JSON SCHEMA → PYDANTIC MODEL
# ==========================================================
def build_pydantic_model(name: str, schema: Dict, root: Dict = None) -> Any:
    if root is None:
        root = schema

    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return Any
        path = ref[2:].split("/")
        target = root
        for part in path:
            target = target.get(part, {})
        return build_pydantic_model(name, target, root)

    if "enum" in schema:
        return Literal[tuple(schema["enum"])]

    if "oneOf" in schema:
        subs = [build_pydantic_model(f"{name}_oneOf_{i}", s, root) for i, s in enumerate(schema["oneOf"])]
        return Union[tuple(subs)]

    if "anyOf" in schema:
        subs = [build_pydantic_model(f"{name}_anyOf_{i}", s, root) for i, s in enumerate(schema["anyOf"])]
        return Union[tuple(subs)]

    if schema.get("type") == "object":
        props = schema.get("properties", {}) or {}
        required = schema.get("required", [])
        fields = {}
        for key, subschema in props.items():
            field_type = build_pydantic_model(f"{name}_{key}", subschema, root)
            default = ... if key in required else None
            fields[key] = (field_type, default)
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        return create_model(safe_name, **fields)

    if schema.get("type") == "array":
        item_schema = schema.get("items", {}) or {}
        item_type = build_pydantic_model(f"{name}_item", item_schema, root)
        return List[item_type]

    if schema.get("type") == "string":
        return str
    if schema.get("type") == "integer":
        return int
    if schema.get("type") == "number":
        return float
    if schema.get("type") == "boolean":
        return bool

    return Any


# ==========================================================
# MCP TOOL WRAPPER
# ==========================================================
class MCPAsyncTool(BaseTool):
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use async.")

    async def _arun(self, *args, **kwargs) -> str:
        logger.info(f"[MCP-TOOL] Executing tool={self.mcp_tool_name} args={kwargs}")

        with yaspin(text=f"Running MCP tool: {self.mcp_tool_name}", color="magenta") as sp:
            result = await self.session.call_tool(self.mcp_tool_name, kwargs)
            sp.ok("✔")

        outputs = []
        for c in result.content or []:
            if getattr(c, "text", None):
                outputs.append(c.text)
            elif getattr(c, "json", None):
                outputs.append(json.dumps(c.json, ensure_ascii=False))
            else:
                outputs.append(str(c))

        return "\n".join(outputs)


# ==========================================================
# MCP CLIENT
# ==========================================================
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

        self.llm = create_sap_llm()

        self.agent_tools: List[BaseTool] = []
        self.worker_agent: Optional[AgentExecutor] = None

        # NEW AGENTS
        self.chat_agent = None
        self.router = None

        # Conversation-only memory
        # Keep only the last 4 conversational turns
        self.memory = ConversationBufferWindowMemory(
            k=4,
            memory_key="history",
            return_messages=True
        )


    # ------------------------------------------------------
    async def connect_to_server(self, server_script: str):

        cmd = "python" if server_script.endswith(".py") else "node"

        params = StdioServerParameters(command=cmd, args=[server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport

        with yaspin(text="Connecting to MCP server...", color="cyan") as sp:
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            sp.ok("✔")

        await self.session.initialize()

        await self._build_agents()

        tools = [t.name for t in (await self.session.list_tools()).tools]
        print("\nMCP Connected! Tools:", tools)

    # ======================================================
    async def _build_agents(self):
        await self._build_agent_tools()
        await self._build_worker_agent()
        await self._build_chat_agent()
        await self._build_router()

    # ======================================================
    async def _build_agent_tools(self):
        tool_list = await self.session.list_tools()

        for tdef in tool_list.tools:
            schema = tdef.inputSchema or {}
            InputModel = build_pydantic_model(f"{tdef.name}_Input", schema)

            tool = MCPAsyncTool(
                name=tdef.name,
                description=tdef.description or "",
                args_schema=InputModel,
                session=self.session,
                mcp_tool_name=tdef.name,
            )
            self.agent_tools.append(tool)

    # ======================================================
    async def _build_worker_agent(self):

        worker_prompt = ChatPromptTemplate.from_messages([
            ("system", """ 
             You are a specialized assistant for SAP Integration Suite, with a focus on designing, creating, and modifying integration artifacts. You have access to a set of tools that help you interact with SAP Integration Suite.

## Available Capabilities and Components

The SAP Integration Suite provides the following key capabilities:

1. **Cloud Integration** - For end-to-end process integration across cloud and on-premise applications
2. **API Management** - For publishing, promoting, and securing APIs
3. **Event Mesh** - For publishing and consuming business events across applications
4. **Integration Advisor** - For specifying B2B integration content
5. **Trading Partner Management** - For managing B2B relationships
6. **Open Connectors** - For connecting to 150+ non-SAP applications
7. **Integration Assessment** - For defining integration landscapes
8. **Other capabilities** including OData Provisioning, Migration Assessment, etc.

## Artifacts within a Package

An integration package can contain several types of artifacts:

1. **Integration Flows (IFlows)** - The main artifact type for defining integration scenarios and message processing ✅ IFlow IDs are unique over packages. So if an iflow ID is provided you don't need to fetch packages. You only need a package for creating an iflow**(Supported)**
2. **Message Mappings** - Define how to transform message formats between sender and receiver ✅ **(Supported)**
3. **Script Collections** - Reusable scripts that can be referenced in integration scenarios ❌ **(Not currently supported)**
4. **Data Types** - XML schemas (XSDs) that define the structure of messages ❌ **(Not currently supported, but can be included within IFlows)**
5. **Message Types** - Definitions based on data types that describe message formats ❌ **(Not currently supported)**
6. **packages** - Abstraction layer to group other artifacts✅ **(Supported)**
**Note:** Currently, only IFlows, packages and Message Mappings are directly supported by the tools. Other artifacts may be included as part of an IFlow's resources.

## Available Tools and Functions

You can access the following tools:

1. **Package Management**
   - `packages` - Get all integration packages
   - `package` - Get content of an integration package by name
   - `create-package` - Create a new integration package

2. **Integration Flow (IFlow) Management**
   - `get-iflow` - Get the data of an IFlow and contained resources
   - `create-empty-iflow` - Create an empty IFlow
   - `update-iflow` - Update or create files/content of an IFlow
   - `get-iflow-endpoints` - Get endpoints of IFlow and its URLs/Protocols
   - `iflow-image` - Get the IFlow logic shown as a diagram
   - `deploy-iflow` - Deploy an IFlow
   - `get-iflow-configurations` - Get all configurations of an IFlow
   - `get-all-iflows` - Get a list of all available IFlows in a Package

3. **Message Mapping Management**
   - `get-messagemapping` - Get data of a Message Mapping
   - `update-message-mapping` - Update Message Mapping files/content
   - `deploy-message-mapping` - Deploy a message-mapping
   - `create-empty-mapping` - Create an empty message mapping
   - `get-all-messagemappings` - Get all available message mappings

4. **Examples and Discovery**
   - `discover-packages` - Get information about Packages from discover center
   - `list-iflow-examples` - Get a list of available IFlow examples
   - `get-iflow-example` - Get an existing IFlow as an example
   - `list-mapping-examples` - Get all available message mapping examples
   - `get-mapping-example` - Get an example provided by list-mapping-examples
   - `create-mapping-testiflow` - Creates an IFlow called if_echo_mapping for testing

5. **Deployment and Monitoring**
   - `get-deploy-error` - Get deployment error information
   - `get-messages` - Get message from message monitoring
   - `count-messages` - Count messages from the message monitoring. Is useful for making summaries etc.
   - `send-http-message` - Send an HTTP request to integration suite

## Key IFlow Components

When working with IFlows, you'll interact with these components:

1. **Adapters** (for connectivity):
   - Sender adapters: HTTPS, AMQP, AS2, FTP, SFTP, Mail, etc.
   - Receiver adapters: HTTP, JDBC, OData, SOAP, AS4, etc.

2. **Message Processing**:
   - Transformations: Mapping, Content Modifier, Converter
   - Routing: Router, Multicast, Splitter, Join
   - External Calls: Request-Reply, Content Enricher
   - Security: Encryptor, Decryptor, Signer, Verifier
   - Storage: Data Store Operations, Persist Message

## Important Guidelines

1. **ALWAYS examine examples first** when developing solutions. Use `list-iflow-examples` and `get-iflow-example` to study existing patterns before creating new ones.

2. **Start with packages and IFlows**. First check existing packages with `packages`, then either use an existing package or create a new one with `create-package`, then create or modify IFlows.

3. **Folder structure matters** in IFlows:
   - `src/main/resources/` is the root
   - `src/main/resources/mapping` contains message mappings
   - `src/main/resources/xsd` contains XSD files
   - `src/main/resources/scripts` contains scripts
   - `src/main/resources/scenarioflows/integrationflow/<iflow id>.iflw` contains the IFlow

4. **Use a step-by-step approach**:
   - Analyze requirements
   - Check examples
   - Create/modify package
   - Create/modify IFlow
   - Deploy and test
   - Check for errors

5. **For errors**, use `get-deploy-error` to troubleshoot deployment issues or `get-messages` to investigate runtime issues.

6. **Be conservative with changes** to existing IFlows - only modify what's needed and preserve the rest.

7. **Message mappings typically live within IFlows**. While standalone message mappings exist (`create-empty-mapping`), in most scenarios message mappings are developed directly within the IFlow that uses them. Only create standalone mappings when specifically required.

8. **For testing mappings**, use `create-mapping-testiflow` to create a test IFlow.

Remember to always think step-by-step and use the tools available to you effectively.
Do not explore all packages unless a package name is unknown."""),
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
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=20,
        )

    # ======================================================
    async def _build_chat_agent(self):
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful and friendly chatbot. "
             "Do NOT call MCP tools. Just talk naturally."),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ])

        self.chat_agent = (chat_prompt | self.llm | StrOutputParser())

    # ======================================================
    async def _build_router(self):
        router_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a router. Decide if the user request requires "
             "MCP tool usage.\n\n"
             "If the user wants to create, update, fetch, deploy, packages, iflows, ID, SAP, Integration suite or "
             "manipulate SAP Integration Suite artifacts → respond ONLY 'TOOL'.\n\n"
             "If greeting, general questions, or explanation → respond ONLY 'CHAT'."),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ])

        self.router = router_prompt | self.llm | StrOutputParser()

    # ======================================================
    async def process_query(self, query: str):
        history = self.memory.load_memory_variables({})["history"]

        with yaspin(text="Processing query...", color="yellow") as sp:
            # 1️⃣ Decide CHAT vs TOOL
            decision = await self.router.ainvoke({
                "input": query,
                "history": history
            })
            decision = decision.strip().upper()

            logger.info(f"[ROUTER] decision={decision}")

            # -----------------------------------
            # CHAT MODE
            # -----------------------------------
            if decision == "CHAT":
                response = await self.chat_agent.ainvoke({
                    "input": query,
                    "history": history
                })
                sp.ok("✔")
                self.memory.save_context({"input": query}, {"output": response})
                return response

            # -----------------------------------
            # TOOL MODE
            # -----------------------------------
            worker_out = await self.worker_agent.ainvoke({"input": query})
            raw_answer = worker_out.get("output", worker_out)

            summary = await self.llm.ainvoke(f"""
            Here is the MCP tool output:

            {raw_answer}

            Create a clear, detailed summary.
            Provide in formative response to the user based on the above output.
            Keep it concise and to the point.
            Do NOT return JSON.
            """)

            final_answer = str(summary.content)

            sp.ok("✔")
            self.memory.save_context({"input": query}, {"output": final_answer})
            return final_answer

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

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
