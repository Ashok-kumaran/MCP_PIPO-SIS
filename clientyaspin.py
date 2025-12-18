import asyncio
import sys
import os
import json
import logging
from typing import Optional, Any, Type, Union, Literal, Dict, List
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
import re


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

def build_pydantic_model(name: str, schema: Dict, root: Optional[Dict] = None) -> Any:
    """
    Recursively converts MCP JSON schema into a Pydantic model.
    Supports objects, arrays, enums, oneOf, anyOf, $ref.
    """

    if root is None:
        root = schema

    # ------------------------------------------------------------
    # $ref support
    # ------------------------------------------------------------
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return Any
        path = ref[2:].split("/")
        target = root
        for part in path:
            target = target.get(part, {})
        return build_pydantic_model(name, target, root)

    # ------------------------------------------------------------
    # enums
    # ------------------------------------------------------------
    if "enum" in schema:
        from typing import Literal
        values = tuple(schema["enum"])
        return Literal[values]

    # ------------------------------------------------------------
    # oneOf / anyOf
    # ------------------------------------------------------------
    if "oneOf" in schema:
        subs = [build_pydantic_model(f"{name}_oneOf_{i}", s, root) for i, s in enumerate(schema["oneOf"])]
        return Union[tuple(subs)]

    if "anyOf" in schema:
        subs = [build_pydantic_model(f"{name}_anyOf_{i}", s, root) for i, s in enumerate(schema["anyOf"])]
        return Union[tuple(subs)]

    # ------------------------------------------------------------
    # Object
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Array
    # ------------------------------------------------------------
    if schema.get("type") == "array":
        item_schema = schema.get("items", {}) or {}
        item_type = build_pydantic_model(f"{name}_item", item_schema, root)
        return List[item_type]

    # ------------------------------------------------------------
    # Primitive
    # ------------------------------------------------------------
    if schema.get("type") == "string":
        return str
    if schema.get("type") == "integer":
        return int
    if schema.get("type") == "number":
        return float
    if schema.get("type") == "boolean":
        return bool

    # ------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------
    return Any

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

        print(f"Running MCP tool: {self.mcp_tool_name}")
        result = await self.session.call_tool(self.mcp_tool_name, kwargs)
        print("Tool executed successfully")

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

        print("Connecting to MCP server...")
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        print("Connected!")
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

            # Build full recursive model
            InputModel = build_pydantic_model(f"{tdef.name}_Input", schema)

            tool = MCPAsyncTool(
                name=tdef.name,
                description=tdef.description or "",
                args_schema=InputModel,
                session=self.session,
                mcp_tool_name=tdef.name,
            )

            self.agent_tools.append(tool)

        logger.info(f"[MCP] Loaded {len(self.agent_tools)} tools with recursive schema support.")

    # ======================================================
    async def _build_worker_agent(self):

        worker_prompt = ChatPromptTemplate.from_messages([
            ("system",

"""
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

When you need help with any integration scenario, I'll guide you through these tools and help you create effective solutions following SAP Integration Suite best practices.

## Getting Help
If you need assistance or are unsure how to proceed, you have a few options:
1.  **Search the Documentation:** Use the `search-docs` tool from the `mcp-integration-suite` server to find relevant information. The documentation covers both general SAP Integration Suite topics and specific TPM functionalities.
2.  **Ask for Help:** If you can't find what you're looking for in the documentation, feel free to ask me directly. I can guide you on how to use the available tools to achieve your goals.

Remember to always think step-by-step and use the tools available to you effectively.
Do not explore all packages unless a package name is unknown.
"""

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
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=20,
            early_stopping_method="force",
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

        print("Processing query...")
        worker_out = await self.worker_agent.ainvoke(
            {"input": query},
            callbacks=[log_agent_step],
        )
        print("Query processed")

        raw_answer = str(worker_out.get("output", ""))

        # ---- ADD THIS BLOCK ----
        summary = await self.llm.ainvoke(f"""
        Here is the MCP tool output from the previous steps:

        {raw_answer}

        Write a clear natural-language summary for the user.
        Use simple sentences.
        Start with “Success:” or “Failed:” depending on what happened.

        Explain:
        - what you did
        - which tools were used
        - what the results mean
        - what the user can do next

        Do NOT return JSON. Use natural language only.
        """)

        return str(summary.content) if hasattr(summary, "content") else str(summary)

# ------------------------


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
