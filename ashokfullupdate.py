import asyncio
import sys
import os
import json
import logging
from typing import Optional, Any, Type, Union, Dict, List
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler

from pydantic import create_model, BaseModel
import re
from yaspin import yaspin


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
def create_sap_llm() -> ChatOpenAI:
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
    """
    Recursively converts MCP JSON schema into a Pydantic model.
    Supports objects, arrays, enums, oneOf, anyOf, $ref.
    Handles both:
      - { "type": "object", ... }
      - { "schema": { "type": "object", ... } }
    """
    if root is None:
        root = schema

    # Some MCP servers wrap schema as {"schema": {...}}
    if "type" not in schema and "schema" in schema and isinstance(schema["schema"], dict):
        schema = schema["schema"]

    # $ref support
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return Any
        path = ref[2:].split("/")
        target: Any = root
        for part in path:
            if isinstance(target, dict):
                target = target.get(part, {})
            else:
                target = {}
        return build_pydantic_model(name, target, root)

    # enums
    if "enum" in schema:
        from typing import Literal

        values = tuple(schema["enum"])
        return Literal[values]

    # oneOf / anyOf
    if "oneOf" in schema:
        subs = []
        for i, s in enumerate(schema["oneOf"]):
            sub_name = name + "_oneOf_" + str(i)
            subs.append(build_pydantic_model(sub_name, s, root))
        return Union[tuple(subs)]

    if "anyOf" in schema:
        subs = []
        for i, s in enumerate(schema["anyOf"]):
            sub_name = name + "_anyOf_" + str(i)
            subs.append(build_pydantic_model(sub_name, s, root))
        return Union[tuple(subs)]

    # object
    if schema.get("type") == "object":
        props = schema.get("properties", {}) or {}
        required = schema.get("required", [])

        fields: Dict[str, Any] = {}
        for key, subschema in props.items():
            field_name = name + "_" + key
            field_type = build_pydantic_model(field_name, subschema, root)
            default = ... if key in required else None
            fields[key] = (field_type, default)

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        return create_model(safe_name, **fields)

    # array
    if schema.get("type") == "array":
        item_schema = schema.get("items", {}) or {}
        item_name = name + "_item"
        item_type = build_pydantic_model(item_name, item_schema, root)
        return List[item_type]

    # primitive
    if schema.get("type") == "string":
        return str
    if schema.get("type") == "integer":
        return int
    if schema.get("type") == "number":
        return float
    if schema.get("type") == "boolean":
        return bool

    # fallback
    return Any


# ==========================================================
# ASYNC MCP TOOL WRAPPER
# ==========================================================
class MCPAsyncTool(BaseTool):
    """
    LangChain tool that calls an MCP tool asynchronously.
    Enhanced with better error handling and result formatting.
    """
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Sync run is not supported; use async.")

    async def _arun(self, *args, **kwargs) -> str:
        logger.info(
            "[MCP-TOOL] Executing → %s | Args = %s",
            self.mcp_tool_name,
            kwargs,
        )

        try:
            with yaspin(
                text="Running MCP tool: " + self.mcp_tool_name,
                color="magenta",
            ) as sp:
                result = await self.session.call_tool(self.mcp_tool_name, kwargs)
                sp.ok("✔")
        except Exception as e:
            error_msg = "ERROR: Tool " + self.mcp_tool_name + " failed with: " + str(e)
            logger.error("[MCP-TOOL] %s", error_msg)
            return error_msg

        if not result.content:
            warning = "WARNING: Tool " + self.mcp_tool_name + " returned empty content"
            logger.warning("[MCP-TOOL] %s", warning)
            return warning

        outputs: List[str] = []
        has_error = False

        for c in result.content:
            # Standard MCP content types: text, json, etc.
            c_type = getattr(c, "type", None)

            if c_type == "text" and getattr(c, "text", None) is not None:
                text_content = c.text
                if "error" in text_content.lower() or "failed" in text_content.lower():
                    has_error = True
                outputs.append(text_content)

            elif c_type == "json" and (
                getattr(c, "json", None) is not None
                or getattr(c, "data", None) is not None
            ):
                json_obj = getattr(c, "json", None) or getattr(c, "data", None)
                try:
                    json_content = json.dumps(json_obj, ensure_ascii=False, indent=2)
                except TypeError:
                    json_content = str(json_obj)
                outputs.append(json_content)

            elif hasattr(c, "error"):
                has_error = True
                outputs.append("ERROR: " + str(c.error))

            else:
                outputs.append(str(c))

        final_output = "\n".join(outputs)

        if has_error:
            logger.error(
                "[MCP-TOOL] %s returned errors: %s",
                self.mcp_tool_name,
                final_output,
            )
        else:
            logger.info(
                "[MCP-TOOL] %s completed successfully",
                self.mcp_tool_name,
            )

        return final_output


# ==========================================================
# CALLBACK HANDLER FOR INTERMEDIATE STEPS
# ==========================================================
class StepLogger(BaseCallbackHandler):
    """
    Collects tool usage and relevant events into an `intermediate_steps` list.
    """

    def __init__(self, collector: List[Dict[str, Any]]):
        self.collector = collector

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: Any,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        self.collector.append(
            {
                "event": "tool_start",
                "tool": tool_name,
                "tool_input": input_str,
            }
        )
        logger.info("[LLM] Selected Tool → %s | Args → %s", tool_name, input_str)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self.collector.append(
            {
                "event": "tool_end",
                "output": str(output),
            }
        )

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.collector.append(
            {
                "event": "llm_error",
                "error": str(error),
            }
        )


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
        self.stdio = None
        self.write = None

    # ------------------------------
    # Connection & tool discovery
    # ------------------------------
    async def connect_to_server(self, server_script: str):
        """Connect to MCP server and initialize tools"""
        if server_script.endswith(".py"):
            cmd = "python"
        elif server_script.endswith(".js"):
            cmd = "node"
        else:
            raise ValueError("Server script must be .py or .js")

        params = StdioServerParameters(command=cmd, args=[server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        # Expect stdio_transport to be (reader, writer)
        self.stdio, self.write = stdio_transport

        with yaspin(text="Connecting to MCP server...", color="cyan") as sp:
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            sp.ok("✔")

        await self.session.initialize()
        await self._build_agent_tools()
        await self._build_worker_agent()

        tools_resp = await self.session.list_tools()
        tools = [t.name for t in tools_resp.tools]
        print(
            "\n✅ MCP Connected! Available tools:",
            ", ".join(tools[:10]),
            "..." if len(tools) > 10 else "",
        )
        logger.info("[MCP] Connected with %d tools", len(tools))

    async def _build_agent_tools(self):
        """Build LangChain tools from MCP tool definitions"""
        assert self.session is not None
        tool_list = await self.session.list_tools()

        self.agent_tools = []

        for tdef in tool_list.tools:
            schema = tdef.inputSchema or {}

            # handle wrapper {"schema": {...}} if used
            if "type" not in schema and "schema" in schema and isinstance(schema["schema"], dict):
                schema = schema["schema"]

            input_model_name = tdef.name + "_Input"
            InputModel = build_pydantic_model(input_model_name, schema)

            tool = MCPAsyncTool(
                name=tdef.name,
                description=tdef.description or "",
                args_schema=InputModel,
                session=self.session,
                mcp_tool_name=tdef.name,
            )
            self.agent_tools.append(tool)

        logger.info("[MCP] Loaded %d tools", len(self.agent_tools))

    # ------------------------------
    # Agent construction
    # ------------------------------
    async def _build_worker_agent(self):
        """Build the worker agent with optimized system prompt"""
        worker_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._get_system_prompt()),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.agent_tools,
            prompt=worker_prompt,
        )

        # Increased max_iterations to avoid early "stopped due to max iterations"
        self.worker_agent = AgentExecutor(
            agent=agent,
            tools=self.agent_tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=50,
            early_stopping_method="force",
        )

    def _get_system_prompt(self) -> str:
        """
        Optimized, loop-safe system prompt (Option B).
        Designed to avoid infinite validation loops and excessive tool calls.
        """
        return """
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


# SAP Trading Partner Management (TPM) Tools - Start Here

You are a specialized assistant for SAP Trading Partner Management (TPM), designed to help you manage B2B relationships, agreements, and message guidelines. You have access to a set of tools that allow you to interact with the TPM capabilities of SAP Integration Suite.

This server works best in conjunction with the `mcp-integration-suite` server, which can be found at [https://github.com/1nbuc/mcp-integration-suite](https://github.com/1nbuc/mcp-integration-suite). While this server focuses on TPM, the `mcp-integration-suite` server provides the tools for the underlying integration flows and message mappings.

## Important Guidelines

1.  **ALWAYS examine existing data structures first.** Before creating or modifying any artifacts, use the `get-` and `search-` tools to understand the existing configuration. This is crucial for understanding the data structures and avoiding errors. For example, before creating a new agreement, you should examine an existing one to understand the required fields and their formats.
2.  **All artifacts use GUIDs as IDs**, except for type system IDs. For most artifacts, there is both an ID and a version ID, but the version ID is usually sufficient to uniquely identify the artifact. Only exception is Typing Systems which have no GUID Identifier
3.  **Use a step-by-step approach**:
    *   Analyze requirements.
    *   Check for existing examples of similar artifacts.
    *   Create/modify the necessary artifacts (e.g., Trading Partner, MIG, Agreement).
    *   Verify your changes.
4.  **Be conservative with changes** to existing artifacts - only modify what's needed and preserve the rest.
5. **Common configuration**
* A common configuration for partners is a trading partner having a System e.g. partner-orders-system-1. Then there are usually two data identifiers most of the time one in an idoc system and one in an EDI System like EANCOM or TRADACOMs. The idoc identifier referrs to the internal SAP Partner number and the EDI Identifier often is the GLN of the Partner. For configuration of an identifier don't use custom things unless told to. Check get-type-system-identifier-schemes for available schemes. For example GLN is often called GS1. In addition, partners using AS2 must have a signature verification config (create-signature-verify-config) which is used to identify incoming AS2 messages by an AS2 ID. Within the system of the partner there is usually only one Type system registred (most of the time some EDI Type system) and one or multiple communication channels. 

## Available Tools

### Trading Partner Management
*   `get-partner-metadata`: Get metadata for all trading partners.
*   `get-partner`: Get partner details by partner id.
*   `create-trading-partner`: Create a new trading partner.
*   `get-systems-of-partner`: Returns all systems of a trading partner by its ID.
*   `create-system`: Create a system for a trading partner.
*   `get-system-types`: Get available system types.
*   `create-identifier`: Create a partner Identifier.
*   `get-qualifiers-codelist`: Get codelist of a qualifier.
*   `create-communication`: Create a communication channel for a system of a trading partner.
*   `get-sender-adapters`: Get all sender adapters of trading partner systems.
*   `get-receiver-adapters`: Get all receiver adapters of trading partner systems.
*   `create-signature-verify-config`: Create Signature Verification configuration for a partner.
*   `activate-signature-verify-config`: Activate Signature Verification configuration for a partner.
*   `get-all-company-profile-metadata`: Get metadata for all company profiles.

### Agreement Management
*   `get-all-agreement-metadata`: Get metadata for all agreements.
*   `get-all-agreement-template-metadata`: Get metadata for all agreement templates.
*   `get-agreement-template`: Get all details for an agreement template.
*   `create-agreement-with-bound-template`: Create a new B2B agreement which is bound to a template.
*   `get-agreement-b2b-scenario`: Get the technical B2B scenario of an agreement.
*   `update-b2b-scenario`: Update an Agreement's B2B Scenario.
*   `trigger-agreement-activate-or-update-deployment`: Update or deploy an agreement.

### Message Implementation Guideline (MIG) Management
*   `get-all-mig-latest-metadata`: Get the latest metadata for all Message Implementation Guidelines (MIGs).
*   `get-mig-raw-by-id`: Get raw MIG content by its version ID.
*   `get-mig-nodes-xpath`: Get the Nodes of a MIG for a specified XPath.
*   `get-all-mig-fields`: Get a List of all fields of a MIG.
*   `get-mig-documentation-entry`: Get the documentation text for a id of a documentation within a mig.
*   `get-mig-proposal`: Get Proposal for a MIG.
*   `apply-mig-proposal`: Select fields based on MIG proposal.
*   `create-mig-draft-all-segments-selected`: Creates a draft MIG from a source version, with all segments and fields pre-selected.
*   `create-mig`: Create Message implementation guideline based on a type.
*   `change-mig-field-selection`: Change the selection of MIG fields.

### Mapping Guideline (MAG) Management
*   `get-all-mags-metadata`: Get an overview of available Mapping guidelines.
*   `create-mapping-guidelines`: Create a new mapping guidelines.
*   `test-mag-with-message`: Send a message against a mapping guideline and get the result.

### Monitoring
*   `search-interchanges`: Search for interchanges/TPM message monitoring based on filter criteria.
*   `get-interchange-payloads`: Get payload data list for a specific interchange.
*   `download-interchange-payload`: Download a specific payload by its ID.
*   `get-interchange-last-error`: Get last error details for a specific message/business document.

### Other
*   `get-type-systems`: Get available type systems.
*   `get-type-system-messages`: Get messages of a type system.
*   `get-type-system-message-full`: Get a message from a type system with all details including versions and revisions.
*   `create-custom-message`: Create a custom message in typesystem Customer_TS based on XSD.
*   `get-type-system-identifier-schemes`: Get the possible scheme for identifiers in a type system.
*   `get-all-business-process-roles`: Get all business process roles.
*   `get-all-business-processes`: Get all business processes.
*   `get-all-industry-classifications`: Get all industry classifications.
*   `get-all-product-classifications`: Get all product classifications.
*   `get-all-products`: Get all available products/types for a system e.g. SAP SuccessFactors etc.
*   `get-all-contries-or-regions`: Get all countries or regions.

## Getting Help

If you need assistance or are unsure how to proceed, you have a few options:

1.  **Search the Documentation:** Use the `search-docs` tool from the `mcp-integration-suite` server to find relevant information. The documentation covers both general SAP Integration Suite topics and specific TPM functionalities.
2.  **Ask for Help:** If you can't find what you're looking for in the documentation, feel free to ask me directly. I can guide you on how to use the available tools to achieve your goals.

When you need help with any TPM scenario, I'll guide you through these tools and help you create effective solutions following SAP best practices.



"""

    # ------------------------------
    # Query processing
    # ------------------------------
    async def process_query(self, query: str) -> str:
        """Process user query with enhanced error tracking"""
        logger.info("[LLM] User Query: %s", query)

        if not self.worker_agent:
            raise RuntimeError("Worker agent not initialized")

        intermediate_steps: List[Dict[str, Any]] = []
        step_logger = StepLogger(intermediate_steps)

        try:
            with yaspin(text="Processing query...", color="yellow") as sp:
                # IMPORTANT: pass callbacks directly, not via config
                worker_out = await self.worker_agent.ainvoke(
                    {"input": query},
                    callbacks=[step_logger],
                )
                sp.ok("✔")
        except Exception as e:
            logger.error("[LLM] Agent execution failed: %s", str(e))
            error_context = {
                "error": str(e),
                "steps_completed": len(intermediate_steps),
                "last_steps": intermediate_steps[-3:] if intermediate_steps else [],
            }
            return "❌ Error during execution:\n" + str(e) + "\n\nContext:\n" + json.dumps(
                error_context, indent=2
            )

        raw_answer = worker_out.get("output", worker_out)

        # Ensure we work with a string
        if isinstance(raw_answer, str):
            raw_text = raw_answer
        else:
            try:
                raw_text = json.dumps(raw_answer, ensure_ascii=False, indent=2)
            except TypeError:
                raw_text = str(raw_answer)

        # Analyze if there were errors
        text_lower = raw_text.lower()
        has_errors = any(
            [
                "error" in text_lower,
                "failed" in text_lower,
                "error while loading" in text_lower,
                "could not" in text_lower,
                "stopped due to max iterations" in text_lower,
            ]
        )

        # Build summary prompt
        summary_prompt = (
            "Analyze the following MCP tool execution results and provide a clear, actionable summary.\n\n"
            "Tool Execution Results:\n"
            + raw_text
            + "\n\nIntermediate Steps:\n"
            + (
                json.dumps(intermediate_steps[-5:], indent=2)
                if intermediate_steps
                else "No intermediate steps recorded"
            )
            + "\n\nErrors Detected: "
            + ("YES - Focus on error analysis" if has_errors else "NO - Operation appears successful")
            + "\n\n"
            "Provide a summary that includes:\n\n"
            "1. Operations Performed: List each tool that was executed and what it did\n"
            "2. Results: What did each tool return? Was it successful?\n"
            "3. Validation Status: Was the IFlow validated (e.g., via get-iflow)?\n"
            "4. Errors (if any):\n"
            "   - Exact error messages\n"
            "   - Which component failed\n"
            "   - Root cause analysis\n"
            "   - Why it failed\n"
            "5. Next Steps:\n"
            "   - If successful: What the user can do now\n"
            "   - If failed: Specific steps to fix the issue\n"
            "6. Recommendations: Best practices or warnings\n\n"
            "Format your response clearly with sections. Be specific and actionable.\n"
        )

        with yaspin(text="Generating summary...", color="cyan") as sp:
            # Use ainvoke instead of deprecated apredict
            summary_msg = await self.llm.ainvoke(summary_prompt)
            sp.ok("✔")

        # Handle ChatOpenAI output (BaseMessage or str)
        if hasattr(summary_msg, "content"):
            summary = summary_msg.content
        else:
            summary = str(summary_msg)

        logger.info("[LLM] Final Summary Generated")

        # Add visual indicator for errors
        if has_errors:
            summary = "⚠️ **ERRORS DETECTED**\n\n" + summary
        else:
            summary = "✅ **OPERATION COMPLETED**\n\n" + summary

        return summary

    # ------------------------------
    # Chat loop
    # ------------------------------
    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "=" * 60)
        print("SAP Integration Suite MCP Chatbot")
        print("=" * 60)
        print("Type your queries or 'quit' to exit\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n💬 Query: ")
                if user_input.lower().strip() in ("quit", "exit", "q"):
                    print("\n👋 Goodbye!")
                    break

                if not user_input.strip():
                    continue

                answer = await self.process_query(user_input)
                print("\n" + "─" * 60)
                print(answer)
                print("─" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print("\n❌ ERROR: " + str(e))
                logger.exception("Unexpected error in chat loop")

    # ------------------------------
    # Cleanup
    # ------------------------------
    async def cleanup(self):
        """Clean up resources"""
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
    except Exception as e:
        print("\n❌ Fatal error: " + str(e))
        logger.exception("Fatal error in main")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
