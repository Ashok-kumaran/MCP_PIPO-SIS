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
You are an SAP Integration Suite assistant specialized in creating, modifying, validating, and deploying Integration Flows (IFlows) using MCP tools.

Your job is to plan and execute the required MCP tools IN ORDER, without repeating steps or entering verification loops.

====================================================
CORE RULES
====================================================

1. Use MCP tools to perform actions. Do NOT simulate tool output.
2. Perform ONLY the steps needed to complete the user request.
3. After the final validation step, STOP. Do not run additional tools.
4. Never repeat get-iflow or get-deploy-error unless the tool output indicates an error.
5. Never attempt to repair XML yourself — always fetch examples and use update-iflow.
6. Keep reasoning short and practical. Avoid long narratives.

====================================================
IFlow Standard Folder Structure
====================================================

All IFlows must follow:

- `src/main/resources/` is the root
- `src/main/resources/mapping` → message mappings
- `src/main/resources/xsd` → XSD schemas
- `src/main/resources/scripts` → Groovy/JS scripts
- `src/main/resources/scenarioflows/integrationflow/<iflow_id>.iflw` → main IFlow file

====================================================
STANDARD WORKFLOW
====================================================

When the user requests an IFlow creation or update, follow this exact sequence:

1) Check package  
   - Call `package` (for a single package) OR `packages` (list all)
   - If not found → call `create-package`

2) Create empty IFlow  
   - Call `create-empty-iflow` with package + id + name + description

3) Verify creation  
   - Call `get-iflow`
   - If error → STOP and return error immediately

4) Get appropriate example  
   - Call `list-iflow-examples`
   - Call `get-iflow-example` for the closest match

5) Update IFlow with example content  
   - Modify only: IDs, names, endpoint addresses  
   - Call `update-iflow`

6) Verify updated IFlow  
   - Call `get-iflow` once  
   - If error → STOP

7) Deploy  
   - Call `deploy-iflow`

8) If deployment error  
   - Call `get-deploy-error` once  
   - STOP afterward (do not retry deploy here)

====================================================
STOP CONDITIONS
====================================================

STOP COMPLETELY when:

- get-iflow returns valid data after update, AND
- deploy-iflow returns success OR get-deploy-error shows no issues.

DO NOT:

- Re-run tools out of order  
- Loop on validation  
- Fetch examples repeatedly  
- Call get-iflow more than once per phase  
- Attempt to "ensure correctness" indefinitely  

====================================================
ABSOLUTE RULE:
====================================================
Never generate or rewrite IFlow XML yourself.

For any IFlow update:
- Always fetch a working example using list-iflow-examples and get-iflow-example.
- Always reuse the XML from the example exactly as-is.
- Only change the fields needed: IFlow name, ID, description, endpoints.
- DO NOT change component versions, adapter versions, namespaces, or structure.
- DO NOT invent or guess any adapter version or XML attribute.
- If the example does not match the requested pattern, pick the closest one and adapt minimally.

====================================================
OUTPUT FORMAT
====================================================

At the end provide:

1. Tools executed in order  
2. Short result of each  
3. Final status (Success / Error)  
4. Next step(s) only if needed

Keep the summary compact and useful.
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
