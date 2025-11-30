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
from pydantic import create_model, BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
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
def create_sap_llm():
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing in .env")

    return ChatOpenAI(
        deployment_id=deployment_id,
        temperature=0,
    )


def build_pydantic_model(name: str, schema: Dict, root: Dict = None) -> Any:
    """
    Recursively converts MCP JSON schema into a Pydantic model.
    Supports objects, arrays, enums, oneOf, anyOf, $ref.
    """
    if root is None:
        root = schema

    # $ref support
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return Any
        path = ref[2:].split("/")
        target = root
        for part in path:
            target = target.get(part, {})
        return build_pydantic_model(name, target, root)

    # enums
    if "enum" in schema:
        from typing import Literal
        values = tuple(schema["enum"])
        return Literal[values]

    # oneOf / anyOf - FIXED: No f-strings
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

    # Object - FIXED: No f-strings
    if schema.get("type") == "object":
        props = schema.get("properties", {}) or {}
        required = schema.get("required", [])

        fields = {}
        for key, subschema in props.items():
            field_name = name + "_" + key
            field_type = build_pydantic_model(field_name, subschema, root)
            default = ... if key in required else None
            fields[key] = (field_type, default)

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        return create_model(safe_name, **fields)

    # Array - FIXED: No f-strings
    if schema.get("type") == "array":
        item_schema = schema.get("items", {}) or {}
        item_name = name + "_item"
        item_type = build_pydantic_model(item_name, item_schema, root)
        return List[item_type]

    # Primitive
    if schema.get("type") == "string":
        return str
    if schema.get("type") == "integer":
        return int
    if schema.get("type") == "number":
        return float
    if schema.get("type") == "boolean":
        return bool

    # Fallback
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
        logger.info(f"[MCP-TOOL] Executing → {self.mcp_tool_name} | Args = {kwargs}")

        try:
            with yaspin(text=f"Running MCP tool: {self.mcp_tool_name}", color="magenta") as sp:
                result = await self.session.call_tool(self.mcp_tool_name, kwargs)
                sp.ok("✔")
        except Exception as e:
            error_msg = f"ERROR: Tool {self.mcp_tool_name} failed with: {str(e)}"
            logger.error(f"[MCP-TOOL] {error_msg}")
            return error_msg

        if not result.content:
            warning = f"WARNING: Tool {self.mcp_tool_name} returned empty content"
            logger.warning(f"[MCP-TOOL] {warning}")
            return warning

        outputs = []
        has_error = False
        
        for c in result.content:
            if getattr(c, "text", None):
                text_content = c.text
                if "error" in text_content.lower() or "failed" in text_content.lower():
                    has_error = True
                outputs.append(text_content)
            elif getattr(c, "json", None):
                json_content = json.dumps(c.json, ensure_ascii=False, indent=2)
                outputs.append(json_content)
            elif hasattr(c, "error"):
                has_error = True
                outputs.append(f"ERROR: {c.error}")
            else:
                outputs.append(str(c))

        final_output = "\n".join(outputs)
        
        if has_error:
            logger.error(f"[MCP-TOOL] {self.mcp_tool_name} returned errors: {final_output}")
        else:
            logger.info(f"[MCP-TOOL] {self.mcp_tool_name} completed successfully")

        return final_output


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
        self.stdio, self.write = stdio_transport

        with yaspin(text="Connecting to MCP server...", color="cyan") as sp:
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            sp.ok("✔")
        
        await self.session.initialize()
        await self._build_agent_tools()
        await self._build_worker_agent()

        tools = [t.name for t in (await self.session.list_tools()).tools]
        print("\n✅ MCP Connected! Available tools:", ", ".join(tools[:10]), "..." if len(tools) > 10 else "")
        logger.info(f"[MCP] Connected with {len(tools)} tools")

    async def _build_agent_tools(self):
        """Build LangChain tools from MCP tool definitions"""
        assert self.session is not None
        tool_list = await self.session.list_tools()

        for tdef in tool_list.tools:
            schema = tdef.inputSchema or {}
            # FIXED: Use string concatenation instead of f-string
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

        logger.info(f"[MCP] Loaded {len(self.agent_tools)} tools")

    async def _build_worker_agent(self):
        """Build the worker agent with enhanced system prompt"""
        worker_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
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
            max_iterations=10,
            early_stopping_method="force",
        )

    def _get_system_prompt(self) -> str:
        """Returns the enhanced system prompt with validation steps"""
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
OUTPUT FORMAT
====================================================

At the end provide:

1. Tools executed in order  
2. Short result of each  
3. Final status (Success / Error)  
4. Next step(s) only if needed

Keep the summary compact and useful.


"""

    async def process_query(self, query: str) -> str:
        """Process user query with enhanced error tracking"""
        logger.info(f"[LLM] User Query: {query}")

        if not self.worker_agent:
            raise RuntimeError("Worker agent not initialized")

        intermediate_steps = []
        
        def log_agent_step(step):
            if isinstance(step, dict):
                intermediate_steps.append(step)
                if "tool" in step:
                    tool_name = step.get("tool", "unknown")
                    tool_input = step.get("tool_input", {})
                    logger.info(f"[LLM] Selected Tool → {tool_name} | Args → {tool_input}")
            return step

        try:
            with yaspin(text="Processing query...", color="yellow") as sp:
                worker_out = await self.worker_agent.ainvoke(
                    {"input": query},
                    callbacks=[log_agent_step],
                )
                sp.ok("✔")
        except Exception as e:
            logger.error(f"[LLM] Agent execution failed: {str(e)}")
            error_context = {
                "error": str(e),
                "steps_completed": len(intermediate_steps),
                "last_steps": intermediate_steps[-3:] if intermediate_steps else []
            }
            return f"❌ Error during execution:\n{str(e)}\n\nContext:\n{json.dumps(error_context, indent=2)}"

        raw_answer = worker_out.get("output", worker_out)

        # Analyze if there were errors
        has_errors = any([
            "ERROR" in raw_answer,
            "failed" in raw_answer.lower(),
            "error while loading" in raw_answer.lower(),
            "could not" in raw_answer.lower(),
        ])

        # Generate comprehensive summary
        summary_prompt = f"""
Analyze the following MCP tool execution results and provide a clear, actionable summary.

Tool Execution Results:
{raw_answer}

Intermediate Steps:
{json.dumps(intermediate_steps[-5:], indent=2) if intermediate_steps else "No intermediate steps recorded"}

Errors Detected: {"YES - Focus on error analysis" if has_errors else "NO - Operation appears successful"}

Provide a summary that includes:

1. **Operations Performed**: List each tool that was executed and what it did
2. **Results**: What did each tool return? Was it successful?
3. **Validation Status**: Were the changes verified? Did get-iflow confirm success?
4. **Errors (if any)**:
   - Exact error messages
   - Which component failed
   - Root cause analysis
   - Why it failed
5. **Next Steps**: 
   - If successful: What the user can do now
   - If failed: Specific steps to fix the issue
6. **Recommendations**: Best practices or warnings

Format your response clearly with sections. Be specific and actionable.
"""

        with yaspin(text="Generating summary...", color="cyan") as sp:
            summary = await self.llm.ainvoke(summary_prompt)
            sp.ok("✔")

        logger.info(f"[LLM] Final Summary Generated")
        
        # Add visual indicator for errors
        if has_errors:
            summary = "⚠️ **ERRORS DETECTED**\n\n" + summary
        else:
            summary = "✅ **OPERATION COMPLETED**\n\n" + summary

        return summary

    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("SAP Integration Suite MCP Chatbot")
        print("="*60)
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
                print("\n" + "─"*60)
                print(answer)
                print("─"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                logger.exception("Unexpected error in chat loop")

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
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Fatal error in main")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())