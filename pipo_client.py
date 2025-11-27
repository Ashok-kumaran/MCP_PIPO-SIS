# pipo_client.py
import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI  # your SAP GenAI hub wrapper
from graph.planner import Planner
from graph.executor import Executor
from mcp_utils.callbacks import LogEverythingCallback
from mcp_utils.schema_parser import build_input_model_for_tool
from mcp_utils.file_manager import FileManager

logger = logging.getLogger("pipo")
logging.basicConfig(level=logging.INFO, filename="pipo_client.log",
                    filemode="a", format="%(asctime)s %(levelname)s %(message)s")
# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

load_dotenv()


def create_sap_llm():
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing in .env")
    return ChatOpenAI(deployment_id=deployment_id, temperature=0)


class MCPClientApp:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.exit_stack = AsyncExitStack()
        self.session = None
        self.stdio = None
        self.write = None
        self.llm = create_sap_llm()
        self.file_mgr = FileManager(base_dir="artifacts")
        self.callbacks = [LogEverythingCallback()]

    async def start(self):
        if self.server_script.endswith(".py"):
            cmd = "python"
        elif self.server_script.endswith(".js"):
            cmd = "node"
        else:
            raise ValueError("Server must be .py or .js")

        params = StdioServerParameters(command=cmd, args=[self.server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        # load tools schemas into pydantic models cache (optional)
        tools = (await self.session.list_tools()).tools
        for t in tools:
            build_input_model_for_tool(t)  # ensures cache ready

        logger.info("MCP client connected. Tools: %s", [t.name for t in tools])

    async def stop(self):
        if self.session:
            try:
                await self.session.shutdown()
            except Exception:
                pass
        await self.exit_stack.aclose()

    async def interactive(self):
        """
        Runs the interactive REPL loop. 
        NOTE: No shutdown is performed here — shutdown happens ONLY in main().
        This avoids AnyIO cancel-scope errors and double-shutdown.
        """

        # --- NEW: Extract tools + schemas ---
        tool_defs = (await self.session.list_tools()).tools

        tools_list = [t.name for t in tool_defs]
        tools_meta = {t.name: (t.inputSchema or {}) for t in tool_defs}

        # Pass schemas to Planner
        planner = Planner(
            self.llm,
            tools_list=tools_list,
            tools_meta=tools_meta
        )

        while True:
            user_prompt = await asyncio.to_thread(input, "\nUser prompt (or 'quit'): ")

            if user_prompt.strip().lower() in ("quit", "exit"):
                print("Exiting interactive mode...")
                break

            # --- PLANNING ---
            try:
                plan = await planner.create_plan(user_prompt)
                logger.info("Planner returned plan: %s", plan)
            except Exception as e:
                logger.exception("Planner failed")
                print(f"Error creating plan: {e}")
                continue

            # --- TOOL NODE SETUP ---
            from graph.tools_node import ToolsNode
            tools_node = ToolsNode(self.session, self.file_mgr)

            # --- EXECUTION ---
            executor = Executor(
                self.session,
                tools_node,
                callbacks=self.callbacks,
                llm=self.llm
            )

            try:
                result = await executor.execute_plan(plan)
            except Exception as e:
                logger.exception("Executor failed")
                print(f"Execution error: {e}")
                continue

            # --- OUTPUT ---
            print("\n=== RESULT ===")
            print(result)
            # If executor returned a natural-language summary, print that only
            if isinstance(result, dict) and "summary" in result and result["summary"]:
                print(result["summary"])
            else:
                # fallback to raw JSON (only if summary missing)
                print(result)

            print("==============\n")
            print("==============\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipo_client.py <server.py|server.js>")
        sys.exit(1)

    server = sys.argv[1]

    async def main():
        app = MCPClientApp(server)
        await app.start()
        try:
            await app.interactive()
        finally:
            await app.stop()

    asyncio.run(main())

