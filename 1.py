import asyncio
import sys
import os
import json
import logging
import re
from typing import Optional, Any, Type, Dict, List, Union
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from yaspin import yaspin

from pydantic import BaseModel, create_model

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain.callbacks.base import BaseCallbackHandler

from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

from deepagents import create_deep_agent

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters


# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    filename="deepagent_terminal.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ==========================================================
# SAP GenAI Hub LLM
# ==========================================================
def create_sap_llm() -> BaseChatModel:
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing in .env")

    return ChatOpenAI(
        deployment_id=deployment_id,
        temperature=0,
    )


# ==========================================================
# JSON Schema → Pydantic Model
# ==========================================================
def build_pydantic_model(name: str, schema: Dict, root: Dict = None) -> Any:
    if root is None:
        root = schema

    if "type" not in schema and "schema" in schema:
        schema = schema["schema"]

    if "$ref" in schema:
        ref = schema["$ref"]
        path = ref[2:].split("/")
        target = root
        for part in path:
            target = target.get(part, {})
        return build_pydantic_model(name, target, root)

    if "enum" in schema:
        from typing import Literal
        return Literal[tuple(schema["enum"])]

    if "oneOf" in schema:
        return Union[
            tuple(
                build_pydantic_model(f"{name}_oneof_{i}", s, root)
                for i, s in enumerate(schema["oneOf"])
            )
        ]

    if schema.get("type") == "object":
        fields = {}
        for key, subschema in schema.get("properties", {}).items():
            fields[key] = (
                build_pydantic_model(f"{name}_{key}", subschema, root),
                ... if key in schema.get("required", []) else None,
            )
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        return create_model(safe_name, **fields)

    if schema.get("type") == "array":
        return List[build_pydantic_model(f"{name}_item", schema["items"], root)]

    return {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }.get(schema.get("type"), Any)


# ==========================================================
# MCP Tool Wrapper
# ==========================================================
class MCPAsyncTool(BaseTool):
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Async only")

    async def _arun(self, **kwargs):
        with yaspin(text=f"Running MCP tool {self.mcp_tool_name}...", color="magenta"):
            try:
                result = await self.session.call_tool(self.mcp_tool_name, kwargs)
            except Exception as e:
                return f"ERROR: {e}"

        blocks = []
        for c in result.content:
            if getattr(c, "type", "") == "text":
                blocks.append(c.text)
            elif getattr(c, "type", "") == "json":
                blocks.append(json.dumps(c.json or c.data, indent=2))
            else:
                blocks.append(str(c))

        output = "\n".join(blocks)
        if len(output) > 5000:
            return "Output too large. Write to file using write_file."

        return output


# ==========================================================
# Debug Callback
# ==========================================================
class StepLogger(BaseCallbackHandler):
    def __init__(self, collector):
        self.collector = collector

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.collector.append({"tool_start": serialized.get("name"), "input": input_str})

    def on_tool_end(self, output, **kwargs):
        self.collector.append({"tool_end": str(output)})


# ==========================================================
# DeepAgent Client
# ==========================================================
class DeepAgentSAP:

    IFLOW_TOOLS = {"packages", "package", "create-package", 
        "get-iflow", "create-empty-iflow", "update-iflow", "deploy-iflow",
        "get-iflow-endpoints", "iflow-image", "get-iflow-configurations", "get-all-iflows",
        "get-messagemapping", "update-message-mapping", "deploy-message-mapping", 
        "create-empty-mapping", "get-all-messagemappings", 
        "discover-packages", "list-iflow-examples", "get-iflow-example",
        "list-mapping-examples", "get-mapping-example", "create-mapping-testiflow",
        "get-deploy-error", "get-messages", "count-messages", "send-http-message"
    }  # unchanged, omitted for brevity
    TPM_TOOLS = { "get-partner-metadata", "get-partner", "create-trading-partner", "get-systems-of-partner",
        "create-system", "get-system-types", "create-identifier", "get-qualifiers-codelist",
        "create-communication", "get-sender-adapters", "get-receiver-adapters", 
        "create-signature-verify-config", "activate-signature-verify-config", "get-all-company-profile-metadata",
        "get-all-agreement-metadata", "get-all-agreement-template-metadata", "get-agreement-template", 
        "create-agreement-with-bound-template", "get-agreement-b2b-scenario", "update-b2b-scenario", 
        "trigger-agreement-activate-or-update-deployment",
        "get-all-mig-latest-metadata", "get-mig-raw-by-id", "get-mig-nodes-xpath", "get-all-mig-fields",
        "get-mig-documentation-entry", "get-mig-proposal", "apply-mig-proposal", 
        "create-mig-draft-all-segments-selected", "create-mig", "change-mig-field-selection",
        "get-all-mags-metadata", "create-mapping-guidelines", "test-mag-with-message",
        "search-interchanges", "get-interchange-payloads", "download-interchange-payload", "get-interchange-last-error",
        "get-type-systems", "get-type-system-messages", "get-type-system-message-full", "create-custom-message",
        "get-type-system-identifier-schemes", "get-all-business-process-roles", "get-all-business-processes", 
        "get-all-industry-classifications", "get-all-product-classifications", "get-all-products",
        "get-all-contries-or-regions"
    }

    def __init__(self, script: str):
        self.script = script
        self.exit_stack = AsyncExitStack()
        self.llm = create_sap_llm()
        self.session: Optional[ClientSession] = None

        self.temp_dir = "./vfs_storage"
        os.makedirs(self.temp_dir, exist_ok=True)

    async def connect_and_initialize(self):
        cmd = "node" if self.script.endswith(".js") else "python"
        params = StdioServerParameters(command=cmd, args=[self.script])

        reader, writer = await self.exit_stack.enter_async_context(
            stdio_client(params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(reader, writer)
        )
        await self.session.initialize()

        await self.load_agents()
        print("✅ DeepAgent system ready")

    async def load_agents(self):
        tools = await self.session.list_tools()

        iflow_tools, tpm_tools = [], []

        for t in tools.tools:
            Model = build_pydantic_model(t.name + "_Input", t.inputSchema or {})
            tool = MCPAsyncTool(
                name=t.name,
                description=t.description or "",
                args_schema=Model,
                session=self.session,
                mcp_tool_name=t.name,
            )
            if t.name in self.IFLOW_TOOLS:
                iflow_tools.append(tool)
            elif t.name in self.TPM_TOOLS:
                tpm_tools.append(tool)

        self.iflow_agent = create_deep_agent(
            llm=self.llm,
            tools=iflow_tools,
            system_prompt="You are an SAP Integration Flow expert.",
            name="integration_flow_agent",
        )

        self.tpm_agent = create_deep_agent(
            llm=self.llm,
            tools=tpm_tools,
            system_prompt="You are an SAP TPM expert.",
            name="tpm_agent",
        )

        self.planner = create_deep_agent(
            llm=self.llm,
            tools=[
                ReadFileTool(root_dir=self.temp_dir),
                WriteFileTool(root_dir=self.temp_dir),
                ListDirectoryTool(root_dir=self.temp_dir),
                self.iflow_agent,
                self.tpm_agent,
            ],
            system_prompt="You are an SAP Integration Suite Architect.",
        )

    async def chat_loop(self):
        print("\n=== SAP DeepAgent Architect ===\n")
        while True:
            q = input("You: ")
            if q.lower() in ("exit", "quit", "q"):
                break
            res = await self.planner.ainvoke({"input": q})
            print("\n--- RESPONSE ---\n", res["output"])

    async def cleanup(self):
        await self.exit_stack.aclose()


# ==========================================================
# MAIN
# ==========================================================
async def main():
    if len(sys.argv) < 2:
        print("Usage: python dagent.py <server.js|server.py>")
        return

    client = DeepAgentSAP(sys.argv[1])
    try:
        await client.connect_and_initialize()
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
