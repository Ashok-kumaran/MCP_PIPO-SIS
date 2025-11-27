# graph/tools_node.py
import json
import logging
from typing import Any, Dict

logger = logging.getLogger("pipo.tools")

from mcp_utils.file_manager import FileManager
from mcp_utils.schema_parser import build_input_model_for_tool

class ToolsNode:
    def __init__(self, session, file_manager: FileManager):
        self.session = session
        self.file_manager = file_manager
        # cache tool models
        self._model_cache = {}

    async def prepare_input(self, validated_input: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares input before calling MCP. This handles file content -> filepaths, variable substitution, etc.
        """
        if "files" in validated_input and isinstance(validated_input["files"], list):
            files = []
            for f in validated_input["files"]:

                # CASE 1: file has content → must write to disk
                if "content" in f and "filepath" not in f:
                    filename = f.get("filename") or "artifact"
                    fp = self.file_manager.save_content(filename, f["content"])

                    file_entry = {
                        "filepath": fp,
                        **{k: v for k, v in f.items() if k not in ("content", "filepath")}
                    }

                    # Ensure appendMode is always present
                    if "appendMode" not in file_entry:
                        file_entry["appendMode"] = False

                    files.append(file_entry)

                # CASE 2: file already has filepath
                else:
                    if "appendMode" not in f:
                        f["appendMode"] = False
                    files.append(f)

            return {**validated_input, "files": files}

        return validated_input


    async def call_tool(self, tool_name: str, input_payload: Dict[str, Any]):
        """
        Calls the MCP server tool via session.call_tool.
        """
        logger.info("Calling tool %s with input %s", tool_name, input_payload)
        resp = await self.session.call_tool(tool_name, input_payload)
        # unwrap response content
        outputs = []
        for c in resp.content or []:
            if getattr(c, "text", None):
                outputs.append(c.text)
            elif getattr(c, "json", None):
                outputs.append(c.json)
            elif getattr(c, "bytes", None):
                outputs.append(c.bytes)
            else:
                outputs.append(str(c))
        logger.info("Tool %s returned %s", tool_name, outputs)
        return outputs

    async def handle_result(self, step: Dict[str, Any], tool_result, state: Dict[str, Any]):
        """
        Basic handler: store tool_result in state.history and optionally extract artifacts.
        """
        entry = {"step": step.get("id"), "tool": step.get("tool"), "result": tool_result}
        state.setdefault("history", []).append(entry)

        # simple artifact extraction heuristic
        if step.get("action") == "create-iflow":
            # assume tool_result contains a created id
            created = None
            for r in tool_result:
                try:
                    o = json.loads(r)
                    if isinstance(o, dict) and o.get("id"):
                        created = o["id"]
                        break
                except Exception:
                    continue
            if created:
                state.setdefault("artifacts", {})["iflow_id"] = created
