# graph/executor.py
import logging
from typing import Any, Dict, List

from graph.summarizer import Summarizer
from mcp_utils.schema_parser import validate_tool_input

logger = logging.getLogger("pipo.executor")

import json


# -----------------------------------------------------------
# JSON TEMPLATE BUILDERS FOR CPI ARTIFACTS
# -----------------------------------------------------------
def build_metadata_json(iflow_id, iflow_name, description):
    template = {
        "id": iflow_id,
        "name": iflow_name,
        "description": description,
        "version": "1.0.0",
        "createdBy": "AI-Generator",
        "updatedBy": "AI-Generator",
        "contentType": "application/vnd.sap-ifl-bpmn+xml",
        "formatVersion": "1.1"
    }
    return json.dumps(template, indent=2)


def build_integration_json(iflow_id, iflow_name, package_id):
    template = {
        "id": iflow_id,
        "name": iflow_name,
        "packageId": package_id,
        "type": "BPMN",
        "version": "1.0.0",
        "artifactVersion": "1.0.0",
        "context": {}
    }
    return json.dumps(template, indent=2)


# -----------------------------------------------------------


class Executor:
    def __init__(self, session, tools_node, callbacks=None, max_iterations=50, llm=None):
        self.session = session
        self.tools = tools_node
        self.callbacks = callbacks or []
        self.max_iterations = max_iterations
        self.summarizer = Summarizer(llm) if llm else None
        self.llm = llm  # <-- needed for final summary

    # =================================================================
    async def execute_plan(self, plan: List[Dict[str, Any]]):
        # =================================================================
        state = {"artifacts": {}, "vars": {}, "history": []}
        results = []

        for idx, step in enumerate(plan, 1):
            tool = step["tool"]
            input_data = step["input"]

            logger.info("Executing step %s with tool=%s input=%s", step["id"], tool, input_data)

            # ----------------------------------------------------------
            # 1) VALIDATE INPUT
            # ----------------------------------------------------------
            try:
                validated_input = await validate_tool_input(self.session, tool, input_data)
            except Exception as e:
                return {"error": f"Validation failed for tool {tool}: {e}"}

            # ----------------------------------------------------------
            # 2) CUSTOM INJECTION — ONLY FOR update-iflow
            # ----------------------------------------------------------
            if tool == "update-iflow":

                iflow_id = validated_input["id"]
                package_id = state.get("package_id", "default-package")
                name = iflow_id
                desc = f"Auto-generated iFlow {iflow_id}"

                # The BPMN XML comes from the previous step result
                # Last history entry contains the tool result (LLM result)
                last_entry = state["history"][-1] if state["history"] else None

                if last_entry and isinstance(last_entry.get("result"), list):
                    bpmn_content = last_entry["result"][0]
                else:
                    bpmn_content = "<!-- ERROR: Missing BPMN content -->"

                # CPI folder structure
                base_path = f"resources/scenarioflows/integrationflow/{iflow_id}.iflw"

                validated_input["files"] = [
                    {
                        "filepath": f"{base_path}/model.bpmn",
                        "content": bpmn_content,
                        "appendMode": False,
                    },
                    {
                        "filepath": f"{base_path}/metadata.json",
                        "content": build_metadata_json(iflow_id, name, desc),
                        "appendMode": False,
                    },
                    {
                        "filepath": f"{base_path}/integration.json",
                        "content": build_integration_json(iflow_id, name, package_id),
                        "appendMode": False,
                    },
                ]

            # ----------------------------------------------------------
            # 3) PREPARE INPUT (file saving, var replacement, etc.)
            # ----------------------------------------------------------
            call_input = await self.tools.prepare_input(validated_input, state)

            # ----------------------------------------------------------
            # 4) TOOL CALL
            # ----------------------------------------------------------
            try:
                tool_result = await self.tools.call_tool(tool, call_input)
            except Exception as e:
                logger.exception("Tool call failed")
                return {"error": f"Tool {tool} failed: {e}"}

            # ----------------------------------------------------------
            # 5) UPDATE STATE / HISTORY
            # ----------------------------------------------------------
            await self.tools.handle_result(step, tool_result, state)

            entry = {"step": step["id"], "tool": tool, "result": tool_result}

            # Inline summarization per-step
            if step.get("summarize", False) and self.summarizer:
                summary = await self.summarizer.summarize(tool_result)
                entry["summary"] = summary

            results.append(entry)
            state["history"].append(entry)

            # Run callbacks
            for cb in self.callbacks:
                try:
                    cb.on_chain_end({"step": step["id"], "result": tool_result})
                except Exception:
                    pass

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        final_summary = ""
        if self.llm:
            try:
                prompt = (
                    "You are an SAP Integration Suite expert.\n"
                    "Convert the following execution log into a clear, human-readable summary.\n"
                    "Explain what happened in each step.\n"
                    "Avoid SAP internal metadata.\n\n"
                    f"EXECUTION LOG:\n{results}"
                )
                resp = await self.llm.ainvoke(prompt)
                final_summary = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                final_summary = f"(Final summary generation failed: {e})"

        # -----------------------------------------------------------------
        return {
            "status": "success",
            "summary": final_summary,
            "results": results,
            "state": state
        }
