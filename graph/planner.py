# graph/planner.py
import json
import logging
from typing import Any, List, Dict

logger = logging.getLogger("pipo.planner")
# --- iFlow Template Loader ---
def load_iflow_template():
    """Load the base CPI iFlow example XML."""
    try:
        with open("examples/base.iflw", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<!-- Failed to load iFlow template: {e} -->"


PLANNER_PROMPT_PATH = "prompts/planner_prompt.txt"


class Planner:
    def __init__(self, llm, tools_list=None, tools_meta=None):
        """
        llm: SAP GenAI Hub ChatOpenAI model.
        tools_list: list of MCP tool names.
        tools_meta: dict { tool_name: inputSchema }
        """
        self.llm = llm
        self.tools_list = tools_list or []
        self.tools_meta = tools_meta or {}

        with open(PLANNER_PROMPT_PATH, "r", encoding="utf-8") as f:
            self.planner_prompt_template = f.read()

    # --------------------------------------------------------------
    async def create_plan(self, user_prompt: str) -> List[Dict[str, Any]]:

        # Convert tools list
        tool_names_str = ", ".join(f'"{t}"' for t in self.tools_list)

        # Convert schemas to readable JSON
        schemas_str = json.dumps(self.tools_meta, indent=2)

        # Build final prompt
        prompt = (
            self.planner_prompt_template
            .replace("{{USER_PROMPT}}", user_prompt)
            .replace("{{TOOLS}}", tool_names_str)
            .replace("{{SCHEMAS}}", schemas_str)
        )

        logger.info("Planner prompt:\n%s", prompt)

        # Automatically inject example CPI iFlow structure when user asks for an iFlow
        lower_prompt = user_prompt.lower()

        if "iflow" in lower_prompt or "cpi" in lower_prompt or "integration flow" in lower_prompt:
            example_iflw = load_iflow_template()
            prompt += (
                "\n\nBelow is an example SAP CPI iFlow XML structure. "
                "Use its structure and pattern when generating a new iFlow for the user's request:\n\n"
                f"```xml\n{example_iflw}\n```"
            )

        # ----------------------------
        # LLM CALL
        # ----------------------------
        resp = await self.llm.ainvoke(prompt)

        # Extract content from AIMessage
        text = getattr(resp, "content", str(resp)).strip()
        logger.info("Raw planner output: %s", text)

        # ----------------------------
        # Remove fenced code blocks
        # ----------------------------
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        # ----------------------------
        # JSON PARSING
        # ----------------------------
        try:
            plan = json.loads(text)
        except Exception as e:
            logger.error("Planner returned invalid JSON: %s", text)
            raise RuntimeError("Planner failed to produce valid JSON") from e

        if not isinstance(plan, list):
            raise RuntimeError("Planner output must be a JSON array")

        # ----------------------------
        # VALIDATE EACH STEP
        # ----------------------------
        for idx, step in enumerate(plan):

            # Valid object
            if not isinstance(step, dict):
                raise RuntimeError(f"Step {idx + 1} must be an object")

            # Required keys
            for key in ("id", "action", "tool", "input", "description"):
                if key not in step:
                    raise RuntimeError(f"Step {idx+1} missing required key '{key}'")

            tool = step["tool"]
            inp = step["input"]

            # Validate tool exists
            if tool not in self.tools_list:
                raise RuntimeError(
                    f"Invalid tool '{tool}' in step {idx+1}. Allowed: {self.tools_list}"
                )

            # Validate input keys vs schema
            schema = self.tools_meta.get(tool, {})
            required = set(schema.get("required", []))

            for r in required:
                if r not in inp:
                    raise RuntimeError(
                        f"Step {idx+1} is missing REQUIRED field '{r}' for tool '{tool}'.\n"
                        f"Expected schema: {schema}"
                    )

        logger.info("Plan validated successfully (%d steps).", len(plan))
        return plan
