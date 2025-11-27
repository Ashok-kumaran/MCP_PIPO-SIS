# graph/summarizer.py
import json
import logging

logger = logging.getLogger("pipo.summarizer")


class Summarizer:
    """
    Summarizes raw MCP tool outputs using the LLM.
    """

    def __init__(self, llm):
        self.llm = llm

    async def summarize(self, raw_output):
        """
        raw_output: list of raw strings or JSON objects from MCP tool.
        """

        # Convert all JSON strings to dicts for clarity
        merged = []
        for item in raw_output:
            if isinstance(item, str):
                try:
                    merged.append(json.loads(item))
                except Exception:
                    merged.append({"raw": item})
            else:
                merged.append(item)

        text_payload = json.dumps(merged, indent=2)

        prompt = f"""
You are an expert SAP Integration Suite assistant.

Summarize the following tool output in a clean, human-readable way.
Focus on:
- package name
- description
- platform
- artifacts (integrations, mappings)
- created/modified fields
- any meaningful metadata

Avoid SAP raw technical metadata like __deferred objects, URIs, etc.

RAW JSON:
{text_payload}

Write only the summary.
"""

        logger.info("Summarizer prompt: %s", prompt)

        resp = await self.llm.ainvoke(prompt)
        try:
            return resp.content
        except:
            return str(resp)
