# mapping_context/xsd_context_builder.py
import re
import json
import requests
import xmlschema
from typing import Optional, Dict

XSD_URL_PATTERN = r"https?://[^\s]+\.xsd"
user_query = ["https://hcp-9590b8a4-fcd8-4832-a678-cc655208784e.s3.amazonaws.com/demo/XSLT%20source.xsd","https://hcp-9590b8a4-fcd8-4832-a678-cc655208784e.s3.amazonaws.com/demo/XSLT%20target.xsd"]

def extract_xsd_urls(text: str) -> Dict[str, str]:
    urls = re.findall(XSD_URL_PATTERN, text)

    result = {}
    for url in urls:
        if "source" in url.lower():
            result["source"] = url
        elif "target" in url.lower():
            result["target"] = url

    return result


def load_xsd(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def xsd_to_flat_schema(xsd_content: str) -> list[dict]:
    schema = xmlschema.XMLSchema(xsd_content)
    elements = []

    for path, elem in schema.elements.items():
        elements.append({
            "path": f"/{path}",
            "type": elem.type.name if elem.type else "string",
            "required": elem.min_occurs > 0,
            "max_occurs": elem.max_occurs
        })

    return elements


def build_schema_context(user_query: str) -> Optional[str]:
    """
    Returns LLM-safe schema context ONLY if XSD URLs are present.
    Otherwise returns None.
    """

    urls = extract_xsd_urls(user_query)
    if not urls:
        return None

    context = {}

    for role, url in urls.items():
        xsd_content = load_xsd(url)
        context[f"{role}_schema"] = xsd_to_flat_schema(xsd_content)

    return f"""
### MESSAGE MAPPING SCHEMA CONTEXT

SOURCE_SCHEMA_JSON:
{json.dumps(context.get("source_schema"), indent=2)}

TARGET_SCHEMA_JSON:
{json.dumps(context.get("target_schema"), indent=2)}

Rules:
- Do NOT invent fields
- Use SAP CPI Message Mapping functions only
- Prefer direct mapping when possible
"""

if __name__ == "__main__":
    test_query = """
    Create message mapping using:
    Source XSD: https://hcp-9590b8a4-fcd8-4832-a678-cc655208784e.s3.amazonaws.com/demo/XSLT%20source.xsd
    Target XSD: https://hcp-9590b8a4-fcd8-4832-a678-cc655208784e.s3.amazonaws.com/demo/XSLT%20target.xsd
    """

    context = build_schema_context(test_query)

    print("\n====== PARSED SCHEMA CONTEXT ======\n")
    print(context)
