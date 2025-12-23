import re
import json
from typing import Optional, Dict

from mapping_context.object_store_client import ObjectStoreClient
from mapping_context.xsd_parser import xsd_to_flat_schema

XSD_PATTERN = r"s3://([^/]+)/(.+\.xsd)"


def extract_xsd_keys(text: str) -> Dict[str, str]:
    """
    Expected format in user input:
    s3://bucket-name/path/to/source.xsd
    """
    matches = re.findall(XSD_PATTERN, text)

    result = {}
    for bucket, key in matches:
        if "source" in key.lower():
            result["source"] = key
        elif "target" in key.lower():
            result["target"] = key

    return result


def build_schema_context(user_query: str) -> Optional[str]:
    keys = extract_xsd_keys(user_query)
    if not keys:
        return None

    store = ObjectStoreClient()
    context = {}

    for role, key in keys.items():
        xsd_content = store.read_object(key)
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
