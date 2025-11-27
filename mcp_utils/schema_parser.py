# mcp_utils/schema_parser.py
import logging
from typing import Any, Dict
from pydantic import create_model, BaseModel, ValidationError

logger = logging.getLogger("pipo.schema")

_tool_model_cache: Dict[str, BaseModel] = {}


def json_type_to_py(t):
    return {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }.get(t, Any)


def build_pydantic_field_from_prop(prop: dict, name_hint: str = "Nested"):
    t = prop.get("type", "string")

    if t == "array":
        items = prop.get("items", {})
        if items.get("type") == "object":
            nested_fields = {}
            for k, v in items.get("properties", {}).items():
                py_type, _ = build_pydantic_field_from_prop(v, k.title())
                required = k in items.get("required", [])
                nested_fields[k] = (py_type, ... if required else None)
            Nested = create_model(f"{name_hint}Item", **nested_fields)
            return (list[Nested], ...)
        return (list, ...)

    if t == "object":
        return (dict, ...)

    return (json_type_to_py(t), ...)


def build_input_model_for_tool(tool_def) -> BaseModel:
    name = tool_def.name

    if name in _tool_model_cache:
        return _tool_model_cache[name]

    schema = tool_def.inputSchema or {}
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []))

    fields = {}
    for key, prop in props.items():
        py_type, _ = build_pydantic_field_from_prop(prop, key.title())
        fields[key] = (py_type, ... if key in required else None)

    Model = create_model(f"{name}_InputModel", **fields)
    _tool_model_cache[name] = Model
    return Model


# --------------------------------------------------------------
# Fully async tool input validation
# --------------------------------------------------------------
async def validate_tool_input(session, tool_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates input against MCP tool schema asynchronously.
    """

    tools = (await session.list_tools()).tools
    tdef = next((t for t in tools if t.name == tool_name), None)

    if not tdef:
        raise RuntimeError(f"Tool {tool_name} not found in MCP server")

    Model = build_input_model_for_tool(tdef)

    try:
        validated = Model(**data)
    except ValidationError as e:
        logger.error("Validation error for tool %s: %s", tool_name, e)
        raise

    return validated.dict()
