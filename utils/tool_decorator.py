"""
Tool Decorator for Claude API.

This module provides a decorator to convert Python functions into
Claude-compatible tool configurations.

Usage:
    from utils.tool_decorator import tool, get_tool_configs

    @tool(
        "my_tool",
        "Does something useful",
        {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First param"},
            },
            "required": ["param1"]
        }
    )
    async def my_tool(args: dict) -> dict:
        return {"content": [{"type": "text", "text": "Done"}]}

    # Get tool configs for Claude API
    configs = get_tool_configs([my_tool])
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, Union, get_type_hints

try:
    from typing import Literal, get_origin, get_args
except ImportError:
    from typing_extensions import Literal, get_origin, get_args

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# Python Type to JSON Schema Conversion
# ============================================================================

# Python type to JSON Schema type mapping
_PYTHON_TYPE_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """
    Convert a Python type to JSON Schema type definition.

    Args:
        py_type: Python type (str, int, float, bool, list, dict, etc.)

    Returns:
        JSON Schema type definition dict
    """
    # Handle basic types
    if py_type in _PYTHON_TYPE_TO_JSON_SCHEMA:
        return {"type": _PYTHON_TYPE_TO_JSON_SCHEMA[py_type]}

    # Handle typing module types (List, Dict, Optional, etc.)
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is list:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    if origin is dict:
        schema = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = _python_type_to_json_schema(args[1])
        return schema

    # Handle Union types (including Optional)
    if origin is Union:
        non_none_types = [t for t in args if t is not type(None)]
        types = [_python_type_to_json_schema(t)["type"] for t in non_none_types]
        if len(types) == 1:
            return {"type": types[0]}
        return {"type": types}

    # Handle Python 3.10+ union syntax (int | str)
    try:
        import types as builtin_types
        if isinstance(py_type, builtin_types.UnionType):
            non_none_types = [t for t in args if t is not type(None)]
            types = [_python_type_to_json_schema(t)["type"] for t in non_none_types]
            if len(types) == 1:
                return {"type": types[0]}
            return {"type": types}
    except AttributeError:
        pass

    # Handle Literal types
    if origin is Literal:
        literal_values = args
        if literal_values:
            first_val = literal_values[0]
            val_type = _PYTHON_TYPE_TO_JSON_SCHEMA.get(type(first_val), "string")
            return {"type": val_type, "enum": list(literal_values)}

    # Default to string for unknown types
    return {"type": "string"}


def _convert_simple_schema(simple_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a simple schema (Python types) to full JSON Schema format.

    Args:
        simple_schema: Dict mapping parameter names to Python types or type dicts
                       e.g., {"name": str, "count": int}
                       or {"name": {"type": str, "description": "..."}}

    Returns:
        Full JSON Schema compatible dict
    """
    # If already in JSON Schema format, return as-is
    if "type" in simple_schema and simple_schema.get("type") == "object":
        return simple_schema

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param_spec in simple_schema.items():
        if isinstance(param_spec, type):
            # Simple type: {"name": str}
            properties[param_name] = _python_type_to_json_schema(param_spec)
            required.append(param_name)
        elif isinstance(param_spec, dict):
            # Dict spec: {"name": {"type": str, "description": "...", "default": ...}}
            prop_schema: dict[str, Any] = {}

            if "type" in param_spec:
                type_val = param_spec["type"]
                if isinstance(type_val, type):
                    prop_schema.update(_python_type_to_json_schema(type_val))
                elif isinstance(type_val, str):
                    prop_schema["type"] = type_val

            if "description" in param_spec:
                prop_schema["description"] = param_spec["description"]

            if "enum" in param_spec:
                prop_schema["enum"] = param_spec["enum"]

            if "default" in param_spec:
                prop_schema["default"] = param_spec["default"]

            if "items" in param_spec:
                prop_schema["items"] = param_spec["items"]

            if "properties" in param_spec:
                prop_schema["properties"] = param_spec["properties"]

            if "required" in param_spec:
                prop_schema["required"] = param_spec["required"]

            properties[param_name] = prop_schema

            # Only add to required if no default value
            if "default" not in param_spec:
                required.append(param_name)
        else:
            # Unknown format, try to convert
            properties[param_name] = {"type": "string"}
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required if required else []
    }


def _extract_schema_from_function(func: Callable) -> dict[str, Any]:
    """
    Extract JSON Schema from function signature and type hints.

    Args:
        func: The function to extract schema from

    Returns:
        JSON Schema dict for the function parameters
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'self', 'cls', and special parameters
        if param_name in ('self', 'cls', 'args', 'kwargs'):
            continue

        # Skip if the function takes a single 'args' dict (common pattern)
        if param_name == 'args' and hints.get(param_name) in (dict, dict[str, Any]):
            continue

        prop_schema: dict[str, Any] = {}

        # Get type from hints or annotation
        if param_name in hints:
            prop_schema = _python_type_to_json_schema(hints[param_name])
        elif param.annotation != inspect.Parameter.empty:
            prop_schema = _python_type_to_json_schema(param.annotation)
        else:
            prop_schema = {"type": "string"}

        properties[param_name] = prop_schema

        # Check if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop_schema["default"] = param.default

    return {
        "type": "object",
        "properties": properties,
        "required": required if required else []
    }


# ============================================================================
# Tool Configuration Class
# ============================================================================

class ToolConfig:
    """
    Represents a Claude-compatible tool configuration.

    This class stores the tool metadata and provides methods to convert
    to various formats (Claude API, etc.).
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.func = func

    def to_claude_tool(self) -> dict[str, Any]:
        """
        Convert to Claude API tool format.

        Returns:
            Dict compatible with Claude API's tool specification
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for to_claude_tool()."""
        return self.to_claude_tool()

    def __repr__(self) -> str:
        desc_preview = self.description[:50] + "..." if len(self.description) > 50 else self.description
        return f"ToolConfig(name='{self.name}', description='{desc_preview}')"


# ============================================================================
# Tool Decorator
# ============================================================================

def tool(
    name: str | None = None,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to convert a Python function into a Claude-compatible tool.

    This decorator attaches tool metadata to a function, allowing it to be
    easily converted to Claude API's tool format.

    Args:
        name: Tool name. Defaults to function name.
        description: Tool description. Defaults to function docstring.
        input_schema: JSON Schema for input parameters. Can be:
            - Full JSON Schema: {"type": "object", "properties": {...}, "required": [...]}
            - Simple type mapping: {"param1": str, "param2": int}
            - None to auto-extract from function signature

    Returns:
        Decorated function with attached tool configuration

    Examples:
        # Example 1: Full JSON Schema
        @tool(
            "search_database",
            "Search the database for records",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        )
        async def search_database(args: dict) -> dict:
            query = args["query"]
            limit = args.get("limit", 10)
            # ... implementation
            return {"content": [{"type": "text", "text": "Results..."}]}

        # Example 2: Simple type mapping
        @tool("greet", "Greets a person by name", {"name": str, "age": int})
        async def greet(args: dict) -> dict:
            return {"content": [{"type": "text", "text": f"Hello {args['name']}!"}]}

        # Example 3: Auto-extract from function signature
        @tool()
        async def process(text: str, count: int = 5) -> dict:
            '''Process text with specified count.'''
            return {"content": [{"type": "text", "text": f"Processed {text}"}]}

        # Get Claude tool config
        config = search_database.tool_config.to_claude_tool()
        # Or use the shorthand:
        config = search_database.to_claude_tool()
    """
    def decorator(func: F) -> F:
        # Determine tool name
        tool_name = name if name is not None else func.__name__

        # Determine description
        tool_description = description
        if tool_description is None:
            tool_description = func.__doc__ or f"Tool: {tool_name}"
            tool_description = tool_description.strip()

        # Determine input schema
        if input_schema is None:
            tool_input_schema = _extract_schema_from_function(func)
        elif "type" in input_schema and input_schema.get("type") == "object":
            # Already in JSON Schema format
            tool_input_schema = input_schema
        else:
            # Simple format - convert to JSON Schema
            tool_input_schema = _convert_simple_schema(input_schema)

        # Create tool config
        config = ToolConfig(
            name=tool_name,
            description=tool_description,
            input_schema=tool_input_schema,
            func=func,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Handle both calling conventions:
            # 1. func(args={"key": "value"}) or func({"key": "value"}) - single dict arg
            # 2. func(key="value") or func(**{"key": "value"}) - unpacked kwargs

            # If called with kwargs and no positional args, check if the function
            # expects a single dict argument (common tool pattern)
            if kwargs and not args:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                # If function has a single parameter named 'args' (tool convention)
                if len(params) == 1 and params[0] == 'args':
                    # Package kwargs into a dict and pass as the single argument
                    return func(kwargs)

            return func(*args, **kwargs)

        # Attach tool config to the wrapper
        wrapper.tool_config = config  # type: ignore
        wrapper.is_tool = True  # type: ignore

        # Convenience methods
        wrapper.to_claude_tool = config.to_claude_tool  # type: ignore
        wrapper.to_dict = config.to_dict  # type: ignore

        return wrapper  # type: ignore

    return decorator


# ============================================================================
# Helper Functions
# ============================================================================

def get_tool_configs(tools: list[Callable]) -> list[dict[str, Any]]:
    """
    Extract Claude API tool configurations from a list of decorated functions.

    Args:
        tools: List of functions decorated with @tool

    Returns:
        List of tool configuration dicts compatible with Claude API

    Example:
        tools = [search_database, greet, process]
        configs = get_tool_configs(tools)

        # Use with Claude API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            tools=configs,
            messages=[...]
        )
    """
    configs = []
    for tool_func in tools:
        if hasattr(tool_func, 'tool_config'):
            configs.append(tool_func.tool_config.to_claude_tool())
        elif hasattr(tool_func, 'to_claude_tool'):
            configs.append(tool_func.to_claude_tool())
        else:
            logger.warning(f"Function {getattr(tool_func, '__name__', str(tool_func))} is not decorated with @tool")
    return configs


def create_tools_from_class(
    cls: type,
    prefix: str | None = None,
    include_private: bool = False,
) -> list[dict[str, Any]]:
    """
    Create tool configurations from all @tool decorated methods of a class.

    Args:
        cls: Class to extract methods from
        prefix: Optional prefix to add to tool names
        include_private: Whether to include methods starting with '_'

    Returns:
        List of tool configuration dicts

    Example:
        class MyTools:
            @tool("search", "Search for items", {"query": str})
            def search(self, args: dict) -> dict:
                ...

            @tool("create", "Create an item", {"name": str})
            def create(self, args: dict) -> dict:
                ...

        configs = create_tools_from_class(MyTools, prefix="my")
        # Results in tools named "my_search" and "my_create"
    """
    tools = []

    for method_name in dir(cls):
        if not include_private and method_name.startswith('_'):
            continue

        method = getattr(cls, method_name)
        if not callable(method):
            continue

        if hasattr(method, 'tool_config'):
            config = method.tool_config.to_claude_tool()
            if prefix:
                config['name'] = f"{prefix}_{config['name']}"
            tools.append(config)

    return tools


def tools_to_claude_format(tools: list[Callable]) -> list[dict[str, Any]]:
    """
    Alias for get_tool_configs. Converts tool functions to Claude API format.

    Args:
        tools: List of @tool decorated functions

    Returns:
        List of tool configs ready for Claude API
    """
    return get_tool_configs(tools)
