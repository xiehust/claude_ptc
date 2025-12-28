"""
工具注册表 - 管理可被代码调用的工具
"""

import json
import inspect
from dataclasses import dataclass, field
from typing import Callable, Any, get_type_hints
from enum import Enum


class ToolCallerType(Enum):
    """工具调用者类型"""
    DIRECT = "direct"  # Claude 直接调用
    CODE_EXECUTION = "code_execution"  # 从代码执行环境调用
    BOTH = "both"  # 两种方式都可以


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    func: Callable
    input_schema: dict
    output_description: str = ""
    allowed_callers: list[ToolCallerType] = field(
        default_factory=lambda: [ToolCallerType.CODE_EXECUTION]
    )
    timeout_seconds: float = 30.0

    def to_claude_tool_schema(self) -> dict:
        """转换为 Claude API 的工具格式"""
        return {
            "name": self.name,
            "description": self.description + (
                f"\n\nOutput format: {self.output_description}"
                if self.output_description else ""
            ),
            "input_schema": self.input_schema
        }

    def to_stub_signature(self) -> str:
        """生成在沙箱中使用的函数签名文档"""
        params = []
        props = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])

        for param_name, param_info in props.items():
            param_type = param_info.get("type", "any")
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
                "array": "list",
                "object": "dict"
            }
            py_type = type_map.get(param_type, "Any")

            if param_name in required:
                params.append(f"{param_name}: {py_type}")
            else:
                params.append(f"{param_name}: {py_type} = None")

        return f"async def {self.name}({', '.join(params)}) -> Any"


class ToolRegistry:
    """
    工具注册表

    管理所有可被 programmatic tool calling 使用的工具
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict | None = None,
        output_description: str = "",
        allowed_callers: list[ToolCallerType] | None = None,
        timeout_seconds: float = 30.0
    ) -> Callable:
        """
        装饰器：注册一个工具函数

        Usage:
            @registry.register(
                description="Query the database",
                output_description="Returns list of dict with 'id' and 'name' keys"
            )
            def query_database(sql: str) -> list[dict]:
                ...
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

            # 自动推断 input_schema
            schema = input_schema
            if schema is None:
                schema = self._infer_schema_from_function(func)

            tool = Tool(
                name=tool_name,
                description=tool_desc,
                func=func,
                input_schema=schema,
                output_description=output_description,
                allowed_callers=allowed_callers or [ToolCallerType.CODE_EXECUTION],
                timeout_seconds=timeout_seconds
            )

            self._tools[tool_name] = tool
            return func

        return decorator

    def register_tool(self, tool: Tool) -> None:
        """直接注册一个 Tool 对象"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """获取工具"""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """获取所有工具"""
        return list(self._tools.values())

    def get_code_execution_tools(self) -> list[Tool]:
        """获取可从代码执行环境调用的工具"""
        return [
            tool for tool in self._tools.values()
            if ToolCallerType.CODE_EXECUTION in tool.allowed_callers
            or ToolCallerType.BOTH in tool.allowed_callers
        ]

    def get_direct_tools(self) -> list[Tool]:
        """获取可直接调用的工具"""
        return [
            tool for tool in self._tools.values()
            if ToolCallerType.DIRECT in tool.allowed_callers
            or ToolCallerType.BOTH in tool.allowed_callers
        ]

    def execute(self, name: str, arguments: dict) -> Any:
        """执行工具"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return tool.func(**arguments)

    async def execute_async(self, name: str, arguments: dict) -> Any:
        """异步执行工具"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        if inspect.iscoroutinefunction(tool.func):
            return await tool.func(**arguments)
        else:
            return tool.func(**arguments)

    def generate_tools_documentation(self) -> str:
        """生成工具文档，用于 system prompt"""
        docs = []
        for tool in self.get_code_execution_tools():
            signature = tool.to_stub_signature()
            docs.append(f"""
### {tool.name}
```python
{signature}
```
**Description:** {tool.description}
{f"**Output:** {tool.output_description}" if tool.output_description else ""}
""")
        return "\n".join(docs)

    def _infer_schema_from_function(self, func: Callable) -> dict:
        """从函数签名自动推断 JSON Schema"""
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}

        properties = {}
        required = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            param_type = hints.get(param_name, str)
            json_type = type_map.get(param_type, "string")

            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter: {param_name}"
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def to_json(self) -> str:
        """序列化工具定义（用于传递给沙箱）"""
        tools_data = []
        for tool in self.get_code_execution_tools():
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_description": tool.output_description
            })
        return json.dumps(tools_data)
