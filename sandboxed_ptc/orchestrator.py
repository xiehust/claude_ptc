"""
Programmatic Tool Calling 调度器

负责：
1. 与 Claude API 通信
2. 管理代码执行工具
3. 协调沙箱执行和工具调用
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from enum import Enum

from .tool_registry import ToolRegistry, ToolCallerType
from .sandbox import SandboxExecutor, SandboxConfig, ExecutionResult
from .exceptions import SandboxError, CodeExecutionError

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式"""
    PROGRAMMATIC = "programmatic"  # 通过代码执行调用工具
    DIRECT = "direct"  # 直接调用工具
    HYBRID = "hybrid"  # 混合模式


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str
    content: Any


@dataclass
class OrchestratorConfig:
    """调度器配置"""
    model: str = "claude-sonnet-4-5-20250514"
    max_tokens: int = 4096
    execution_mode: ExecutionMode = ExecutionMode.PROGRAMMATIC
    max_iterations: int = 10  # 防止无限循环
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)


class ProgrammaticToolOrchestrator:
    """
    Programmatic Tool Calling 调度器

    实现类似 Anthropic 官方 Programmatic Tool Calling 的机制，
    但使用自定义沙箱执行环境。

    Usage:
        orchestrator = ProgrammaticToolOrchestrator(api_key="...")

        @orchestrator.tool_registry.register(
            description="Query the database"
        )
        def query_database(sql: str) -> list[dict]:
            return db.execute(sql)

        result = await orchestrator.run("分析过去一个月的销售数据")
        print(result)
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: OrchestratorConfig | None = None
    ):
        self.api_key = api_key
        self.config = config or OrchestratorConfig()
        self.tool_registry = ToolRegistry()
        self.sandbox = SandboxExecutor(
            self.tool_registry,
            self.config.sandbox_config
        )
        self._client = None
        self._conversation_history: list[ConversationMessage] = []

    @property
    def client(self):
        """懒加载 Anthropic 客户端"""
        if self._client is None:
            try:
                import anthropic
                if self.api_key:
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                else:
                    self._client = anthropic.AnthropicBedrock()
            except ImportError:
                raise ImportError(
                    "Anthropic SDK not installed. Run: pip install anthropic"
                )
        return self._client

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tools_doc = self.tool_registry.generate_tools_documentation()

        return f"""你是一个强大的 AI 助手，可以通过代码执行环境来完成复杂任务。

## 代码执行环境

你有一个 Python 代码执行环境，其中预定义了以下异步工具函数：

{tools_doc}

## 使用方法

当需要执行多步骤任务或处理数据时，使用 `execute_code` 工具编写 Python 代码。

关键规则：
1. **所有工具调用必须使用 `await`**，例如：`result = await query_database(sql="SELECT * FROM users")`
2. **使用 `print()` 输出结果**，这是你获取执行结果的唯一方式
3. 可以在代码中进行数据处理、过滤、聚合、条件判断等
4. 代码执行完成后，你会看到 print 输出的内容
5. 如果需要多次调用同一工具（如循环查询），写成循环比多次单独调用更高效

## 最佳实践

1. **批量处理**：将多个相关操作写在一段代码中
```python
results = {{}}
for region in ["East", "West", "Central"]:
    data = await query_database(sql=f"SELECT SUM(revenue) FROM sales WHERE region='{{region}}'")
    results[region] = data
print(f"Regional revenue: {{results}}")
```

2. **数据过滤**：先获取数据，在代码中过滤
```python
all_logs = await fetch_logs(server_id="main")
errors = [log for log in all_logs if "ERROR" in log]
print(f"Found {{len(errors)}} errors")
for err in errors[-5:]:  # 只显示最后 5 条
    print(err)
```

3. **条件逻辑**：根据中间结果决定下一步
```python
file_info = await get_file_info(path="/data/large.csv")
if file_info["size"] > 1000000:
    summary = await get_file_summary(path="/data/large.csv")
else:
    content = await read_file(path="/data/large.csv")
    summary = content
print(summary)
```

4. **提前终止**：找到所需结果后立即停止
```python
servers = ["us-east", "eu-west", "ap-south"]
for server in servers:
    status = await check_health(server_id=server)
    if status["healthy"]:
        print(f"Found healthy server: {{server}}")
        break
```

## 重要提示

- 代码在安全沙箱中执行，无法访问网络或文件系统
- 只有通过预定义的工具函数才能与外部系统交互
- 执行结果只包含 print 输出，不会自动返回变量值
"""

    def _create_code_execution_tool(self) -> dict:
        """创建代码执行工具定义"""
        return {
            "name": "execute_code",
            "description": """在沙箱环境中执行 Python 代码。

代码可以调用预定义的异步工具函数来完成任务。
使用 print() 输出你需要看到的结果。

适用场景：
- 需要多次调用工具（如循环遍历）
- 需要对工具返回的数据进行处理、过滤、聚合
- 需要根据中间结果做条件判断
- 需要批量处理多个相似任务

注意：所有工具调用必须使用 await，例如：
result = await query_database(sql="SELECT * FROM users")
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的 Python 代码。使用 await 调用工具函数，使用 print() 输出结果。"
                    }
                },
                "required": ["code"]
            }
        }

    def _get_all_tools(self) -> list[dict]:
        """获取所有工具定义"""
        tools = [self._create_code_execution_tool()]

        # 添加可直接调用的工具
        for tool in self.tool_registry.get_direct_tools():
            tools.append(tool.to_claude_tool_schema())

        return tools

    async def run(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None
    ) -> str:
        """
        执行用户请求

        Args:
            user_message: 用户输入
            conversation_history: 可选的对话历史

        Returns:
            Claude 的最终回复
        """
        # 初始化对话历史
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": user_message})

        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            # 调用 Claude API
            logger.info(f"Iteration {iteration}: Calling Claude API")

            response = self.client.beta.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self._build_system_prompt(),
                messages=messages,
                tools=self._get_all_tools()
            )

            # 检查停止原因
            if response.stop_reason == "end_turn":
                # 完成，提取文本回复
                text_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text
                return text_content

            elif response.stop_reason == "tool_use":
                # 需要执行工具
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_content.append({
                            "type": "text",
                            "text": block.text
                        })
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })

                        # 执行工具
                        tool_result = await self._execute_tool(
                            block.name,
                            block.input,
                            block.id
                        )
                        tool_results.append(tool_result)

                # 添加助手消息和工具结果
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

            else:
                # 未知停止原因
                logger.warning(f"Unknown stop reason: {response.stop_reason}")
                break

        raise RuntimeError(f"Exceeded maximum iterations ({self.config.max_iterations})")

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str
    ) -> dict:
        """执行工具并返回结果"""

        if tool_name == "execute_code":
            # 在沙箱中执行代码
            code = tool_input.get("code", "")
            logger.info(f"Executing code in sandbox:\n{code}")

            try:
                result = await self.sandbox.execute(code)

                if result.success:
                    content = result.stdout
                    if not content:
                        content = "(代码执行成功，但没有 print 输出)"
                else:
                    content = f"执行错误: {result.stderr}"

                logger.info(f"Sandbox execution result: {content[:200]}...")

                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content
                }

            except SandboxError as e:
                logger.error(f"Sandbox error: {e}")
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": f"沙箱执行错误: {str(e)}",
                    "is_error": True
                }

        else:
            # 直接调用工具
            try:
                result = await self.tool_registry.execute_async(
                    tool_name,
                    tool_input
                )

                # 序列化结果
                if isinstance(result, (dict, list)):
                    content = json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    content = str(result)

                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content
                }

            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": f"工具执行错误: {str(e)}",
                    "is_error": True
                }

    async def run_streaming(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """
        流式执行用户请求

        Args:
            user_message: 用户输入
            conversation_history: 可选的对话历史

        Yields:
            流式响应文本片段
        """
        # 初始化对话历史
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": user_message})

        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            # 流式调用 Claude API
            with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self._build_system_prompt(),
                messages=messages,
                tools=self._get_all_tools()
            ) as stream:
                collected_content = []
                current_tool_use = None

                for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            pass  # 文本块开始
                        elif event.content_block.type == "tool_use":
                            current_tool_use = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": ""
                            }

                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield event.delta.text
                            collected_content.append({
                                "type": "text",
                                "text": event.delta.text
                            })
                        elif hasattr(event.delta, "partial_json"):
                            if current_tool_use:
                                current_tool_use["input"] += event.delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool_use:
                            # 解析工具输入
                            try:
                                current_tool_use["input"] = json.loads(
                                    current_tool_use["input"]
                                )
                            except json.JSONDecodeError:
                                current_tool_use["input"] = {}

                            collected_content.append({
                                "type": "tool_use",
                                "id": current_tool_use["id"],
                                "name": current_tool_use["name"],
                                "input": current_tool_use["input"]
                            })
                            current_tool_use = None

                # 获取最终消息
                final_message = stream.get_final_message()

                if final_message.stop_reason == "end_turn":
                    return

                elif final_message.stop_reason == "tool_use":
                    # 执行工具
                    tool_results = []

                    for block in collected_content:
                        if block.get("type") == "tool_use":
                            yield f"\n[执行工具: {block['name']}...]\n"

                            result = await self._execute_tool(
                                block["name"],
                                block["input"],
                                block["id"]
                            )
                            tool_results.append(result)

                    # 更新对话历史
                    messages.append({
                        "role": "assistant",
                        "content": collected_content
                    })
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

    def reset_conversation(self) -> None:
        """重置对话历史"""
        self._conversation_history.clear()

    def register_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict | None = None,
        output_description: str = "",
        allowed_callers: list[ToolCallerType] | None = None,
        timeout_seconds: float = 30.0
    ):
        """
        装饰器：注册工具

        快捷方式，等同于 self.tool_registry.register(...)
        """
        return self.tool_registry.register(
            name=name,
            description=description,
            input_schema=input_schema,
            output_description=output_description,
            allowed_callers=allowed_callers,
            timeout_seconds=timeout_seconds
        )
