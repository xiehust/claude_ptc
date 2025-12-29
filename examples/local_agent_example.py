#!/usr/bin/env python3
"""
Local Sandbox Agent Example

This example demonstrates how to build an AI Agent using:
- AnthropicBedrock client for Claude API access
- Local sandbox for code execution (no Docker required)
- Team Expense API tools for real-world business analysis

Features:
- No Docker dependency - code runs in local Python process
- Faster startup (no container overhead)
- Session reuse for state persistence
- Direct tool calls + Programmatic tool calling

Warning:
- Local sandbox provides NO security isolation
- Only use with trusted code or in development environments
- For production with untrusted code, use Docker-based SandboxExecutor

Usage:
    # Basic demo
    python local_agent_example.py

    # Session reuse demo
    python local_agent_example.py --session-reuse

    # Interactive mode
    python local_agent_example.py -i

    # Verbose logging
    python local_agent_example.py -v

    # Disable visualization
    python local_agent_example.py --no-viz

Requirements:
    pip install anthropic
    AWS credentials configured (for Bedrock)
"""

import asyncio
import json
import logging
from dataclasses import dataclass

# Add project root to path for imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sandboxed_ptc import ToolRegistry, ToolCallerType
from sandboxed_ptc.local_sandbox import LocalSandboxExecutor, LocalSandboxConfig, ExecutionResult

# Import team expense API tools
from utils.team_expense_api import get_team_members, get_expenses, get_custom_budget

# Import visualizer
from utils.visualize import visualize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================
# 1. Mock Weather API (Direct Call Tool)
# ============================================================

def get_weather(city: str, units: str = "celsius") -> str:
    """
    Mock weather API - returns simulated weather data for a city.
    This is a DIRECT CALL tool - Claude calls it directly, not via code execution.
    """
    import random
    random.seed(42)  # Consistent results for demo

    weather_data = {
        "beijing": {"temp_c": 15, "condition": "Partly Cloudy", "humidity": 45, "wind_kph": 12},
        "shanghai": {"temp_c": 22, "condition": "Sunny", "humidity": 60, "wind_kph": 8},
        "new york": {"temp_c": 18, "condition": "Cloudy", "humidity": 55, "wind_kph": 15},
        "london": {"temp_c": 12, "condition": "Rainy", "humidity": 80, "wind_kph": 20},
        "tokyo": {"temp_c": 20, "condition": "Clear", "humidity": 50, "wind_kph": 10},
    }

    city_lower = city.lower().strip()
    if city_lower in weather_data:
        data = weather_data[city_lower]
    else:
        data = {
            "temp_c": random.randint(5, 35),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Clear"]),
            "humidity": random.randint(30, 90),
            "wind_kph": random.randint(5, 30)
        }

    if units.lower() == "fahrenheit":
        temp = data["temp_c"] * 9/5 + 32
        temp_unit = "F"
    else:
        temp = data["temp_c"]
        temp_unit = "C"

    result = {
        "city": city.title(),
        "temperature": f"{temp:.1f}{temp_unit}",
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "wind": f"{data['wind_kph']} km/h"
    }

    return json.dumps(result, ensure_ascii=False)


# ============================================================
# 2. Tool Configurations
# ============================================================

TOOL_CONFIGS = [
    {
        "name": "get_team_members",
        "description": 'Returns a list of team members for a given department. Returns JSON array of team member objects with id, name, role, level, and contact info. Available departments: engineering, sales, marketing.',
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "The department name (case-insensitive)"
                }
            },
            "required": ["department"]
        }
    },
    {
        "name": "get_expenses",
        "description": "Returns expense line items for an employee in a specific quarter. Returns JSON array of expense objects with date, category, amount, status, etc. Categories: travel, lodging, meals, software, equipment, conference, office, internet. Only status='approved' expenses count toward budget limits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The unique employee identifier"
                },
                "quarter": {
                    "type": "string",
                    "description": "Quarter identifier: Q1, Q2, Q3, or Q4"
                }
            },
            "required": ["employee_id", "quarter"]
        }
    },
    {
        "name": "get_custom_budget",
        "description": 'Get the custom quarterly travel budget for a specific employee. Most employees have standard $5,000 quarterly travel budget. Some have custom exceptions. Returns JSON object with user_id, has_custom_budget, travel_budget, reason, currency.',
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The unique employee identifier"
                }
            },
            "required": ["user_id"]
        }
    }
]

DIRECT_TOOL_CONFIGS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units (default: celsius)"
                }
            },
            "required": ["city"]
        }
    }
]

TOOL_FUNCTIONS = {
    "get_team_members": get_team_members,
    "get_expenses": get_expenses,
    "get_custom_budget": get_custom_budget,
}

DIRECT_TOOL_FUNCTIONS = {
    "get_weather": get_weather,
}


# ============================================================
# 3. Agent Configuration
# ============================================================

@dataclass
class AgentConfig:
    """Agent configuration"""
    model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_tokens: int = 8192
    max_iterations: int = 15
    enable_visualization: bool = True


# ============================================================
# 4. Local Sandbox Agent
# ============================================================

class LocalSandboxAgent:
    """
    AI Agent using AnthropicBedrock with Local Sandbox for Code Execution

    This version uses a local sandbox for code execution (no Docker required).
    Suitable for development, testing, or environments without Docker.

    Features:
    - No Docker dependency
    - Fast startup
    - Session reuse support
    - Multi-turn conversation
    - Direct + Programmatic tool calling

    Warning:
    - No security isolation
    - Only use with trusted code
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        sandbox_config: LocalSandboxConfig | None = None,
        enable_session_reuse: bool = False
    ):
        self.config = config or AgentConfig()
        self.sandbox_config = sandbox_config or LocalSandboxConfig(
            timeout_seconds=60.0,
            enable_session_reuse=enable_session_reuse,
            session_timeout_seconds=270.0,
        )
        self.enable_session_reuse = enable_session_reuse
        self.tool_registry = ToolRegistry()
        self._client = None
        self._conversation_history: list[dict] = []
        self._sandbox: LocalSandboxExecutor | None = None
        self._visualizer = None
        self._current_session_id: str | None = None

        # Register tools
        self._register_tools_from_config()

        # Initialize visualizer
        if self.config.enable_visualization:
            self._visualizer = visualize(auto_show=True)

        logger.info(f"Initialized LocalSandboxAgent with model: {self.config.model}")
        if enable_session_reuse:
            logger.info("Session reuse mode enabled")

    def _register_tools_from_config(self) -> None:
        """Register tools from configs"""
        # Code execution tools
        for tool_config in TOOL_CONFIGS:
            name = tool_config["name"]
            func = TOOL_FUNCTIONS.get(name)
            if func:
                self.tool_registry.register(
                    name=name,
                    description=tool_config["description"],
                    input_schema=tool_config["input_schema"],
                    allowed_callers=[ToolCallerType.CODE_EXECUTION]
                )(func)
                logger.debug(f"Registered tool: {name}")

        # Direct call tools
        for tool_config in DIRECT_TOOL_CONFIGS:
            name = tool_config["name"]
            func = DIRECT_TOOL_FUNCTIONS.get(name)
            if func:
                self.tool_registry.register(
                    name=name,
                    description=tool_config["description"],
                    input_schema=tool_config["input_schema"],
                    allowed_callers=[ToolCallerType.DIRECT]
                )(func)
                logger.debug(f"Registered direct tool: {name}")

    async def __aenter__(self):
        """Async context manager entry"""
        logger.info("Local sandbox agent ready")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._current_session_id and self._sandbox:
            await self._sandbox.close_session(self._current_session_id)
            logger.info(f"Closed session: {self._current_session_id}")
        if self._sandbox:
            self._sandbox.stop_cleanup_task()
        logger.info("Local sandbox agent closed")

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    @property
    def active_sessions(self) -> dict:
        if self._sandbox:
            return self._sandbox.active_sessions
        return {}

    @property
    def sandbox(self) -> LocalSandboxExecutor:
        if self._sandbox is None:
            self._sandbox = LocalSandboxExecutor(self.tool_registry, self.sandbox_config)
        return self._sandbox

    @property
    def client(self):
        if self._client is None:
            from anthropic import AnthropicBedrock
            self._client = AnthropicBedrock()
            logger.info("AnthropicBedrock client initialized")
        return self._client

    def _build_system_prompt(self) -> str:
        """Build the system prompt"""
        tools_doc = self.tool_registry.generate_tools_documentation()

        session_note = ""
        if self.enable_session_reuse:
            session_note = """
## Session State Persistence
The code execution environment supports state persistence between executions.
Variables defined in one code block are available in subsequent executions.
"""
        else:
            session_note = """
## Stateless Execution
Each code execution starts fresh. Variables do NOT persist between executions.
Complete all work in a single code block when possible.
"""

        return f"""You are an AI assistant with a Python code execution environment.

## Code Execution Environment

You can use the `execute_code` tool to run Python code. Within your code, call these async tool functions:

{tools_doc}

## Usage Rules

1. Use `await` for all tool calls: `result = await get_team_members(department="engineering")`
2. Use `print()` to output results - this is the only way to see execution output
3. Use `json.loads()` to parse JSON strings returned by tools
4. Use `asyncio.gather()` for parallel execution when calling tools for multiple items
{session_note}
## Best Practices

1. **Parallel Execution** - Use asyncio.gather() for multiple independent calls:
```python
import asyncio
import json

employee_ids = ["ENG001", "ENG002", "ENG003"]
tasks = [get_expenses(employee_id=eid, quarter="Q3") for eid in employee_ids]
results = await asyncio.gather(*tasks)

for eid, data in zip(employee_ids, results):
    expenses = json.loads(data)
    print(f"{{eid}}: {{len(expenses)}} expenses")
```

2. **Data Processing** - Fetch and analyze in one block:
```python
import json

members_data = await get_team_members(department="engineering")
members = json.loads(members_data)

for member in members:
    print(f"{{member['name']}} - {{member['role']}}")
```

3. **Conditional Logic** - Handle branching within one execution:
```python
import json

budget_data = await get_custom_budget(user_id="ENG001")
budget = json.loads(budget_data)

if budget["has_custom_budget"]:
    print(f"Custom budget: ${{budget['travel_budget']}}")
else:
    print("Standard budget: $5,000")
```
"""

    def _create_execute_code_tool(self) -> dict:
        """Create the execute_code tool definition"""
        tools_doc = self.tool_registry.generate_tools_documentation()
        return {
            "name": "execute_code",
            "description": f"""Execute Python code in the local sandbox.

Available async tool functions:
{tools_doc}

Use await for tool calls, print() for output.
Example: result = await get_team_members(department="engineering")
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }

    async def _execute_code(self, code: str) -> dict:
        """Execute code in the local sandbox"""
        logger.info(f"Executing code:\n{code}")

        if self.enable_session_reuse:
            result = await self.sandbox.execute(
                code,
                session_id=self._current_session_id,
                reuse_session=True
            )
            if isinstance(result, tuple):
                exec_result, session_id = result
                if self._current_session_id is None:
                    self._current_session_id = session_id
                    logger.info(f"Created session: {session_id}")
            else:
                exec_result = result
        else:
            exec_result = await self.sandbox.execute(code)

        if exec_result.success:
            content = exec_result.stdout or "(Code executed successfully, no output)"
        else:
            content = f"Execution Error: {exec_result.stderr}"

        logger.info(f"Result: {content[:500]}...")
        return {"success": exec_result.success, "output": content}

    async def chat(
        self,
        message: str,
        reset_history: bool = False
    ) -> tuple[str, int, int]:
        """
        Send a message and get a response

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        if reset_history:
            self._conversation_history.clear()
            logger.info("Conversation history cleared")

        self._conversation_history.append({
            "role": "user",
            "content": message
        })

        total_input_tokens = 0
        total_output_tokens = 0
        iteration = 0

        # Build tools list
        tools = [self._create_execute_code_tool()]
        for tool_config in DIRECT_TOOL_CONFIGS:
            tools.append({
                "name": tool_config["name"],
                "description": tool_config["description"],
                "input_schema": tool_config["input_schema"]
            })

        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self._build_system_prompt(),
                messages=self._conversation_history,
                tools=tools
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            if self._visualizer:
                self._visualizer.capture(response)

            logger.info(f"Stop reason: {response.stop_reason}")

            if response.stop_reason == "end_turn":
                text_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text

                self._conversation_history.append({
                    "role": "assistant",
                    "content": text_content
                })

                return text_content, total_input_tokens, total_output_tokens

            elif response.stop_reason == "tool_use":
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

                        logger.info(f"Tool call: {block.name}")

                        if block.name == "execute_code":
                            code = block.input.get("code", "")
                            result = await self._execute_code(code)
                            content = result["output"]
                        elif block.name in DIRECT_TOOL_FUNCTIONS:
                            try:
                                func = DIRECT_TOOL_FUNCTIONS[block.name]
                                content = func(**block.input)
                            except Exception as e:
                                content = f"Error: {str(e)}"
                        else:
                            content = f"Unknown tool: {block.name}"

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content
                        })

                self._conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                self._conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })

            else:
                logger.warning(f"Unknown stop reason: {response.stop_reason}")
                break

        raise RuntimeError(f"Exceeded max iterations ({self.config.max_iterations})")

    def reset(self) -> None:
        """Reset conversation history"""
        self._conversation_history.clear()

    @property
    def history(self) -> list[dict]:
        return self._conversation_history.copy()


# ============================================================
# 5. Demo Functions
# ============================================================

async def run_demo(enable_visualization: bool = True):
    """Run the local sandbox agent demonstration"""
    print("=" * 70)
    print("Local Sandbox Agent Demo (No Docker Required)")
    print("=" * 70)

    config = AgentConfig(
        model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=8192,
        max_iterations=15,
        enable_visualization=enable_visualization
    )

    sandbox_config = LocalSandboxConfig(
        timeout_seconds=60.0,
        enable_session_reuse=False,
    )

    async with LocalSandboxAgent(config=config, sandbox_config=sandbox_config) as agent:
        print(f"\nLocal Sandbox Agent created with {len(agent.tool_registry.get_all())} tools:")
        for tool in agent.tool_registry.get_all():
            callers = [c.value for c in tool.allowed_callers]
            print(f"  - {tool.name} ({', '.join(callers)})")

        print("\nLocal Sandbox (no Docker):")
        print(f"  - Timeout: {sandbox_config.timeout_seconds}s")
        print(f"  - Session Reuse: {sandbox_config.enable_session_reuse}")

        # Demo query
        print("\n" + "=" * 70)
        query = "Which engineering team members exceeded their Q3 travel budget? Standard quarterly travel budget is $5,000. For anyone who exceeded, check if they have a custom budget exception."
        print(f"Query: {query}")
        print("=" * 70)

        response, input_tokens, output_tokens = await agent.chat(query)
        print("\n--- Agent Response ---")
        print(response)
        print(f"\nToken Usage: {input_tokens} in / {output_tokens} out")


async def run_session_demo(enable_visualization: bool = True):
    """Demo with session reuse"""
    print("=" * 70)
    print("Local Sandbox Session Reuse Demo")
    print("=" * 70)

    config = AgentConfig(
        model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        enable_visualization=enable_visualization
    )

    sandbox_config = LocalSandboxConfig(
        timeout_seconds=60.0,
        enable_session_reuse=True,
        session_timeout_seconds=270.0,
    )

    async with LocalSandboxAgent(
        config=config,
        sandbox_config=sandbox_config,
        enable_session_reuse=True
    ) as agent:
        print("Session reuse enabled - state persists between executions\n")

        # First query
        print("=" * 70)
        query1 = "Get the engineering team members and count them. Store the data for later."
        print(f"Query 1: {query1}")
        print("=" * 70)

        response1, t1_in, t1_out = await agent.chat(query1)
        print(f"\n{response1}")
        print(f"Session: {agent.current_session_id}")

        # Second query (uses same session)
        print("\n" + "=" * 70)
        query2 = "Using the team data from before, which members are senior level or above?"
        print(f"Query 2: {query2}")
        print("=" * 70)

        response2, t2_in, t2_out = await agent.chat(query2)
        print(f"\n{response2}")

        # Show session info
        print("\n--- Active Sessions ---")
        for sid, info in agent.active_sessions.items():
            print(f"  {sid}: {info['execution_count']} executions")


async def interactive_mode(enable_visualization: bool = True, session_reuse: bool = False):
    """Interactive chat mode"""
    print("=" * 70)
    print("Local Sandbox Agent - Interactive Mode")
    print("Type 'quit' to exit, 'reset' to clear history")
    if session_reuse:
        print("Session reuse: ENABLED (variables persist)")
    print("=" * 70)

    config = AgentConfig(enable_visualization=enable_visualization)
    sandbox_config = LocalSandboxConfig(enable_session_reuse=session_reuse)

    async with LocalSandboxAgent(
        config=config,
        sandbox_config=sandbox_config,
        enable_session_reuse=session_reuse
    ) as agent:
        print(f"\nTools: {[t.name for t in agent.tool_registry.get_all()]}")
        print("\nExample queries:")
        print("  - 'Get all engineering team members'")
        print("  - 'Analyze Q3 expenses for sales team'")
        print("  - 'What's the weather in Tokyo?'\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                elif user_input.lower() == "reset":
                    agent.reset()
                    print("History cleared.\n")
                    continue
                elif not user_input:
                    continue

                response, in_tok, out_tok = await agent.chat(user_input)
                print(f"\nAgent: {response}")
                print(f"[Tokens: {in_tok} in / {out_tok} out]\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.exception("Error")
                print(f"\nError: {e}\n")


async def run_low_level_demo():
    """Low-level LocalSandboxExecutor demo"""
    print("=" * 70)
    print("Low-Level LocalSandboxExecutor Demo")
    print("=" * 70)

    registry = ToolRegistry()

    @registry.register(name="add", description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    @registry.register(name="multiply", description="Multiply two numbers")
    def multiply(a: int, b: int) -> int:
        return a * b

    config = LocalSandboxConfig(
        timeout_seconds=30.0,
        enable_session_reuse=True,
    )

    executor = LocalSandboxExecutor(registry, config)

    try:
        # Execution 1 - create session
        print("\n--- Execution 1: Set variables ---")
        code1 = """
x = 10
y = 20
print(f"x = {x}, y = {y}")
"""
        result1, session_id = await executor.execute(code1, reuse_session=True)
        print(f"Output: {result1.stdout}")
        print(f"Session: {session_id}")

        # Execution 2 - variables persist
        print("\n--- Execution 2: Use persisted variables ---")
        code2 = """
z = x + y
print(f"z = x + y = {z}")
"""
        result2, _ = await executor.execute(code2, session_id=session_id)
        print(f"Output: {result2.stdout}")

        # Execution 3 - tool calls
        print("\n--- Execution 3: Tool calls ---")
        code3 = """
sum_result = await add(a=z, b=100)
product = await multiply(a=x, b=y)
print(f"add({z}, 100) = {sum_result}")
print(f"multiply({x}, {y}) = {product}")
"""
        result3, _ = await executor.execute(code3, session_id=session_id)
        print(f"Output: {result3.stdout}")
        print(f"Tool calls: {result3.tool_calls_count}")

        # Show sessions
        print("\n--- Sessions ---")
        for sid, info in executor.active_sessions.items():
            print(f"  {sid}: {info['execution_count']} executions")

    finally:
        await executor.close_all_sessions()
        executor.stop_cleanup_task()


# ============================================================
# 6. Main Entry Point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local Sandbox Agent Demo")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("-s", "--session-reuse", action="store_true", help="Enable session reuse")
    parser.add_argument("--low-level", action="store_true", help="Low-level executor demo")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    enable_viz = not args.no_viz

    if args.low_level:
        asyncio.run(run_low_level_demo())
    elif args.interactive:
        asyncio.run(interactive_mode(enable_viz, args.session_reuse))
    elif args.session_reuse:
        asyncio.run(run_session_demo(enable_viz))
    else:
        asyncio.run(run_demo(enable_viz))


if __name__ == "__main__":
    main()
