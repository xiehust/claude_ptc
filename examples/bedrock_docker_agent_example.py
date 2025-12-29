#!/usr/bin/env python3
"""
Bedrock Docker Sandbox Agent Example

This example demonstrates how to build an AI Agent using:
- AnthropicBedrock client for Claude API access
- Docker sandbox for secure, isolated code execution
- Team Expense API tools for real-world business analysis

Features:
- Docker-based sandboxed code execution
- Network isolation and resource limits
- Multi-turn conversation with history
- Direct tool calls + Programmatic tool calling
- **Container session reuse** (like official Anthropic PTC):
  - Keep container running across multiple code executions
  - Preserve state (variables) between executions
  - Session timeout (default 4.5 minutes, configurable)
  - Automatic expired session cleanup

Usage:
    # Basic demo (single container per execution)
    python bedrock_docker_agent_example.py

    # Session reuse demo (container persists across executions)
    python bedrock_docker_agent_example.py --session-reuse

    # Low-level session API demo
    python bedrock_docker_agent_example.py --low-level

    # Interactive mode
    python bedrock_docker_agent_example.py -i

    # Interactive mode with session reuse
    python bedrock_docker_agent_example.py -i --session-reuse

    # Verbose logging
    python bedrock_docker_agent_example.py -v

Requirements:
    pip install anthropic docker
    AWS credentials configured
    Docker daemon running
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

from sandboxed_ptc import ToolRegistry, ToolCallerType, SandboxSession, ExecutionResult
from sandboxed_ptc.sandbox import SandboxExecutor, SandboxConfig

# Import team expense API tools
from utils.team_expense_api import get_team_members, get_expenses, get_custom_budget

# Import visualizer
from utils.visualize import visualize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)
logging.getLogger('docker').setLevel(logging.WARNING)
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

    # Mock weather data for different cities
    weather_data = {
        "beijing": {"temp_c": 15, "condition": "Partly Cloudy", "humidity": 45, "wind_kph": 12},
        "shanghai": {"temp_c": 22, "condition": "Sunny", "humidity": 60, "wind_kph": 8},
        "new york": {"temp_c": 18, "condition": "Cloudy", "humidity": 55, "wind_kph": 15},
        "london": {"temp_c": 12, "condition": "Rainy", "humidity": 80, "wind_kph": 20},
        "tokyo": {"temp_c": 20, "condition": "Clear", "humidity": 50, "wind_kph": 10},
        "paris": {"temp_c": 14, "condition": "Overcast", "humidity": 65, "wind_kph": 18},
        "sydney": {"temp_c": 25, "condition": "Sunny", "humidity": 40, "wind_kph": 5},
        "san francisco": {"temp_c": 16, "condition": "Foggy", "humidity": 75, "wind_kph": 22},
    }

    city_lower = city.lower().strip()

    if city_lower in weather_data:
        data = weather_data[city_lower]
    else:
        # Generate random weather for unknown cities
        data = {
            "temp_c": random.randint(5, 35),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Clear", "Windy"]),
            "humidity": random.randint(30, 90),
            "wind_kph": random.randint(5, 30)
        }

    # Convert temperature if needed
    if units.lower() == "fahrenheit":
        temp = data["temp_c"] * 9/5 + 32
        temp_unit = "°F"
    else:
        temp = data["temp_c"]
        temp_unit = "°C"

    result = {
        "city": city.title(),
        "temperature": f"{temp:.1f}{temp_unit}",
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "wind": f"{data['wind_kph']} km/h",
        "units": units.lower()
    }

    return json.dumps(result, ensure_ascii=False)


# ============================================================
# 2. Tool Configurations
# ============================================================

# Code execution tool configurations (called via execute_code)
TOOL_CONFIGS = [
    {
        "name": "get_team_members",
        "description": 'Returns a list of team members for a given department. Each team member includes their ID, name, role, level (junior, mid, senior, staff, principal), and contact information. Use this to get a list of people whose expenses you want to analyze. Available departments are: engineering, sales, and marketing.\n\nRETURN FORMAT: Returns a JSON string containing an ARRAY of team member objects (not wrapped in an outer object). Parse with json.loads() to get a list. Example: [{"id": "ENG001", "name": "Alice", ...}, {"id": "ENG002", ...}]',
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "The department name. Case-insensitive.",
                }
            },
            "required": ["department"],
        },
    },
    {
        "name": "get_expenses",
        "description": "Returns all expense line items for a given employee in a specific quarter. Each expense includes extensive metadata: date, category, description, amount (in USD), currency, status (approved, pending, rejected), receipt URL, approval chain, merchant name and location, payment method, and project codes. An employee may have 20-50+ expense line items per quarter, and each line item contains substantial metadata for audit and compliance purposes. Categories include: 'travel' (flights, trains, rental cars, taxis, parking), 'lodging' (hotels, airbnb), 'meals', 'software', 'equipment', 'conference', 'office', and 'internet'. IMPORTANT: Only expenses with status='approved' should be counted toward budget limits.\n\nRETURN FORMAT: Returns a JSON string containing an ARRAY of expense objects (not wrapped in an outer object with an 'expenses' key). Parse with json.loads() to get a list directly. Example: [{\"expense_id\": \"ENG001_Q3_001\", \"amount\": 1250.50, \"category\": \"travel\", ...}, {...}]",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The unique employee identifier",
                },
                "quarter": {
                    "type": "string",
                    "description": "Quarter identifier: 'Q1', 'Q2', 'Q3', or 'Q4'",
                },
            },
            "required": ["employee_id", "quarter"],
        },
    },
    {
        "name": "get_custom_budget",
        "description": 'Get the custom quarterly travel budget for a specific employee. Most employees have a standard $5,000 quarterly travel budget. However, some employees have custom budget exceptions based on their role requirements. This function checks if a specific employee has a custom budget assigned.\n\nRETURN FORMAT: Returns a JSON string containing a SINGLE OBJECT (not an array). Parse with json.loads() to get a dict. Example: {"user_id": "ENG001", "has_custom_budget": false, "travel_budget": 5000, "reason": "Standard", "currency": "USD"}',
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The unique employee identifier",
                }
            },
            "required": ["user_id"],
        },
    },
]

# Direct call tool configurations (Claude calls directly, not via code execution)
DIRECT_TOOL_CONFIGS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a city. This tool provides real-time weather data including temperature, conditions, humidity, and wind speed. Use this when you need to check weather conditions for travel planning or general information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for (e.g., 'Beijing', 'New York', 'London')",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units. Default is 'celsius'.",
                    "default": "celsius"
                }
            },
            "required": ["city"],
        },
    },
]

# Map tool names to actual functions
TOOL_FUNCTIONS = {
    "get_team_members": get_team_members,
    "get_expenses": get_expenses,
    "get_custom_budget": get_custom_budget,
}

# Direct call tool functions
DIRECT_TOOL_FUNCTIONS = {
    "get_weather": get_weather,
}


# ============================================================
# 3. Agent Configuration
# ============================================================

@dataclass
class AgentConfig:
    """Agent configuration"""
    model: str = "global.anthropic.claude-opus-4-5-20251101-v1:0"
    max_tokens: int = 8192
    max_iterations: int = 15
    temperature: float = 0.7
    enable_visualization: bool = True


# ============================================================
# 4. Bedrock Docker Sandbox Agent
# ============================================================

class BedrockDockerSandboxAgent:
    """
    AI Agent using AnthropicBedrock with Docker Sandbox for Code Execution

    This version uses Docker sandbox for secure, isolated code execution.
    Suitable for production environments where security is critical.

    Features:
    - Docker-based sandboxed code execution
    - Network isolation and resource limits
    - Multi-turn conversation with history
    - Tool registration with external configs
    - Direct tool calls + Programmatic tool calling

    Usage:
        agent = BedrockDockerSandboxAgent()
        response = await agent.chat("Analyze expense data")

        # Or with context manager
        async with BedrockDockerSandboxAgent() as agent:
            response = await agent.chat("Analyze expense data")
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        sandbox_config: SandboxConfig | None = None,
        enable_session_reuse: bool | None = None
    ):
        self.config = config or AgentConfig()
        self.sandbox_config = sandbox_config or SandboxConfig(
            memory_limit="256m",
            timeout_seconds=60.0,
            network_disabled=True,
            enable_session_reuse=enable_session_reuse or False,
            session_timeout_seconds=270.0,  # 4.5 minutes like official PTC
        )
        # Use explicit parameter if provided, otherwise read from sandbox_config
        self.enable_session_reuse = enable_session_reuse if enable_session_reuse is not None else self.sandbox_config.enable_session_reuse
        self.tool_registry = ToolRegistry()
        self._client = None
        self._conversation_history: list[dict] = []
        self._sandbox: SandboxExecutor | None = None
        self._visualizer = None
        self._current_session_id: str | None = None

        # Register tools from config
        self._register_tools_from_config()

        # Initialize visualizer if enabled
        if self.config.enable_visualization:
            self._visualizer = visualize(auto_show=True)

        logger.info(f"Initialized BedrockDockerSandboxAgent with model: {self.config.model}")
        if enable_session_reuse:
            logger.info("Session reuse mode enabled")

    def _register_tools_from_config(self) -> None:
        """Register tools from TOOL_CONFIGS and DIRECT_TOOL_CONFIGS"""
        # Register code execution tools
        for tool_config in TOOL_CONFIGS:
            name = tool_config["name"]
            func = TOOL_FUNCTIONS.get(name)

            if func is None:
                logger.warning(f"Function not found for tool: {name}")
                continue

            self.tool_registry.register(
                name=name,
                description=tool_config["description"],
                input_schema=tool_config["input_schema"],
                allowed_callers=[ToolCallerType.CODE_EXECUTION]
            )(func)

            logger.info(f"Registered tool: {name}")

        # Register direct call tools
        for tool_config in DIRECT_TOOL_CONFIGS:
            name = tool_config["name"]
            func = DIRECT_TOOL_FUNCTIONS.get(name)

            if func is None:
                logger.warning(f"Function not found for direct tool: {name}")
                continue

            self.tool_registry.register(
                name=name,
                description=tool_config["description"],
                input_schema=tool_config["input_schema"],
                allowed_callers=[ToolCallerType.DIRECT]
            )(func)

            logger.info(f"Registered direct tool: {name}")

    async def __aenter__(self):
        """Async context manager entry"""
        logger.info("Docker sandbox agent ready")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Close session if using session reuse
        if self._current_session_id and self._sandbox:
            await self._sandbox.close_session(self._current_session_id)
            logger.info(f"Closed session: {self._current_session_id}")
        logger.info("Docker sandbox agent closed")

    @property
    def current_session_id(self) -> str | None:
        """Get the current session ID (for container reuse mode)"""
        return self._current_session_id

    @property
    def active_sessions(self) -> dict:
        """Get all active sessions info"""
        if self._sandbox:
            return self._sandbox.active_sessions
        return {}

    @property
    def sandbox(self) -> SandboxExecutor:
        """Lazy load sandbox executor"""
        if self._sandbox is None:
            self._sandbox = SandboxExecutor(self.tool_registry, self.sandbox_config)
        return self._sandbox

    @property
    def client(self):
        """Lazy load AnthropicBedrock client"""
        if self._client is None:
            from anthropic import AnthropicBedrock
            self._client = AnthropicBedrock()
            logger.info("AnthropicBedrock client initialized")
        return self._client

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool documentation"""
        tools_doc = self.tool_registry.generate_tools_documentation()

        return f"""You are a powerful AI assistant capable of completing complex tasks through a code execution environment.

## Code Execution Environment

You have a Python code execution environment with the following predefined asynchronous tool functions:

{tools_doc}

## Usage

When you need to execute multi-step tasks, use the `execute_code` tool to write Python code.

### Key Rules:
1. All tool calls must use `await`, for example: `result = await query_sales(region="East")`
2. Use `print()` to output results - this is the only way for you to get execution results
3. You can perform data processing, filtering, aggregation, and conditional logic in your code
4. After code execution completes, you will see the content output by print

## Best Practices for coding

1. **Parallel Execution (Recommended)**: When calling the same tool for multiple independent items, use `asyncio.gather()` for parallel execution instead of sequential loops
```python
import asyncio

# Fetch expenses for all employees in parallel (recommended)
employee_ids = ["ENG001", "ENG002", "ENG003"]
expense_tasks = [
    get_expenses(employee_id=emp_id, quarter="Q3")
    for emp_id in employee_ids
]
expenses_results = await asyncio.gather(*expense_tasks)

# Process results
for emp_id, expenses_json in zip(employee_ids, expenses_results):
    expenses = json.loads(expenses_json)
    print(f"{{emp_id}}: {{len(expenses)}} expenses")
```

2. **Batch Processing**: Write multiple related operations in a single code block
```python
results = {{}}
for region in ["East", "West", "Central"]:
    data = await query_sales(region=region)
    results[region] = sum(item["revenue"] for item in data)
print(f"Regional revenue: {{results}}")
```

3. **Data Filtering**: Fetch data first, then filter in code
```python
servers = await list_servers()
for server in servers:
    status = await check_server_health(server_id=server)
    if status["status"] != "healthy":
        print(f"Problem: {{server}} - {{status}}")
```

4. **Conditional Logic**: Decide next steps based on intermediate results
```python
file_info = await get_file_info(path="/data/large.csv")
if file_info["size"] > 1000000:
    summary = await get_file_summary(path="/data/large.csv")
else:
    content = await read_file(path="/data/large.csv")
    summary = content
print(summary)
```

5. **Early Termination**: Stop immediately once the desired result is found
```python
servers = ["us-east", "eu-west", "ap-south"]
for server in servers:
    status = await check_health(server_id=server)
    if status["healthy"]:
        print(f"Found healthy server: {{server}}")
        break
        

## Docker Sandbox Features

- Secure, isolated execution environment
- Network disabled for security
- Resource limits enforced (memory, CPU)
- Timeout protection

## Best Practices

1. **Batch Processing**: Combine multiple operations in one code block to minimize round-trips
2. **Always parse JSON**: Tool functions return JSON strings
3. **Handle errors gracefully**: Use try/except for robust code
"""

    def _create_execute_code_tool(self) -> dict:
        """Create the execute_code tool definition"""
        return {
            "name": "execute_code",
            "description": """Execute Python code in a sandbox environment.

The code can call predefined asynchronous tool functions to complete tasks.
Use `print()` to output the results you need to see.

Applicable Scenarios:
- Need to call tools multiple times (e.g., loop iterations)
- Need to process, filter, or aggregate data returned by tools
- Need to make conditional decisions based on intermediate results
- Need to batch process multiple similar tasks

Note: All tool calls must use `await`, for example:
result = await query_database(sql="SELECT * FROM users")
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute in Docker sandbox. Use await for tool calls, print() for output."
                    }
                },
                "required": ["code"]
            }
        }

    async def _execute_code(self, code: str) -> dict:
        """Execute code using the Docker sandbox"""
        logger.info(f"Executing code in Docker sandbox:\n{code}")

        # Use session reuse if enabled
        if self.enable_session_reuse:
            execution_result = await self.sandbox.execute(
                code,
                session_id=self._current_session_id,
                reuse_session=True
            )
            # Unpack result and session_id (returns tuple in session mode)
            if isinstance(execution_result, tuple):
                result, session_id = execution_result
                # Store session_id for next execution
                if self._current_session_id is None:
                    self._current_session_id = session_id
                    logger.info(f"Created new session: {session_id}")
                else:
                    logger.debug(f"Reused session: {session_id}")
            else:
                result = execution_result
        else:
            # Single execution mode (no session reuse)
            result = await self.sandbox.execute(code)

        if result.success:
            content = result.stdout or "(Code executed successfully, but no print output)"
        else:
            content = f"Execution Error: {result.stderr}"

        logger.info(f"Sandbox execution result: {content[:500]}...")
        return {"success": result.success, "output": content}

    async def chat(
        self,
        message: str,
        reset_history: bool = False
    ) -> tuple[str, int, int]:
        """
        Send a message and get a response

        Args:
            message: User message
            reset_history: If True, clear conversation history before processing

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        if reset_history:
            self._conversation_history.clear()
            logger.info("Conversation history cleared")

        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": message
        })
        total_input_tokens = 0
        total_output_tokens = 0
        iteration = 0

        # Build tools list: execute_code + direct call tools
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

            # Call Claude API
            response = self.client.beta.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self._build_system_prompt(),
                betas=["tool-examples-2025-10-29"],
                messages=self._conversation_history,
                tools=tools
            )
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Visualize the response if enabled
            if self._visualizer:
                self._visualizer.capture(response)

            logger.info(f"Stop reason: {response.stop_reason}")

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract text response
                text_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text

                # Add assistant response to history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": text_content
                })

                return text_content, total_input_tokens, total_output_tokens

            elif response.stop_reason == "tool_use":
                # Process tool calls
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

                        # Execute the tool
                        if block.name == "execute_code":
                            code = block.input.get("code", "")
                            result = await self._execute_code(code)
                            content = result["output"]
                        elif block.name in DIRECT_TOOL_FUNCTIONS:
                            # Direct tool call
                            try:
                                func = DIRECT_TOOL_FUNCTIONS[block.name]
                                result = func(**block.input)
                                content = result
                                logger.info(f"Direct tool result: {content[:200]}..." if len(content) > 200 else f"Direct tool result: {content}")
                            except Exception as e:
                                content = f"Error calling {block.name}: {str(e)}"
                                logger.error(content)
                        else:
                            content = f"Unknown tool: {block.name}"

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content
                        })

                # Add to conversation history
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

        raise RuntimeError(f"Exceeded maximum iterations ({self.config.max_iterations})")

    def reset(self) -> None:
        """Reset conversation history"""
        self._conversation_history.clear()
        logger.info("Agent conversation reset")

    @property
    def history(self) -> list[dict]:
        """Get conversation history"""
        return self._conversation_history.copy()


# ============================================================
# 5. Demo Functions
# ============================================================

async def run_demo(enable_visualization: bool = True):
    """Run the Docker sandbox agent demonstration"""
    print("=" * 70)
    print("Bedrock Docker Sandbox Agent Demo")
    print("=" * 70)

    config = AgentConfig(
        # model="global.anthropic.claude-opus-4-5-20251101-v1:0",
        model = 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
        max_tokens=8192,
        max_iterations=15,
        enable_visualization=enable_visualization
    )

    sandbox_config = SandboxConfig(
        memory_limit="256m",
        timeout_seconds=60.0,
        network_disabled=True,
        enable_session_reuse=True,
    )

    # Use async context manager to manage Docker sandbox lifecycle
    async with BedrockDockerSandboxAgent(config=config, sandbox_config=sandbox_config) as agent:
        print(f"\nDocker Sandbox Agent created with {len(agent.tool_registry.get_all())} tools:")
        for tool in agent.tool_registry.get_all():
            print(f"  - {tool.name} ({tool.allowed_callers[0].value})")

        print("\nDocker Sandbox Configuration:")
        print(f"  - Memory Limit: {sandbox_config.memory_limit}")
        print(f"  - Timeout: {sandbox_config.timeout_seconds}s")
        print(f"  - Network Disabled: {sandbox_config.network_disabled}")
        print(f"  - Session Reuse: {sandbox_config.enable_session_reuse}")

        # Demo: Team Expense Analysis
        print("\n" + "=" * 70)
        query = "Which engineering team members exceeded their Q3 travel budget? Standard quarterly travel budget is $5,000. However, some employees have custom budget limits. For anyone who exceeded the $5,000 standard budget, check if they have a custom budget exception. If they do, use that custom limit instead to determine if they truly exceeded their budget."

        print(f"Query: {query}")
        print("=" * 70)
        response, total_input_tokens, total_output_tokens = await agent.chat(query)
        print("\n--- Agent Response ---")
        print(response)
        print(f"\nToken Usage:")
        print(f"  - Input tokens: {total_input_tokens}")
        print(f"  - Output tokens: {total_output_tokens}")
        print(f"  - Total tokens: {total_input_tokens + total_output_tokens}")


async def interactive_mode(enable_visualization: bool = True):
    """Run agent in interactive mode"""
    print("=" * 70)
    print("Docker Sandbox Agent - Interactive Mode")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 70)

    config = AgentConfig(
        model="global.anthropic.claude-opus-4-5-20251101-v1:0",
        max_tokens=8192,
        max_iterations=15,
        enable_visualization=enable_visualization
    )

    async with BedrockDockerSandboxAgent(config=config) as agent:
        print(f"\nAvailable tools: {[t.name for t in agent.tool_registry.get_all()]}")
        print("\nExample queries:")
        print("  - 'Get all engineering team members'")
        print("  - 'Analyze Q3 expenses for the sales team'")
        print("  - 'Check budget compliance for marketing department'")
        print("  - 'What's the weather in Tokyo?'")
        print("\nStart chatting with the agent:\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                elif user_input.lower() == "reset":
                    agent.reset()
                    print("Conversation history cleared.\n")
                    continue
                elif not user_input:
                    continue

                response, input_tokens, output_tokens = await agent.chat(user_input)
                print(f"\nAgent: {response}")
                print(f"[Tokens: {input_tokens} in / {output_tokens} out]\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.exception("Error during chat")
                print(f"\nError: {e}\n")


async def interactive_mode_with_session(enable_visualization: bool = True):
    """Run agent in interactive mode with session reuse enabled"""
    print("=" * 70)
    print("Docker Sandbox Agent - Interactive Mode (Session Reuse Enabled)")
    print("Type 'quit' to exit, 'reset' to clear history, 'session' to view session info")
    print("=" * 70)

    config = AgentConfig(
        model="global.anthropic.claude-opus-4-5-20251101-v1:0",
        max_tokens=8192,
        max_iterations=15,
        enable_visualization=enable_visualization
    )

    sandbox_config = SandboxConfig(
        memory_limit="256m",
        timeout_seconds=60.0,
        network_disabled=True,
        enable_session_reuse=True,
        session_timeout_seconds=270.0,
    )

    async with BedrockDockerSandboxAgent(
        config=config,
        sandbox_config=sandbox_config,
        enable_session_reuse=True
    ) as agent:
        print(f"\nAvailable tools: {[t.name for t in agent.tool_registry.get_all()]}")
        print(f"Session reuse: ENABLED")
        print(f"Session timeout: {sandbox_config.session_timeout_seconds}s")
        print("\nNote: Variables persist between code executions!")
        print("\nStart chatting with the agent:\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                elif user_input.lower() == "reset":
                    agent.reset()
                    print("Conversation history cleared.\n")
                    continue
                elif user_input.lower() == "session":
                    print(f"\nCurrent Session ID: {agent.current_session_id}")
                    for sid, info in agent.active_sessions.items():
                        print(f"  Container: {info['container_id']}")
                        print(f"  Executions: {info['execution_count']}")
                        print(f"  Expires: {info['expires_at']}")
                    print()
                    continue
                elif not user_input:
                    continue

                response, input_tokens, output_tokens = await agent.chat(user_input)
                print(f"\nAgent: {response}")
                print(f"[Tokens: {input_tokens} in / {output_tokens} out | Session: {agent.current_session_id[:12] if agent.current_session_id else 'None'}...]\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.exception("Error during chat")
                print(f"\nError: {e}\n")


# ============================================================
# 6. Session Reuse Demo Functions
# ============================================================

async def run_session_reuse_demo(enable_visualization: bool = True):
    """
    Demonstrate Docker container session reuse feature

    This demo shows how the container is reused across multiple code executions,
    maintaining state (variables) between executions - just like official PTC.
    """
    print("=" * 70)
    print("Docker Sandbox Container Reuse Demo")
    print("=" * 70)
    print("\nThis demo shows container reuse across multiple executions.")
    print("State (variables) is preserved between executions.\n")

    config = AgentConfig(
        model="global.anthropic.claude-opus-4-5-20251101-v1:0",
        max_tokens=8192,
        max_iterations=15,
        enable_visualization=enable_visualization
    )

    sandbox_config = SandboxConfig(
        memory_limit="256m",
        timeout_seconds=60.0,
        network_disabled=True,
        enable_session_reuse=True,
        session_timeout_seconds=270.0,  # 4.5 minutes like official PTC
    )

    async with BedrockDockerSandboxAgent(
        config=config,
        sandbox_config=sandbox_config,
        enable_session_reuse=True
    ) as agent:
        print(f"Docker Sandbox Agent created with session reuse enabled")
        print(f"Session timeout: {sandbox_config.session_timeout_seconds}s\n")

        # First query - this will create the session
        print("=" * 70)
        query1 = "Get the list of engineering team members and store the count in a variable. Print how many members there are."
        print(f"Query 1: {query1}")
        print("=" * 70)

        response1, tokens1_in, tokens1_out = await agent.chat(query1)
        print("\n--- Response 1 ---")
        print(response1)
        print(f"\nSession ID: {agent.current_session_id}")
        print(f"Tokens: {tokens1_in} in / {tokens1_out} out")

        # Second query - reuses the same container, variables persist
        print("\n" + "=" * 70)
        query2 = "Now, using the team data already loaded, find the average number of members per level (junior, mid, senior, etc.). The data should still be in memory from the previous execution."
        print(f"Query 2 (reuses container): {query2}")
        print("=" * 70)

        response2, tokens2_in, tokens2_out = await agent.chat(query2)
        print("\n--- Response 2 ---")
        print(response2)
        print(f"\nSession ID (same as before): {agent.current_session_id}")
        print(f"Tokens: {tokens2_in} in / {tokens2_out} out")

        # Show session info
        print("\n" + "=" * 70)
        print("Active Sessions Info:")
        for sid, info in agent.active_sessions.items():
            print(f"  Session: {sid}")
            print(f"    Container: {info['container_id']}")
            print(f"    Executions: {info['execution_count']}")
            print(f"    Expires: {info['expires_at']}")
        print("=" * 70)


async def run_low_level_session_demo():
    """
    Low-level demo of SandboxExecutor session reuse without Agent wrapper

    This shows the direct API for session management.
    """
    print("=" * 70)
    print("Low-Level Sandbox Session Reuse Demo")
    print("=" * 70)
    print("\nDirect SandboxExecutor API usage with session management.\n")

    # Create tool registry with a simple tool
    registry = ToolRegistry()

    @registry.register(
        name="add_numbers",
        description="Add two numbers together"
    )
    def add_numbers(a: int, b: int) -> int:
        return a + b

    # Create sandbox config with session reuse
    config = SandboxConfig(
        memory_limit="128m",
        timeout_seconds=30.0,
        network_disabled=True,
        enable_session_reuse=True,
        session_timeout_seconds=60.0,  # 1 minute for demo
    )

    executor = SandboxExecutor(registry, config)

    try:
        # First execution - creates new session
        print("--- Execution 1: Create session and set variable ---")
        code1 = """
x = 10
y = 20
print(f"Initial values: x={x}, y={y}")
"""
        exec_result1 = await executor.execute(code1, reuse_session=True)
        # Unpack tuple result (ExecutionResult, session_id)
        result1, session_id = exec_result1 if isinstance(exec_result1, tuple) else (exec_result1, None)
        print(f"Output: {result1.stdout}")
        print(f"Session ID: {session_id}")
        print(f"Success: {result1.success}")
        print(f"Execution time: {result1.execution_time_ms:.2f}ms")

        # Second execution - reuses session, variables persist
        print("\n--- Execution 2: Reuse session, variables persist ---")
        code2 = """
# x and y should still exist from previous execution
z = x + y
print(f"z = x + y = {z}")
print(f"x is still: {x}")
"""
        exec_result2 = await executor.execute(code2, session_id=session_id)
        result2, session_id2 = exec_result2 if isinstance(exec_result2, tuple) else (exec_result2, None)
        print(f"Output: {result2.stdout}")
        print(f"Session ID (same): {session_id2}")
        print(f"Success: {result2.success}")

        # Third execution with tool call
        print("\n--- Execution 3: Tool call in reused session ---")
        code3 = """
# Call a tool and store result
result = await add_numbers(a=z, b=100)  # z=30 from previous execution
print(f"add_numbers({z}, 100) = {result}")
"""
        exec_result3 = await executor.execute(code3, session_id=session_id)
        result3, _ = exec_result3 if isinstance(exec_result3, tuple) else (exec_result3, None)
        print(f"Output: {result3.stdout}")
        print(f"Tool calls: {result3.tool_calls_count}")
        print(f"Success: {result3.success}")

        # Show active sessions
        print("\n--- Active Sessions ---")
        sessions = executor.active_sessions
        for sid, info in sessions.items():
            print(f"  {sid}: {info['execution_count']} executions, expires {info['expires_at']}")

        # Close session explicitly
        print("\n--- Closing session ---")
        if session_id:
            closed = await executor.close_session(session_id)
            print(f"Session closed: {closed}")
        else:
            print("No session to close")

    finally:
        # Cleanup all sessions
        await executor.close_all_sessions()
        executor.stop_cleanup_task()
        print("\nAll sessions cleaned up.")


# ============================================================
# 7. Main Entry Point
# ============================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bedrock Docker Sandbox Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo (single container per execution)
  python bedrock_docker_agent_example.py

  # Run session reuse demo (container reused across executions)
  python bedrock_docker_agent_example.py --session-reuse

  # Run low-level session API demo
  python bedrock_docker_agent_example.py --low-level

  # Interactive mode
  python bedrock_docker_agent_example.py -i

  # Interactive mode with session reuse
  python bedrock_docker_agent_example.py -i --session-reuse

  # Verbose logging
  python bedrock_docker_agent_example.py -v

  # Disable visualization
  python bedrock_docker_agent_example.py --no-viz
"""
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--session-reuse", "-s",
        action="store_true",
        help="Enable container session reuse (like official PTC)"
    )
    parser.add_argument(
        "--low-level",
        action="store_true",
        help="Run low-level SandboxExecutor session demo"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable response visualization"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    enable_viz = not args.no_viz

    if args.low_level:
        asyncio.run(run_low_level_session_demo())
    elif args.interactive:
        asyncio.run(interactive_mode_with_session(enable_visualization=enable_viz) if args.session_reuse else interactive_mode(enable_visualization=enable_viz))
    elif args.session_reuse:
        asyncio.run(run_session_reuse_demo(enable_visualization=enable_viz))
    else:
        asyncio.run(run_demo(enable_visualization=enable_viz))


if __name__ == "__main__":
    main()
