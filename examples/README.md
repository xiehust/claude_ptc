# Sandboxed Programmatic Tool Calling Examples

This directory contains examples demonstrating how to use the Docker-based Programmatic Tool Calling mechanism.

## Examples Overview

| File | Description | Complexity |
|------|-------------|------------|
| `basic_usage.py` | Basic sandboxed PTC with Orchestrator | Simple |
| `bedrock_docker_agent_example.py` | Full Agent with Docker sandbox | Advanced |

## Quick Start

```bash
# Install dependencies
pip install anthropic docker

# Configure AWS credentials (for Bedrock)
aws configure

# Ensure Docker is running
docker info

# Run examples
python basic_usage.py
python bedrock_docker_agent_example.py
```

## Example Details

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the core `ProgrammaticToolOrchestrator` with Docker sandbox execution.

**Features:**
- Tool registration with decorators
- Sandboxed code execution in Docker container
- IPC communication for tool calls
- Streaming response support

**Run:**
```bash
python basic_usage.py
```

### 2. Bedrock Docker Agent (`bedrock_docker_agent_example.py`)

A full-featured AI Agent using AnthropicBedrock with Docker sandbox.

**Features:**
- Multi-turn conversation with history
- Docker container session reuse (like official PTC)
- State persistence between code executions
- Direct tool calls + Programmatic tool calling
- Interactive mode support

**Run:**
```bash
# Basic demo
python bedrock_docker_agent_example.py

# Session reuse demo (container persists across executions)
python bedrock_docker_agent_example.py --session-reuse

# Interactive mode
python bedrock_docker_agent_example.py -i

# Interactive mode with session reuse
python bedrock_docker_agent_example.py -i --session-reuse

# Low-level session API demo
python bedrock_docker_agent_example.py --low-level

# Verbose logging
python bedrock_docker_agent_example.py -v

# Disable visualization
python bedrock_docker_agent_example.py --no-viz
```

**Example Tools:**
- `get_team_members(department)` - Get team members by department
- `get_expenses(employee_id, quarter)` - Get employee expenses
- `get_custom_budget(user_id)` - Check custom budget limits
- `get_weather(city)` - Get weather (direct call tool)

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────┐
│  BedrockDockerSandboxAgent      │
│  - Conversation Management      │
│  - Session Reuse Support        │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  AnthropicBedrock Client        │
│  - Claude API calls             │
│  - Tool use handling            │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  execute_code Tool              │
│  - Code execution in sandbox    │
│  - Tool function injection      │
│  - Output capture               │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  SandboxExecutor (Docker)       │
│  - Container management         │
│  - IPC communication            │
│  - Session reuse                │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  ToolRegistry                   │
│  - Tool definitions             │
│  - Schema management            │
│  - Function execution           │
└─────────────────────────────────┘
```

## Creating Your Own Agent

```python
from anthropic import AnthropicBedrock
from sandboxed_ptc import ToolRegistry, SandboxExecutor, SandboxConfig

# 1. Create tool registry
registry = ToolRegistry()

# 2. Register tools
@registry.register(
    description="Search the database",
    output_description="Returns list of matching records"
)
def search_db(query: str) -> list[dict]:
    return db.search(query)

# 3. Create sandbox executor with session reuse
config = SandboxConfig(
    memory_limit="256m",
    timeout_seconds=60.0,
    network_disabled=True,
    enable_session_reuse=True,
    session_timeout_seconds=270.0,  # 4.5 minutes
)
executor = SandboxExecutor(registry, config)

# 4. Execute code with session reuse
result, session_id = await executor.execute(
    "data = await search_db(query='test')\nprint(data)",
    reuse_session=True
)

# 5. Subsequent executions reuse the same container
result, session_id = await executor.execute(
    "print(len(data))",  # 'data' variable persists
    session_id=session_id
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Bedrock | `us-west-2` |
| `AWS_ACCESS_KEY_ID` | AWS access key | From credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | From credentials |

## Troubleshooting

### Error: "No module named 'anthropic'"
```bash
pip install anthropic
```

### Error: "No module named 'docker'"
```bash
pip install docker
```

### Error: "Cannot connect to Docker daemon"
```bash
# Ensure Docker is running
docker info

# On Linux, you may need to add user to docker group
sudo usermod -aG docker $USER
```

### Error: "AWS credentials not found"
```bash
# Configure AWS credentials
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Error: "Module 'sandboxed_ptc' not found"
```bash
# Run from project root or add to path
cd /path/to/claude_ptc
python -m examples.bedrock_docker_agent_example
```
