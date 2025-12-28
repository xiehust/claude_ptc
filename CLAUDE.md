# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Sandboxed Programmatic Tool Calling (PTC)** - a self-hosted implementation of Anthropic's Programmatic Tool Calling mechanism using Docker containers for secure code execution. It enables Claude to generate Python code that calls tools dynamically, with all code executing in isolated Docker sandboxes.

Key value: Instead of N model round-trips for N tool calls, Claude generates code that executes all tools in one sandbox session, returning only the final `print()` output to context.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Docker is running (required for sandbox execution)
docker info

# Run the main example (requires AWS credentials for Bedrock)
python examples/bedrock_docker_agent_example.py

# Run with options
python examples/bedrock_docker_agent_example.py -v              # Verbose logging
python examples/bedrock_docker_agent_example.py -i              # Interactive mode
python examples/bedrock_docker_agent_example.py --session-reuse # Enable container reuse
python examples/bedrock_docker_agent_example.py --no-viz        # Disable visualization
python examples/bedrock_docker_agent_example.py --docker        # Use Docker sandbox

# Run basic usage example
python examples/basic_usage.py

# Run tests
pytest
pytest -xvs tests/test_specific.py::test_name  # Single test
```

## Architecture

### Core Components (`sandboxed_ptc/`)

- **`SandboxExecutor`** (`sandbox.py`): Manages Docker container lifecycle and IPC communication. Supports session reuse where containers persist between executions for state preservation.

- **`ToolRegistry`** (`tool_registry.py`): Tool registration with decorator API. Auto-generates JSON schemas from function signatures. Supports `ToolCallerType.DIRECT` (Claude calls directly), `CODE_EXECUTION` (from sandbox), or `BOTH`.

- **`ProgrammaticToolOrchestrator`** (`orchestrator.py`): Coordinates Claude API calls with sandbox execution. Builds system prompts with tool documentation, handles the tool_use loop.

### IPC Protocol (Host ↔ Docker Container)

Tool calls from sandbox use stderr (to avoid mixing with `print()` output):
```
__PTC_TOOL_CALL__{"call_id": "uuid", "tool_name": "...", "arguments": {...}}__PTC_END_CALL__
```

Results sent back via stdin:
```
__PTC_TOOL_RESULT__{"call_id": "uuid", "result": ..., "error": null}__PTC_END_RESULT__
```

Final output via stdout:
```
__PTC_OUTPUT__{"success": true, "output": "...", "error": null}__PTC_END_OUTPUT__
```

### Execution Flow

1. User request → Orchestrator builds system prompt with tool docs
2. Claude returns `tool_use: execute_code` with Python code
3. SandboxExecutor creates Docker container (or reuses session)
4. Code runs, `await tool()` triggers IPC to host
5. Host executes tool via ToolRegistry, sends result back
6. Container continues execution, returns final `print()` output
7. Output sent back to Claude as `tool_result`

### Docker Security Settings

Containers run with: network disabled, read-only filesystem, non-root user, all capabilities dropped, memory/CPU limits, no-new-privileges.

## Key Patterns

### Tool Registration

```python
@registry.register(
    description="Query the database",
    output_description="Returns list[dict]",
    allowed_callers=[ToolCallerType.CODE_EXECUTION]
)
def query_database(sql: str) -> list[dict]:
    return db.execute(sql)
```

### Session Reuse

```python
config = SandboxConfig(enable_session_reuse=True, session_timeout_seconds=270.0)
executor = SandboxExecutor(registry, config)

result, session_id = await executor.execute("x = 10", reuse_session=True)
result, session_id = await executor.execute("print(x + 5)", session_id=session_id)  # x persists
```

## File Structure

```
sandboxed_ptc/     # Core library
  sandbox.py       # Docker execution, IPC, session management
  tool_registry.py # Tool definitions and schema generation
  orchestrator.py  # Claude API coordination
  exceptions.py    # Custom exceptions

examples/
  bedrock_docker_agent_example.py  # Full agent with Bedrock
  basic_usage.py                   # Minimal example

utils/
  team_expense_api.py   # Mock API for examples
  visualize.py          # Response visualization

Dockerfile         # Sandbox container image
```
