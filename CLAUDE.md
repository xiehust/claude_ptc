# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Sandboxed Programmatic Tool Calling (PTC)** - a self-hosted implementation of Anthropic's Programmatic Tool Calling mechanism. It enables Claude to generate Python code that calls tools dynamically, with code executing in either:
- **Docker containers** (secure, isolated) - for production with untrusted code
- **Local Python process** (fast, no Docker) - for development/testing

Key value: Instead of N model round-trips for N tool calls, Claude generates code that executes all tools in one sandbox session, returning only the final `print()` output to context.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Docker sandbox (requires Docker)
docker info  # Ensure Docker is running
python examples/bedrock_docker_agent_example.py

# Run with Local sandbox (no Docker required)
python examples/local_agent_example.py

# Common options (both examples support these)
-v              # Verbose logging
-i              # Interactive mode
--session-reuse # Enable session/state persistence
--no-viz        # Disable visualization
--low-level     # Low-level executor API demo

# Run tests
pytest
pytest -xvs tests/test_specific.py::test_name  # Single test
```

## Architecture

### Core Components (`sandboxed_ptc/`)

- **`SandboxExecutor`** (`sandbox.py`): Manages Docker container lifecycle and IPC communication. Supports session reuse where containers persist between executions for state preservation. Use for production with untrusted code.

- **`LocalSandboxExecutor`** (`local_sandbox.py`): Executes code in local Python process without Docker. Same API as `SandboxExecutor` (drop-in replacement). Supports session reuse for state persistence. Use for development/testing or environments without Docker. **Warning: No security isolation.**

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

### Session Reuse (Docker)

```python
config = SandboxConfig(enable_session_reuse=True, session_timeout_seconds=270.0)
executor = SandboxExecutor(registry, config)

result, session_id = await executor.execute("x = 10", reuse_session=True)
result, session_id = await executor.execute("print(x + 5)", session_id=session_id)  # x persists
```

### Local Sandbox (No Docker)

```python
from sandboxed_ptc import LocalSandboxExecutor, LocalSandboxConfig

config = LocalSandboxConfig(timeout_seconds=60.0, enable_session_reuse=True)
executor = LocalSandboxExecutor(registry, config)

# Same API as SandboxExecutor
result, session_id = await executor.execute("x = 10", reuse_session=True)
result, session_id = await executor.execute("print(x + 5)", session_id=session_id)  # x persists
```

## File Structure

```
sandboxed_ptc/     # Core library
  sandbox.py       # Docker execution, IPC, session management
  local_sandbox.py # Local execution (no Docker), same API
  tool_registry.py # Tool definitions and schema generation
  orchestrator.py  # Claude API coordination
  exceptions.py    # Custom exceptions

examples/
  bedrock_docker_agent_example.py  # Full agent with Docker sandbox
  local_agent_example.py           # Full agent with local sandbox (no Docker)
  basic_usage.py                   # Minimal example

utils/
  team_expense_api.py   # Mock API for examples
  visualize.py          # Response visualization

Dockerfile         # Sandbox container image (for Docker mode)
```
