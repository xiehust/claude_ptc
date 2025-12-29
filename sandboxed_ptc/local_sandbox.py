"""
Local Sandbox Executor - Execute code in local Python process without Docker

This executor provides a simple alternative to the Docker-based SandboxExecutor
for development/testing or environments where Docker is not available.

Key differences from SandboxExecutor:
- No Docker dependency - code runs in the same Python process
- No network/filesystem isolation (less secure)
- Faster startup (no container overhead)
- Tool functions are injected directly into execution globals
- Same API as SandboxExecutor for drop-in replacement

Warning:
- This executor provides NO security isolation
- Only use with trusted code or in development environments
- For production use with untrusted code, use SandboxExecutor (Docker)
"""

import asyncio
import json
import uuid
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import logging
import traceback

from .tool_registry import ToolRegistry
from .exceptions import TimeoutError

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Code execution result (compatible with sandbox.py)"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    tool_calls_count: int = 0
    execution_time_ms: float = 0


@dataclass
class LocalSandboxConfig:
    """Local sandbox configuration"""
    timeout_seconds: float = 60.0
    max_output_size: int = 100000  # Max characters for captured output
    # Session configuration
    session_timeout_seconds: float = 270.0  # 4.5 minutes (same as Docker version)
    enable_session_reuse: bool = True


@dataclass
class LocalSession:
    """Local session for state persistence"""
    session_id: str
    exec_globals: dict
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime
    execution_count: int = 0
    is_busy: bool = False

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() > self.expires_at

    def refresh(self, timeout_seconds: float) -> None:
        """Refresh session expiration time"""
        self.last_used_at = datetime.now()
        self.expires_at = self.last_used_at + timedelta(seconds=timeout_seconds)


class OutputCapture:
    """Capture print() output"""

    def __init__(self, max_size: int = 100000):
        self.outputs: list[str] = []
        self.max_size = max_size
        self._current_size = 0

    def write(self, text: str) -> None:
        if self._current_size < self.max_size:
            self.outputs.append(text)
            self._current_size += len(text)

    def flush(self) -> None:
        pass

    def get_output(self) -> str:
        return "".join(self.outputs)

    def clear(self) -> None:
        self.outputs = []
        self._current_size = 0


class LocalSandboxExecutor:
    """
    Local Sandbox Executor - Execute code without Docker

    Provides the same API as SandboxExecutor but runs code directly in
    the local Python process. Suitable for development, testing, or
    environments where Docker is not available.

    Features:
    - Same API as SandboxExecutor (drop-in replacement)
    - Session reuse support (state persistence)
    - Async tool function injection
    - Output capture
    - Timeout protection

    Limitations:
    - No security isolation (code runs in same process)
    - No network/filesystem restrictions
    - Only use with trusted code

    Usage:
        executor = LocalSandboxExecutor(registry)

        # Single execution (no state persistence)
        result = await executor.execute("print('hello')")

        # With session reuse (state persists between executions)
        result1, session_id = await executor.execute("x = 10", reuse_session=True)
        result2, session_id = await executor.execute("print(x + 5)", session_id=session_id)
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        config: LocalSandboxConfig | None = None
    ):
        self.tool_registry = tool_registry
        self.config = config or LocalSandboxConfig()

        # Session management
        self._sessions: dict[str, LocalSession] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_running = False

    def _create_tool_function(self, tool_name: str):
        """Create an async tool function that calls the registry"""
        async def tool_func(**kwargs) -> Any:
            logger.debug(f"Tool call: {tool_name}({kwargs})")
            result = await self.tool_registry.execute_async(tool_name, kwargs)
            logger.debug(f"Tool result: {str(result)[:200]}...")
            return result

        return tool_func

    def _prepare_exec_globals(self, existing_globals: dict | None = None) -> dict:
        """Prepare execution globals with tool functions injected"""
        exec_globals = existing_globals.copy() if existing_globals else {
            "__builtins__": __builtins__,
            "asyncio": asyncio,
            "json": json,
        }

        # Inject tool functions
        for tool in self.tool_registry.get_code_execution_tools():
            exec_globals[tool.name] = self._create_tool_function(tool.name)

        return exec_globals

    async def execute(
        self,
        code: str,
        session_id: str | None = None,
        reuse_session: bool = False
    ) -> ExecutionResult | tuple[ExecutionResult, str]:
        """
        Execute Python code

        Args:
            code: Python code to execute
            session_id: Optional session ID for state persistence
            reuse_session: Enable session reuse mode

        Returns:
            - Single mode: ExecutionResult
            - Session mode: (ExecutionResult, session_id)
        """
        if session_id or reuse_session:
            return await self._execute_with_session(code, session_id)

        return await self._execute_single(code)

    async def _execute_single(self, code: str) -> ExecutionResult:
        """Execute code without session (stateless)"""
        start_time = time_module.time()
        tool_calls_count = 0

        # Prepare execution environment
        exec_globals = self._prepare_exec_globals()

        # Track tool calls
        original_funcs = {}
        for tool in self.tool_registry.get_code_execution_tools():
            original_func = exec_globals[tool.name]
            original_funcs[tool.name] = original_func

            async def tracked_func(
                _original=original_func,
                **kwargs
            ) -> Any:
                nonlocal tool_calls_count
                tool_calls_count += 1
                return await _original(**kwargs)

            exec_globals[tool.name] = tracked_func

        # Capture output
        output_capture = OutputCapture(self.config.max_output_size)
        exec_globals["print"] = lambda *args, **kwargs: output_capture.write(
            " ".join(str(a) for a in args) + kwargs.get("end", "\n")
        )

        try:
            result = await self._run_code_with_timeout(code, exec_globals)
            execution_time = (time_module.time() - start_time) * 1000

            if result["success"]:
                return ExecutionResult(
                    success=True,
                    stdout=output_capture.get_output(),
                    stderr="",
                    return_code=0,
                    tool_calls_count=tool_calls_count,
                    execution_time_ms=execution_time
                )
            else:
                return ExecutionResult(
                    success=False,
                    stdout=output_capture.get_output(),
                    stderr=result.get("error", "Unknown error"),
                    return_code=1,
                    tool_calls_count=tool_calls_count,
                    execution_time_ms=execution_time
                )

        except asyncio.TimeoutError:
            raise TimeoutError(self.config.timeout_seconds, "Code execution")

    async def _execute_with_session(
        self,
        code: str,
        session_id: str | None = None
    ) -> tuple[ExecutionResult, str]:
        """Execute code with session (state persists)"""
        start_time = time_module.time()
        tool_calls_count = 0

        # Get or create session
        session = None
        if session_id:
            session = self.get_session(session_id)
            if session is None:
                logger.warning(f"Session {session_id} not found or expired, creating new")

        if session is None:
            session = self._create_session()
            self.start_cleanup_task()

        session.is_busy = True

        try:
            session.refresh(self.config.session_timeout_seconds)

            # Use session's exec_globals directly (preserves state)
            exec_globals = session.exec_globals

            # Ensure base modules are available
            exec_globals["__builtins__"] = __builtins__
            exec_globals["asyncio"] = asyncio
            exec_globals["json"] = json

            # Get tool names for tracking
            tool_names = {t.name for t in self.tool_registry.get_code_execution_tools()}

            # Inject tool functions with call tracking
            tool_call_counter = [0]  # Use list for mutable closure
            for tool in self.tool_registry.get_code_execution_tools():
                base_func = self._create_tool_function(tool.name)

                def make_wrapper(fn):
                    async def wrapper(**kwargs):
                        tool_call_counter[0] += 1
                        return await fn(**kwargs)
                    return wrapper

                exec_globals[tool.name] = make_wrapper(base_func)

            # Capture output
            output_capture = OutputCapture(self.config.max_output_size)
            exec_globals["print"] = lambda *args, **kwargs: output_capture.write(
                " ".join(str(a) for a in args) + kwargs.get("end", "\n")
            )

            result = await self._run_code_with_timeout(code, exec_globals)

            # Update tool call count
            tool_calls_count = tool_call_counter[0]

            # Clean up temporary items from exec_globals (keep user variables)
            # Remove the custom print (will be re-added next execution)
            if "print" in exec_globals:
                del exec_globals["print"]
            # Remove tool functions (will be re-added next execution)
            for name in tool_names:
                if name in exec_globals:
                    del exec_globals[name]
            # Remove __user_main__ if it exists
            if "__user_main__" in exec_globals:
                del exec_globals["__user_main__"]

            execution_time = (time_module.time() - start_time) * 1000
            session.execution_count += 1

            if result["success"]:
                exec_result = ExecutionResult(
                    success=True,
                    stdout=output_capture.get_output(),
                    stderr="",
                    return_code=0,
                    tool_calls_count=tool_calls_count,
                    execution_time_ms=execution_time
                )
            else:
                exec_result = ExecutionResult(
                    success=False,
                    stdout=output_capture.get_output(),
                    stderr=result.get("error", "Unknown error"),
                    return_code=1,
                    tool_calls_count=tool_calls_count,
                    execution_time_ms=execution_time
                )

            return exec_result, session.session_id

        finally:
            session.is_busy = False

    async def _run_code_with_timeout(
        self,
        code: str,
        exec_globals: dict
    ) -> dict:
        """Run code with timeout protection"""

        async def execute_code():
            # To preserve variables in exec_globals (for session state),
            # we wrap the code but explicitly copy locals back to globals
            indented_code = "\n".join("    " + line for line in code.split("\n"))

            # Create wrapper that copies locals to globals after execution
            wrapped_code = f"""
async def __user_main__(__exec_globals__):
    # Make globals accessible
    globals().update(__exec_globals__)

{indented_code}

    # Copy all new local variables back to globals
    __new_vars__ = {{k: v for k, v in locals().items()
                    if not k.startswith('_') and k != '__exec_globals__'}}
    __exec_globals__.update(__new_vars__)
"""
            try:
                # Compile and execute the function definition
                exec(compile(wrapped_code, "<user_code>", "exec"), exec_globals)
                # Run the async function, passing exec_globals
                await exec_globals["__user_main__"](exec_globals)
                # Cleanup
                if "__user_main__" in exec_globals:
                    del exec_globals["__user_main__"]
                return {"success": True, "error": None}

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.debug(f"Code execution error: {error_msg}")
                logger.debug(traceback.format_exc())
                return {"success": False, "error": error_msg}

        try:
            return await asyncio.wait_for(
                execute_code(),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(self.config.timeout_seconds, "Code execution")

    # ==================== Session Management ====================

    def _create_session(self) -> LocalSession:
        """Create a new session"""
        session_id = f"local_sess_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        session = LocalSession(
            session_id=session_id,
            exec_globals={
                "__builtins__": __builtins__,
                "asyncio": asyncio,
                "json": json,
            },
            created_at=now,
            expires_at=now + timedelta(seconds=self.config.session_timeout_seconds),
            last_used_at=now,
            execution_count=0,
            is_busy=False
        )

        self._sessions[session_id] = session
        logger.info(f"Created local session: {session_id}")
        return session

    def get_session(self, session_id: str) -> LocalSession | None:
        """Get a session by ID"""
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            return session
        elif session and session.is_expired():
            # Session expired, clean up
            self._sessions.pop(session_id, None)
            logger.info(f"Session {session_id} expired and removed")
            return None
        return None

    async def close_session(self, session_id: str) -> bool:
        """Close and cleanup a session"""
        session = self._sessions.pop(session_id, None)
        if session:
            logger.info(f"Closed session: {session_id}")
            return True
        return False

    async def close_all_sessions(self) -> None:
        """Close all sessions"""
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)

    async def _cleanup_expired_sessions(self) -> None:
        """Background task to cleanup expired sessions"""
        while self._cleanup_running:
            await asyncio.sleep(60.0)  # Check every minute

            expired_ids = [
                sid for sid, session in self._sessions.items()
                if session.is_expired()
            ]

            for session_id in expired_ids:
                logger.info(f"Cleaning up expired session: {session_id}")
                await self.close_session(session_id)

    def start_cleanup_task(self) -> None:
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
            logger.debug("Session cleanup task started")

    def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task"""
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    @property
    def active_sessions(self) -> dict[str, dict]:
        """Get all active sessions info"""
        return {
            sid: {
                "session_id": sid,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "execution_count": session.execution_count,
                "is_busy": session.is_busy
            }
            for sid, session in self._sessions.items()
            if not session.is_expired()
        }
