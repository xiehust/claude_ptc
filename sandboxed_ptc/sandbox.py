"""
沙箱执行器 - 在隔离的 Docker 容器中执行代码

核心机制：
1. 代码在无网络的 Docker 容器中执行
2. 工具调用通过 stdin/stdout IPC 传递到主进程
3. 主进程执行实际工具，将结果返回给容器
4. 容器中的代码继续执行

容器复用模式（v2）：
- 支持通过 session_id 复用已有容器
- 容器在会话期间保持运行，支持多次代码执行
- 会话过期后自动清理（默认 4.5 分钟）
- 状态在同一会话的多次执行间保持
"""

import asyncio
import json
import uuid
import os
import tempfile
import shutil
import threading
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable
from pathlib import Path
import logging

from .tool_registry import ToolRegistry
from .exceptions import (
    SandboxError,
    TimeoutError,
    CodeExecutionError,
    ContainerError,
    IPCError,
    ToolExecutionError
)

logger = logging.getLogger(__name__)

# IPC 协议标记
IPC_TOOL_CALL_START = "__PTC_TOOL_CALL__"
IPC_TOOL_CALL_END = "__PTC_END_CALL__"
IPC_TOOL_RESULT_START = "__PTC_TOOL_RESULT__"
IPC_TOOL_RESULT_END = "__PTC_END_RESULT__"
IPC_CODE_OUTPUT_START = "__PTC_OUTPUT__"
IPC_CODE_OUTPUT_END = "__PTC_END_OUTPUT__"


@dataclass
class ExecutionResult:
    """代码执行结果"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    tool_calls_count: int = 0
    execution_time_ms: float = 0


@dataclass
class ToolCallRequest:
    """工具调用请求"""
    call_id: str
    tool_name: str
    arguments: dict


@dataclass
class ToolCallResponse:
    """工具调用响应"""
    call_id: str
    success: bool
    result: Any = None
    error: str | None = None


@dataclass
class SandboxConfig:
    """沙箱配置"""
    image: str = "python:3.11-slim"
    memory_limit: str = "256m"
    cpu_quota: int = 50000  # 50% of one CPU
    cpu_period: int = 100000
    timeout_seconds: float = 60.0
    network_disabled: bool = True
    read_only: bool = True
    working_dir: str = "/workspace"
    # 自定义镜像（如果需要预装依赖）
    custom_image: str | None = None
    # 会话配置（容器复用模式）
    session_timeout_seconds: float = 270.0  # 4.5 分钟，与官方 PTC 一致
    enable_session_reuse: bool = True  # 是否启用容器复用
    cleanup_interval_seconds: float = 60.0  # 清理检查间隔


@dataclass
class SandboxSession:
    """沙箱会话 - 用于容器复用"""
    session_id: str
    container: Any  # Docker container object
    socket: Any  # IPC socket
    temp_dir: str  # 临时目录路径
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime
    execution_count: int = 0
    is_busy: bool = False  # 是否正在执行代码

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return datetime.now() > self.expires_at

    def refresh(self, timeout_seconds: float) -> None:
        """刷新会话过期时间"""
        self.last_used_at = datetime.now()
        self.expires_at = self.last_used_at + timedelta(seconds=timeout_seconds)


class SandboxExecutor:
    """
    沙箱代码执行器

    在隔离的 Docker 容器中执行代码，支持工具调用拦截

    支持两种模式：
    1. 单次执行模式（默认）：每次执行创建新容器，执行后销毁
    2. 会话复用模式：通过 session_id 复用已有容器，支持状态保持

    会话复用示例：
        executor = SandboxExecutor(registry, config)

        # 首次执行，创建新会话
        result, session_id = await executor.execute("x = 1", reuse_session=True)

        # 复用会话，变量 x 仍然存在
        result, session_id = await executor.execute("print(x + 1)", session_id=session_id)

        # 手动关闭会话
        await executor.close_session(session_id)
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        config: SandboxConfig | None = None
    ):
        self.tool_registry = tool_registry
        self.config = config or SandboxConfig()
        self._docker_client = None
        self._runner_script_path: Path | None = None

        # 会话管理
        self._sessions: dict[str, SandboxSession] = {}
        self._sessions_lock = threading.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_running = False

    @property
    def docker_client(self):
        """懒加载 Docker 客户端"""
        if self._docker_client is None:
            try:
                import docker
                self._docker_client = docker.from_env()
            except ImportError:
                raise SandboxError(
                    "Docker SDK not installed. Run: pip install docker"
                )
            except Exception as e:
                raise ContainerError(f"Failed to connect to Docker: {e}")
        return self._docker_client

    def _get_runner_script(self, loop_mode: bool = False) -> str:
        """
        生成在容器内执行的 runner 脚本

        Args:
            loop_mode: 是否启用循环模式（用于容器复用）
                - False: 单次执行后退出
                - True: 循环等待多次代码执行，支持状态保持
        """
        tools_info = self.tool_registry.to_json()

        return f'''#!/usr/bin/env python3
"""
Sandbox Runner - 在沙箱内执行用户代码
支持工具调用拦截和 IPC 通信
支持循环模式用于容器复用
"""

import sys
import json
import asyncio
import uuid
from typing import Any

# IPC 协议标记
IPC_TOOL_CALL_START = "{IPC_TOOL_CALL_START}"
IPC_TOOL_CALL_END = "{IPC_TOOL_CALL_END}"
IPC_TOOL_RESULT_START = "{IPC_TOOL_RESULT_START}"
IPC_TOOL_RESULT_END = "{IPC_TOOL_RESULT_END}"
IPC_CODE_OUTPUT_START = "{IPC_CODE_OUTPUT_START}"
IPC_CODE_OUTPUT_END = "{IPC_CODE_OUTPUT_END}"

# 循环模式标记
LOOP_MODE = {str(loop_mode)}
EXIT_SIGNAL = "__EXIT_SESSION__"
READY_SIGNAL = "__READY__"

# 工具定义
TOOLS_INFO = {tools_info}

# 工具调用结果缓存
_pending_results: dict[str, asyncio.Future] = {{}}
_result_lock = asyncio.Lock()

# 持久化执行环境（用于状态保持）
_persistent_globals: dict = {{}}


def _send_tool_call(tool_name: str, arguments: dict) -> str:
    """发送工具调用请求到主进程"""
    call_id = str(uuid.uuid4())
    request = {{
        "call_id": call_id,
        "tool_name": tool_name,
        "arguments": arguments
    }}
    # 发送到 stderr（避免与 print 输出混淆）
    message = f"{{IPC_TOOL_CALL_START}}{{json.dumps(request)}}{{IPC_TOOL_CALL_END}}"
    print(message, file=sys.stderr, flush=True)
    return call_id


def _receive_tool_result(call_id: str, timeout: float = 30.0) -> Any:
    """从主进程接收工具调用结果"""
    # 从 stdin 读取结果
    while True:
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError(f"EOF while waiting for tool result: {{call_id}}")

        line = line.strip()
        if IPC_TOOL_RESULT_START in line and IPC_TOOL_RESULT_END in line:
            start = line.find(IPC_TOOL_RESULT_START) + len(IPC_TOOL_RESULT_START)
            end = line.find(IPC_TOOL_RESULT_END)
            result_json = line[start:end]
            result = json.loads(result_json)

            if result.get("call_id") == call_id:
                if result.get("error"):
                    raise RuntimeError(f"Tool error: {{result['error']}}")
                return result.get("result")


def _create_tool_function(tool_name: str):
    """创建工具调用函数"""
    async def tool_func(**kwargs) -> Any:
        # 发送调用请求
        call_id = _send_tool_call(tool_name, kwargs)
        # 等待结果（同步阻塞，因为需要与外部进程通信）
        # 使用线程池来避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _receive_tool_result(call_id)
        )
        return result

    return tool_func


# 动态创建工具函数
_tool_functions = {{}}
for tool_info in TOOLS_INFO:
    tool_name = tool_info["name"]
    _tool_functions[tool_name] = _create_tool_function(tool_name)


class OutputCapture:
    """捕获 print 输出"""
    def __init__(self):
        self.outputs = []
        self._original_stdout = sys.stdout

    def write(self, text):
        if text.strip():  # 忽略空行
            self.outputs.append(text)

    def flush(self):
        pass

    def get_output(self) -> str:
        return "".join(self.outputs)

    def clear(self):
        self.outputs = []


async def execute_user_code(code: str, exec_globals: dict) -> dict:
    """执行用户代码"""
    # 注入工具函数（每次执行都需要）
    for name, func in _tool_functions.items():
        exec_globals[name] = func

    # 捕获输出
    output_capture = OutputCapture()
    exec_globals["print"] = lambda *args, **kwargs: output_capture.write(
        " ".join(str(a) for a in args) + kwargs.get("end", "\\n")
    )

    # 包装用户代码为异步函数
    indented_code = "\\n".join("    " + line for line in code.split("\\n"))
    wrapped_code = f"""
async def __user_main__():
{{indented_code}}
"""

    try:
        # 先编译并执行定义
        exec(compile(wrapped_code, "<user_code>", "exec"), exec_globals)
        # 然后 await 执行用户的异步函数
        await exec_globals["__user_main__"]()
        # 清理临时函数
        if "__user_main__" in exec_globals:
            del exec_globals["__user_main__"]
        return {{
            "success": True,
            "output": output_capture.get_output(),
            "error": None
        }}
    except Exception as e:
        return {{
            "success": False,
            "output": output_capture.get_output(),
            "error": str(e)
        }}


def read_code_block() -> str | None:
    """从 stdin 读取一个代码块"""
    code_lines = []
    reading_code = False

    for line in sys.stdin:
        line = line.rstrip("\\n")

        # 检查退出信号
        if line == EXIT_SIGNAL:
            return None

        if line == "__CODE_START__":
            reading_code = True
            continue
        elif line == "__CODE_END__":
            break
        elif reading_code:
            code_lines.append(line)

    return "\\n".join(code_lines) if code_lines else ""


def main_single():
    """单次执行模式"""
    code = read_code_block()

    if code is None or not code:
        error_result = json.dumps({{"success": False, "output": "", "error": "No code provided"}})
        print(f"{{IPC_CODE_OUTPUT_START}}{{error_result}}{{IPC_CODE_OUTPUT_END}}", flush=True)
        return

    # 准备执行环境
    exec_globals = {{
        "__builtins__": __builtins__,
        "asyncio": asyncio,
        "json": json,
    }}

    # 执行代码
    try:
        result = asyncio.run(execute_user_code(code, exec_globals))
    except Exception as e:
        result = {{"success": False, "output": "", "error": str(e)}}

    # 输出结果
    print(f"{{IPC_CODE_OUTPUT_START}}{{json.dumps(result)}}{{IPC_CODE_OUTPUT_END}}", flush=True)


def main_loop():
    """循环执行模式（用于容器复用）"""
    # 持久化执行环境 - 变量在多次执行间保持
    exec_globals = {{
        "__builtins__": __builtins__,
        "asyncio": asyncio,
        "json": json,
    }}

    # 发送就绪信号
    print(f"{{READY_SIGNAL}}", file=sys.stderr, flush=True)

    while True:
        code = read_code_block()

        # 收到退出信号
        if code is None:
            break

        # 空代码
        if not code:
            error_result = json.dumps({{"success": False, "output": "", "error": "No code provided"}})
            print(f"{{IPC_CODE_OUTPUT_START}}{{error_result}}{{IPC_CODE_OUTPUT_END}}", flush=True)
            continue

        # 执行代码
        try:
            result = asyncio.run(execute_user_code(code, exec_globals))
        except Exception as e:
            result = {{"success": False, "output": "", "error": str(e)}}

        # 输出结果
        print(f"{{IPC_CODE_OUTPUT_START}}{{json.dumps(result)}}{{IPC_CODE_OUTPUT_END}}", flush=True)


def main():
    """主入口"""
    if LOOP_MODE:
        main_loop()
    else:
        main_single()


if __name__ == "__main__":
    main()
'''

    # ==================== 会话管理方法 ====================

    async def _create_session(self) -> SandboxSession:
        """创建新的会话（启动容器并保持运行）"""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        # 创建临时目录存放 runner 脚本
        temp_dir = tempfile.mkdtemp(prefix="ptc_sandbox_")
        os.chmod(temp_dir, 0o755)

        # 写入 runner 脚本（循环模式）
        runner_path = os.path.join(temp_dir, "runner.py")
        with open(runner_path, "w") as f:
            f.write(self._get_runner_script(loop_mode=True))
        os.chmod(runner_path, 0o644)

        # 准备容器配置
        image = self.config.custom_image or self.config.image
        container_config = {
            "image": image,
            "command": ["python", "-u", "/sandbox/runner.py"],  # -u 禁用缓冲
            "detach": True,
            "stdin_open": True,
            "network_disabled": self.config.network_disabled,
            "mem_limit": self.config.memory_limit,
            "cpu_period": self.config.cpu_period,
            "cpu_quota": self.config.cpu_quota,
            "read_only": self.config.read_only,
            "volumes": {
                temp_dir: {"bind": "/sandbox", "mode": "ro"}
            },
            "working_dir": self.config.working_dir,
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
        }

        # 创建并启动容器
        logger.info(f"Creating session container: {session_id}")
        container = self.docker_client.containers.create(**container_config)

        try:
            socket = container.attach_socket(
                params={"stdin": True, "stdout": True, "stderr": True, "stream": True}
            )
            socket._sock.setblocking(True)
            logger.debug(f"Socket attached to container: {container.id[:12]}")

            # Now start the container - socket will receive all output from the start
            container.start()
            logger.debug(f"Container started: {container.id[:12]}")

            # 等待就绪信号
            ready = await self._wait_for_ready(socket, timeout=10.0)
            if not ready:
                raise ContainerError("Container failed to become ready")

            session = SandboxSession(
                session_id=session_id,
                container=container,
                socket=socket,
                temp_dir=temp_dir,
                created_at=now,
                expires_at=now + timedelta(seconds=self.config.session_timeout_seconds),
                last_used_at=now,
                execution_count=0,
                is_busy=False
            )

            with self._sessions_lock:
                self._sessions[session_id] = session

            logger.info(f"Session created: {session_id}, expires at {session.expires_at}")
            return session

        except Exception as e:
            # 清理失败的容器
            try:
                container.stop(timeout=1)
                container.remove(force=True)
            except Exception:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ContainerError(f"Failed to create session: {e}")

    async def _wait_for_ready(self, socket, timeout: float = 10.0) -> bool:
        """等待容器就绪信号"""
        import select

        start_time = time_module.time()
        while time_module.time() - start_time < timeout:
            try:
                readable, _, _ = select.select([socket._sock], [], [], 0.1)
                if readable:
                    data = self._read_from_container(socket, timeout=0.1)
                    if data and "__READY__" in data:
                        logger.debug("Container ready signal received")
                        return True
            except Exception as e:
                logger.debug(f"Wait for ready error: {e}")
        return False

    def get_session(self, session_id: str) -> SandboxSession | None:
        """获取会话"""
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired():
                return session
            elif session and session.is_expired():
                # 会话已过期，异步清理
                asyncio.create_task(self.close_session(session_id))
                return None
            return None

    async def close_session(self, session_id: str) -> bool:
        """关闭并清理会话"""
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)

        if session is None:
            return False

        logger.info(f"Closing session: {session_id}")

        try:
            # 发送退出信号
            try:
                self._send_to_container(session.socket, "__EXIT_SESSION__\n")
            except Exception:
                pass

            # 停止并删除容器
            try:
                session.container.stop(timeout=5)
                session.container.remove(force=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup container: {e}")

            # 清理临时目录
            shutil.rmtree(session.temp_dir, ignore_errors=True)

            return True

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False

    async def close_all_sessions(self) -> None:
        """关闭所有会话"""
        with self._sessions_lock:
            session_ids = list(self._sessions.keys())

        for session_id in session_ids:
            await self.close_session(session_id)

    async def _cleanup_expired_sessions(self) -> None:
        """清理过期会话（后台任务）"""
        while self._cleanup_running:
            await asyncio.sleep(self.config.cleanup_interval_seconds)

            with self._sessions_lock:
                expired_ids = [
                    sid for sid, session in self._sessions.items()
                    if session.is_expired()
                ]

            for session_id in expired_ids:
                logger.info(f"Cleaning up expired session: {session_id}")
                await self.close_session(session_id)

    def start_cleanup_task(self) -> None:
        """启动后台清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
            logger.debug("Session cleanup task started")

    def stop_cleanup_task(self) -> None:
        """停止后台清理任务"""
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    @property
    def active_sessions(self) -> dict[str, dict]:
        """获取所有活跃会话信息"""
        with self._sessions_lock:
            return {
                sid: {
                    "session_id": sid,
                    "container_id": session.container.id[:12],
                    "created_at": session.created_at.isoformat(),
                    "expires_at": session.expires_at.isoformat(),
                    "execution_count": session.execution_count,
                    "is_busy": session.is_busy
                }
                for sid, session in self._sessions.items()
                if not session.is_expired()
            }

    # ==================== 执行方法 ====================

    async def execute(
        self,
        code: str,
        session_id: str | None = None,
        reuse_session: bool = False
    ) -> ExecutionResult | tuple[ExecutionResult, str]:
        """
        在沙箱中执行代码

        Args:
            code: 要执行的 Python 代码
            session_id: 可选，复用已有会话的 ID
            reuse_session: 是否启用会话复用模式
                - False: 单次执行模式，创建新容器，执行后销毁
                - True: 会话复用模式，保持容器运行

        Returns:
            - 单次模式: ExecutionResult
            - 复用模式: (ExecutionResult, session_id)
        """
        # 会话复用模式
        if session_id or reuse_session:
            return await self._execute_with_session(code, session_id)

        # 单次执行模式（向后兼容）
        return await self._execute_single(code)

    async def _execute_with_session(
        self,
        code: str,
        session_id: str | None = None
    ) -> tuple[ExecutionResult, str]:
        """使用会话执行代码（支持容器复用）"""
        start_time = time_module.time()
        tool_calls_count = 0

        # 获取或创建会话
        session = None
        if session_id:
            session = self.get_session(session_id)
            if session is None:
                logger.warning(f"Session {session_id} not found or expired, creating new")

        if session is None:
            session = await self._create_session()
            # 启动清理任务
            self.start_cleanup_task()

        # 标记会话繁忙
        session.is_busy = True

        try:
            # 刷新过期时间
            session.refresh(self.config.session_timeout_seconds)

            # 发送代码到容器
            code_payload = f"__CODE_START__\n{code}\n__CODE_END__\n"
            logger.debug(f"Sending code to session {session.session_id}")
            self._send_to_container(session.socket, code_payload)

            # 处理 IPC 通信
            final_result = None

            async def process_output():
                nonlocal tool_calls_count, final_result

                while True:
                    try:
                        data = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self._read_from_container(session.socket, timeout=0.1)
                        )

                        if data is None:
                            session.container.reload()
                            if session.container.status != "running":
                                raise ContainerError("Container stopped unexpectedly")
                            continue

                        logger.debug(f"Received: {data[:200]}..." if len(data) > 200 else f"Received: {data}")

                        lines = data.split("\n")
                        for line in lines:
                            if not line.strip():
                                continue

                            if IPC_TOOL_CALL_START in line and IPC_TOOL_CALL_END in line:
                                tool_calls_count += 1
                                await self._handle_tool_call(session.socket, line)

                            elif IPC_CODE_OUTPUT_START in line and IPC_CODE_OUTPUT_END in line:
                                start = line.find(IPC_CODE_OUTPUT_START) + len(IPC_CODE_OUTPUT_START)
                                end = line.find(IPC_CODE_OUTPUT_END)
                                result_json = line[start:end]
                                final_result = json.loads(result_json)
                                return

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing output: {e}")
                        raise

            try:
                await asyncio.wait_for(process_output(), timeout=self.config.timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(self.config.timeout_seconds, "Code execution")

            execution_time = (time_module.time() - start_time) * 1000
            session.execution_count += 1

            if final_result:
                result = ExecutionResult(
                    success=final_result.get("success", False),
                    stdout=final_result.get("output", ""),
                    stderr=final_result.get("error", "") or "",
                    return_code=0 if final_result.get("success") else 1,
                    tool_calls_count=tool_calls_count,
                    execution_time_ms=execution_time
                )
            else:
                result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="No output received",
                    return_code=1,
                    tool_calls_count=tool_calls_count,
                    execution_time_ms=execution_time
                )

            return result, session.session_id

        finally:
            session.is_busy = False

    async def _execute_single(self, code: str) -> ExecutionResult:
        """单次执行模式（原有逻辑，向后兼容）"""
        start_time = time_module.time()
        tool_calls_count = 0

        # 创建临时目录存放 runner 脚本
        temp_dir = tempfile.mkdtemp(prefix="ptc_sandbox_")

        try:
            # 设置目录权限为 world-readable (755)
            os.chmod(temp_dir, 0o755)

            # 写入 runner 脚本
            runner_path = os.path.join(temp_dir, "runner.py")
            with open(runner_path, "w") as f:
                f.write(self._get_runner_script())

            # 设置文件权限为 world-readable (644)
            os.chmod(runner_path, 0o644)

            # 准备容器配置
            image = self.config.custom_image or self.config.image
            container_config = {
                "image": image,
                "command": ["python", "/sandbox/runner.py"],
                "detach": True,
                "stdin_open": True,
                "network_disabled": self.config.network_disabled,
                "mem_limit": self.config.memory_limit,
                "cpu_period": self.config.cpu_period,
                "cpu_quota": self.config.cpu_quota,
                "read_only": self.config.read_only,
                "volumes": {
                    temp_dir: {"bind": "/sandbox", "mode": "ro"}
                },
                "working_dir": self.config.working_dir,
                # 安全选项
                "security_opt": ["no-new-privileges"],
                "cap_drop": ["ALL"],
            }

            # 创建并启动容器
            logger.info(f"Creating sandbox container with image: {image}")
            container = self.docker_client.containers.create(**container_config)

            try:
                container.start()
                logger.debug(f"Container started: {container.id[:12]}")

                # 附加到容器的 stdin/stdout
                socket = container.attach_socket(
                    params={"stdin": True, "stdout": True, "stderr": True, "stream": True}
                )
                # 保持阻塞模式，使用 select 来实现超时
                socket._sock.setblocking(True)

                # 发送代码到容器
                code_payload = f"__CODE_START__\n{code}\n__CODE_END__\n"
                logger.debug(f"Sending code to container:\n{code_payload}")
                self._send_to_container(socket, code_payload)

                # 处理 IPC 通信
                stdout_buffer = []
                stderr_buffer = []
                final_result = None

                async def process_container_output():
                    nonlocal tool_calls_count, final_result

                    while True:
                        try:
                            # 非阻塞读取
                            data = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self._read_from_container(socket, timeout=0.1)
                            )

                            if data is None:
                                # 检查容器是否还在运行
                                container.reload()
                                if container.status != "running":
                                    logger.debug(f"Container stopped, status: {container.status}")
                                    break
                                continue

                            logger.debug(f"Received data from container: {data[:200]}..." if len(data) > 200 else f"Received data: {data}")

                            # 解析输出
                            lines = data.split("\n")
                            for line in lines:
                                if not line.strip():
                                    continue

                                # 检查是否是工具调用请求
                                if IPC_TOOL_CALL_START in line and IPC_TOOL_CALL_END in line:
                                    tool_calls_count += 1
                                    await self._handle_tool_call(socket, line)

                                # 检查是否是最终输出
                                elif IPC_CODE_OUTPUT_START in line and IPC_CODE_OUTPUT_END in line:
                                    start = line.find(IPC_CODE_OUTPUT_START) + len(IPC_CODE_OUTPUT_START)
                                    end = line.find(IPC_CODE_OUTPUT_END)
                                    result_json = line[start:end]
                                    final_result = json.loads(result_json)
                                    return

                                else:
                                    # 普通输出
                                    stdout_buffer.append(line)

                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing output: {e}")
                            stderr_buffer.append(str(e))
                            break

                # 设置超时
                try:
                    await asyncio.wait_for(
                        process_container_output(),
                        timeout=self.config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        self.config.timeout_seconds,
                        "Code execution"
                    )

                # 获取容器退出码
                container.reload()
                exit_code = container.attrs.get("State", {}).get("ExitCode", -1)

                # 获取容器日志
                logs = container.logs(stdout=True, stderr=True).decode("utf-8")

                execution_time = (time_module.time() - start_time) * 1000

                if final_result:
                    return ExecutionResult(
                        success=final_result.get("success", False),
                        stdout=final_result.get("output", ""),
                        stderr=final_result.get("error", "") or "",
                        return_code=0 if final_result.get("success") else 1,
                        tool_calls_count=tool_calls_count,
                        execution_time_ms=execution_time
                    )
                else:
                    return ExecutionResult(
                        success=exit_code == 0,
                        stdout="\n".join(stdout_buffer),
                        stderr=logs if exit_code != 0 else "",
                        return_code=exit_code,
                        tool_calls_count=tool_calls_count,
                        execution_time_ms=execution_time
                    )

            finally:
                # 清理容器
                try:
                    container.stop(timeout=5)
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup container: {e}")

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _handle_tool_call(self, socket, line: str) -> None:
        """处理工具调用请求"""
        try:
            # 解析请求
            start = line.find(IPC_TOOL_CALL_START) + len(IPC_TOOL_CALL_START)
            end = line.find(IPC_TOOL_CALL_END)
            request_json = line[start:end]
            request = json.loads(request_json)

            call_id = request["call_id"]
            tool_name = request["tool_name"]
            arguments = request["arguments"]

            logger.info(f"Tool call: {tool_name}({arguments})")

            # 执行工具
            try:
                result = await self.tool_registry.execute_async(tool_name, arguments)
                response = ToolCallResponse(
                    call_id=call_id,
                    success=True,
                    result=result
                )
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                response = ToolCallResponse(
                    call_id=call_id,
                    success=False,
                    error=str(e)
                )

            # 发送结果回容器
            response_data = {
                "call_id": response.call_id,
                "result": response.result,
                "error": response.error
            }
            response_line = f"{IPC_TOOL_RESULT_START}{json.dumps(response_data)}{IPC_TOOL_RESULT_END}\n"
            self._send_to_container(socket, response_line)

        except Exception as e:
            logger.error(f"Error handling tool call: {e}")
            raise IPCError(f"Failed to handle tool call: {e}")

    def _send_to_container(self, socket, data: str) -> None:
        """发送数据到容器"""
        try:
            socket._sock.sendall(data.encode("utf-8"))
        except Exception as e:
            raise IPCError(f"Failed to send data to container: {e}")

    def _read_from_container(self, socket, timeout: float = 1.0) -> str | None:
        """从容器读取数据（处理 Docker 多路复用流）"""
        import select
        import struct

        try:
            readable, _, _ = select.select([socket._sock], [], [], timeout)
            if not readable:
                return None

            # Docker 多路复用流格式:
            # - 1 byte: stream type (0=stdin, 1=stdout, 2=stderr)
            # - 3 bytes: unused
            # - 4 bytes: payload size (big-endian)
            # - N bytes: payload
            result_parts = []

            while True:
                # 尝试读取 header
                readable, _, _ = select.select([socket._sock], [], [], 0.01)
                if not readable:
                    break

                header = socket._sock.recv(8)
                if not header or len(header) < 8:
                    break

                # 解析 header
                stream_type = header[0]
                payload_size = struct.unpack('>I', header[4:8])[0]

                if payload_size == 0:
                    continue

                # 读取 payload
                payload = b''
                while len(payload) < payload_size:
                    chunk = socket._sock.recv(payload_size - len(payload))
                    if not chunk:
                        break
                    payload += chunk

                if payload:
                    try:
                        result_parts.append(payload.decode("utf-8"))
                    except UnicodeDecodeError:
                        # 跳过无法解码的数据
                        pass

            return "".join(result_parts) if result_parts else None

        except Exception as e:
            logger.debug(f"Read from container error: {e}")
            return None

    def build_custom_image(
        self,
        base_image: str = "python:3.11-slim",
        additional_packages: list[str] | None = None,
        pip_packages: list[str] | None = None,
        tag: str = "ptc-sandbox:latest"
    ) -> str:
        """
        构建自定义沙箱镜像

        Args:
            base_image: 基础镜像
            additional_packages: 额外的系统包
            pip_packages: 额外的 Python 包
            tag: 镜像标签

        Returns:
            构建的镜像 ID
        """
        dockerfile_content = f"""FROM {base_image}

# 安全设置
RUN useradd -m -s /bin/bash sandbox && \\
    mkdir -p /workspace && \\
    chown sandbox:sandbox /workspace

"""
        if additional_packages:
            dockerfile_content += f"""
# 安装系统包
RUN apt-get update && apt-get install -y --no-install-recommends \\
    {' '.join(additional_packages)} && \\
    rm -rf /var/lib/apt/lists/*

"""
        if pip_packages:
            dockerfile_content += f"""
# 安装 Python 包
RUN pip install --no-cache-dir {' '.join(pip_packages)}

"""
        dockerfile_content += """
# 切换到非 root 用户
USER sandbox
WORKDIR /workspace
"""

        # 创建临时目录构建镜像
        temp_dir = tempfile.mkdtemp(prefix="ptc_build_")
        try:
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            logger.info(f"Building custom sandbox image: {tag}")
            image, _ = self.docker_client.images.build(
                path=temp_dir,
                tag=tag,
                rm=True
            )

            # 更新配置使用新镜像
            self.config.custom_image = tag

            return image.id

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
