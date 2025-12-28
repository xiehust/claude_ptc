"""自定义异常类"""


class SandboxError(Exception):
    """沙箱相关错误的基类"""
    pass


class ToolExecutionError(SandboxError):
    """工具执行失败"""
    def __init__(self, tool_name: str, message: str, original_error: Exception | None = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class TimeoutError(SandboxError):
    """执行超时"""
    def __init__(self, timeout_seconds: float, operation: str = "code execution"):
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(f"{operation} timed out after {timeout_seconds} seconds")


class CodeExecutionError(SandboxError):
    """代码执行错误"""
    def __init__(self, message: str, stdout: str = "", stderr: str = "", return_code: int = -1):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        super().__init__(message)


class ContainerError(SandboxError):
    """Docker 容器相关错误"""
    pass


class IPCError(SandboxError):
    """进程间通信错误"""
    pass
