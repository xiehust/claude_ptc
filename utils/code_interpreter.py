"""
Code Interpreter Tool for Claude Agent SDK.

Re-implementation of the Strands Agent SDK AgentCore Code Interpreter
(https://github.com/strands-agents/tools/blob/main/src/strands_tools/code_interpreter/agent_core_code_interpreter.py)
adapted for use with Claude Agent SDK's @tool decorator.

This tool provides code execution in isolated AWS Bedrock AgentCore sandbox environments
with support for Python, JavaScript, and TypeScript, plus file operations.
"""

import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from utils.tool_decorator import tool, get_tool_configs

logger = logging.getLogger(__name__)


# Module-level session cache - persists across tool invocations
_session_mapping: dict[str, str] = {}  # user_session_name -> aws_session_id
_session_clients: dict[str, Any] = {}  # user_session_name -> client instance


class LanguageType(str, Enum):
    """Supported programming languages for code execution."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


@dataclass
class SessionInfo:
    """Information about a code interpreter session."""
    session_id: str  # AWS CI session ID
    description: str
    client: Any  # BedrockAgentCoreCodeInterpreterClient


class CodeInterpreterTool:
    """
    Code Interpreter tool manager for Claude Agent SDK.

    Wraps AWS Bedrock AgentCore Code Interpreter with session management
    and provides tools compatible with Claude Agent SDK's @tool decorator.

    Features:
    - Multi-language support (Python, JavaScript, TypeScript)
    - Persistent session management with auto-reconnection
    - File operations (read, write, list, remove)
    - Shell command execution

    Usage:
        from src.tools.code_interpreter import CodeInterpreterTool, create_code_interpreter_tools

        # Create tool instance
        ci_tool = CodeInterpreterTool(region="us-west-2")

        # Get MCP server with all tools
        mcp_server = create_code_interpreter_tools(region="us-west-2")

        # Use with ClaudeAgentOptions
        options = ClaudeAgentOptions(
            mcp_servers={"code_interpreter": mcp_server},
            allowed_tools=["mcp__code_interpreter__execute_code", ...]
        )
    """

    def __init__(
        self,
        region: str | None = None,
        identifier: str | None = None,
        session_name: str | None = None,
        auto_create: bool = True,
        persist_sessions: bool = True,
    ) -> None:
        """
        Initialize the Code Interpreter tool.

        Args:
            region: AWS region for the code interpreter service (e.g., "us-west-2")
            identifier: Custom code interpreter identifier. Defaults to "aws.codeinterpreter.v1"
            session_name: Session identifier for tracking. None generates random ID.
            auto_create: Automatically create sessions if they don't exist. Default: True
            persist_sessions: Prevent session cleanup on destruction. Default: True
        """
        self.region = region or self._resolve_region()
        self.identifier = identifier or "aws.codeinterpreter.v1"
        self.auto_create = auto_create
        self.persist_sessions = persist_sessions

        if session_name is None:
            self.default_session = f"session-{uuid.uuid4().hex[:12]}"
        else:
            self.default_session = session_name

        self._sessions: dict[str, SessionInfo] = {}
        self._client_class = None  # Lazy load AWS client

        logger.info(
            f"Initialized CodeInterpreterTool with session='{self.default_session}', "
            f"identifier='{self.identifier}', auto_create={auto_create}"
        )

    def _resolve_region(self) -> str:
        """Resolve AWS region from environment or default."""
        import os
        return (
            os.environ.get("AWS_REGION") or
            os.environ.get("AWS_DEFAULT_REGION") or
            "us-west-2"
        )

    def _get_client_class(self) -> type:
        """Lazy load the Bedrock AgentCore client class."""
        if self._client_class is None:
            try:
                from bedrock_agentcore.tools.code_interpreter_client import (
                    CodeInterpreter as BedrockAgentCoreCodeInterpreterClient
                )
                self._client_class = BedrockAgentCoreCodeInterpreterClient
            except ImportError as e:
                raise ImportError(
                    "bedrock-agentcore package is required for Code Interpreter. "
                    "Install with: pip install bedrock-agentcore"
                ) from e
        return self._client_class

    def _ensure_session(self, session_name: str | None) -> tuple[str, dict[str, Any] | None]:
        """
        Ensure a session exists, creating if necessary.

        Returns:
            Tuple of (session_name, error_dict or None)
        """
        target_session = session_name or self.default_session

        logger.debug(f"Ensuring session: {target_session}")

        # Check local cache first
        if target_session in self._sessions:
            logger.debug(f"Using cached session: {target_session}")
            return target_session, None

        # Check module-level cache for AWS session ID
        aws_session_id = _session_mapping.get(target_session)

        if aws_session_id and target_session in _session_clients:
            # Found in module cache - try to reconnect
            logger.debug(f"Found session in module cache: {target_session} -> {aws_session_id}")

            try:
                client = _session_clients[target_session]
                session_info = client.get_session(
                    interpreter_id=self.identifier,
                    session_id=aws_session_id
                )

                if session_info.get("status") == "READY":
                    self._sessions[target_session] = SessionInfo(
                        session_id=aws_session_id,
                        description="Reconnected via module cache",
                        client=client
                    )
                    logger.info(f"Reconnected to existing session: {target_session}")
                    return target_session, None
                else:
                    logger.warning(f"Session {target_session} not READY, removing from cache")
                    del _session_mapping[target_session]
                    del _session_clients[target_session]
            except Exception as e:
                logger.debug(f"Session reconnection failed: {e}")
                if target_session in _session_mapping:
                    del _session_mapping[target_session]
                if target_session in _session_clients:
                    del _session_clients[target_session]

        # Session not found - create new if auto_create enabled
        if self.auto_create:
            logger.info(f"Auto-creating session: {target_session}")
            result = self._init_session(target_session, "Auto-initialized session")
            if result.get("status") != "success":
                return target_session, result
            return target_session, None

        # auto_create=False and session doesn't exist
        error_msg = f"Session '{target_session}' not found. Create it first using init_session."
        logger.debug(error_msg)
        return target_session, {"status": "error", "content": [{"type": "text", "text": error_msg}]}

    def _init_session(self, session_name: str, description: str) -> dict[str, Any]:
        """Initialize a new code interpreter session."""
        logger.info(f"Initializing session: {session_name} - {description}")

        # Check if session already exists
        if session_name in self._sessions:
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"Session '{session_name}' already exists"}]
            }

        if session_name in _session_mapping:
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"Session '{session_name}' is already in use"}]
            }

        try:
            ClientClass = self._get_client_class()
            client = ClientClass(region=self.region)

            # Start session with identifier and name
            client.start(identifier=self.identifier, name=session_name)

            aws_session_id = client.session_id

            # Store in module-level cache
            _session_mapping[session_name] = aws_session_id
            _session_clients[session_name] = client

            # Store session info locally
            self._sessions[session_name] = SessionInfo(
                session_id=aws_session_id,
                description=description,
                client=client
            )

            logger.info(f"Initialized session: {session_name} (AWS ID: {aws_session_id})")

            return {
                "status": "success",
                "content": [{
                    "type": "text",
                    "text": f"Session '{session_name}' initialized successfully (ID: {aws_session_id})"
                }]
            }
        except Exception as e:
            logger.error(f"Failed to initialize session '{session_name}': {e}")
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"Failed to initialize session: {str(e)}"}]
            }

    def _process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process AWS response into tool result format."""
        if "stream" in response:
            event_stream = response["stream"]
            for event in event_stream:
                if "result" in event:
                    result = event["result"]
                    is_error = response.get("isError", False)
                    content = str(result.get("content", ""))
                    return {
                        "status": "success" if not is_error else "error",
                        "content": [{"type": "text", "text": content}]
                    }
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"Failed to process response: {response}"}]
            }
        return response

    # Tool implementations

    def execute_code(
        self,
        code: str,
        language: str = "python",
        session_name: str | None = None,
        clear_context: bool = False,
    ) -> dict[str, Any]:
        """Execute code in a sandbox session."""
        session_name, error = self._ensure_session(session_name)
        if error:
            return error

        logger.debug(f"Executing {language} code in session '{session_name}'")

        try:
            params = {
                "code": code,
                "language": language,
                "clearContext": clear_context
            }
            response = self._sessions[session_name].client.invoke("executeCode", params)
            return self._process_response(response)
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"Code execution failed: {str(e)}"}]
            }

    def execute_command(
        self,
        command: str,
        session_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command in a sandbox session."""
        session_name, error = self._ensure_session(session_name)
        if error:
            return error

        logger.debug(f"Executing command in session '{session_name}'")

        try:
            params = {"command": command}
            response = self._sessions[session_name].client.invoke("executeCommand", params)
            return self._process_response(response)
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"Command execution failed: {str(e)}"}]
            }

    def read_files(
        self,
        paths: list[str],
        session_name: str | None = None,
    ) -> dict[str, Any]:
        """Read files from a sandbox session."""
        session_name, error = self._ensure_session(session_name)
        if error:
            return error

        logger.debug(f"Reading files from session '{session_name}'")

        try:
            params = {"paths": paths}
            response = self._sessions[session_name].client.invoke("readFiles", params)
            return self._process_response(response)
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"File read failed: {str(e)}"}]
            }

    def write_files(
        self,
        files: list[dict[str, str]],
        session_name: str | None = None,
    ) -> dict[str, Any]:
        """Write files to a sandbox session."""
        session_name, error = self._ensure_session(session_name)
        if error:
            return error

        logger.debug(f"Writing {len(files)} files to session '{session_name}'")

        try:
            params = {"content": files}
            response = self._sessions[session_name].client.invoke("writeFiles", params)
            return self._process_response(response)
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"File write failed: {str(e)}"}]
            }

    def list_files(
        self,
        path: str = ".",
        session_name: str | None = None,
    ) -> dict[str, Any]:
        """List files in a sandbox session directory."""
        session_name, error = self._ensure_session(session_name)
        if error:
            return error

        logger.debug(f"Listing files in session '{session_name}'")

        try:
            params = {"path": path}
            response = self._sessions[session_name].client.invoke("listFiles", params)
            return self._process_response(response)
        except Exception as e:
            logger.error(f"File list failed: {e}")
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"File list failed: {str(e)}"}]
            }

    def remove_files(
        self,
        paths: list[str],
        session_name: str | None = None,
    ) -> dict[str, Any]:
        """Remove files from a sandbox session."""
        session_name, error = self._ensure_session(session_name)
        if error:
            return error

        logger.debug(f"Removing files from session '{session_name}'")

        try:
            params = {"paths": paths}
            response = self._sessions[session_name].client.invoke("removeFiles", params)
            return self._process_response(response)
        except Exception as e:
            logger.error(f"File remove failed: {e}")
            return {
                "status": "error",
                "content": [{"type": "text", "text": f"File remove failed: {str(e)}"}]
            }

    def list_sessions(self) -> dict[str, Any]:
        """List all sessions created by this tool instance."""
        sessions_info = []
        for name, info in self._sessions.items():
            sessions_info.append({
                "sessionName": name,
                "description": info.description,
                "sessionId": info.session_id,
            })

        return {
            "status": "success",
            "content": [{
                "type": "text",
                "text": f"Active sessions: {len(sessions_info)}\n" +
                       "\n".join([f"- {s['sessionName']}: {s['description']}" for s in sessions_info])
            }]
        }

    def cleanup(self) -> None:
        """Clean up sessions (call on shutdown if persist_sessions=False)."""
        if not self.persist_sessions:
            logger.info("Cleaning up Code Interpreter sessions")
            for session_name, session in list(self._sessions.items()):
                try:
                    session.client.stop()
                    logger.debug(f"Stopped session: {session_name}")
                except Exception as e:
                    logger.debug(f"Session {session_name} cleanup skipped: {e}")
            self._sessions.clear()


# Global tool instance for sharing across decorated functions
_tool_instance: CodeInterpreterTool | None = None


def _get_tool_instance(region: str | None = None) -> CodeInterpreterTool:
    """Get or create the global tool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = CodeInterpreterTool(region=region)
    return _tool_instance


# Tool functions using @tool decorator

@tool(
    "init_session",
    "Initialize a new code interpreter sandbox session. Creates an isolated environment for code execution with its own file system and state.",
    {
        "session_name": str,
        "description": str,
    }
)
async def init_session(args: dict[str, Any]) -> dict[str, Any]:
    """Initialize a new code interpreter session."""
    tool_instance = _get_tool_instance()
    result = tool_instance._init_session(
        session_name=args["session_name"],
        description=args["description"]
    )
    return {"content": result["content"]}


@tool(
    "code_execution_20250825",
    """Execute code in an isolated sandbox environment. Supports Python, JavaScript, and TypeScript.

Examples:
- Python: print("Hello"), import pandas as pd

The session maintains state between executions unless clear_context is True.

This enviroment comes with a comprehensive set of **pre-installed Python libraries** available immediately without requiring additional installation. These libraries are organized into 11 major categories:

### **1. Data Analysis and Visualization** (27 libraries)
Core libraries for data manipulation and visualization including:
- **pandas**, **numpy** - Data manipulation and numerical computing
- **matplotlib**, **plotly**, **bokeh** - Visualization tools
- **scipy**, **statsmodels**, **sympy** - Scientific and statistical computing
- **pyarrow**, **numba**, **numexpr** - Performance optimization

### **2. Machine Learning and AI** (27 libraries)
Comprehensive ML/AI toolkit including:
- **scikit-learn** - Machine learning algorithms
- **torch**, **torchvision**, **torchaudio** - PyTorch deep learning
- **xgboost** - Gradient boosting
- **spacy**, **nltk**, **textblob** - Natural language processing
- **openai** - OpenAI API client
- **scikit-image** - Image processing

### **3. Mathematical and Optimization** (10 libraries)
Specialized math libraries:
- **cvxpy**, **ortools**, **pulp** - Optimization and linear programming
- **networkx**, **igraph** - Network/graph analysis
- **z3-solver** - Theorem proving

### **4. Web and API Development** (50+ libraries)
Full web development stack:
- **FastAPI**, **Flask**, **Django** - Web frameworks
- **requests**, **httpx** - HTTP clients
- **beautifulsoup4** - Web scraping
- **uvicorn**, **gunicorn**, **Hypercorn** - ASGI/WSGI servers
- WebSocket, HTTP/2, async support libraries

### **5. Cloud and Database** (6 libraries)
- **boto3** - AWS SDK
- **duckdb** - In-process SQL database
- **SQLAlchemy** - SQL ORM
- **pymongo**, **redis**, **psycopg2-binary** - Database drivers

### **6. File Processing and Documents** (20 libraries)
Office document handling:
- **openpyxl**, **xlrd**, **XlsxWriter** - Excel files
- **PyPDF2**, **pdfplumber**, **pypdfium2** - PDF processing
- **python-docx**, **docx2txt** - Word documents
- **reportlab**, **fpdf** - PDF generation

### **7. Image and Media Processing** (15 libraries)
Multimedia capabilities:
- **pillow** - Image processing
- **opencv-python** - Computer vision
- **moviepy**, **ffmpeg-python** - Video editing
- **pydub**, **gtts** - Audio manipulation

### **8. Development Tools and Utilities** (70+ libraries)
Extensive developer tooling:
- **pydantic** - Data validation
- **click**, **typer** - CLI frameworks
- **tqdm** - Progress bars
- **loguru**, **rich** - Logging and formatting
- **cryptography**, **bcrypt** - Security
- **ipython** - Interactive shell
- **Faker** - Fake data generation

### **9. Text and Markup Processing** (11 libraries)
- **markdown-it-py**, **markdown2** - Markdown processing
- **lxml** - XML/HTML processing
- **regex**, **chardet** - Text utilities

### **10. Geospatial and Mapping** (3 libraries)
- **shapely**, **pyshp**, **branca** - Geographic data manipulation

### **11. Document Processing Support** (12 libraries)
Additional document tools:
- **python-pptx** - PowerPoint files
- **pypandoc** - Document conversion
- **tabula-py** - PDF table extraction
- **xmltodict** - XML conversion

This extensive library collection makes the Code Interpreter suitable for diverse tasks from data analysis and machine learning to web scraping and document generation, all within a secure, managed environment.

""",
    {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The code to execute"
            },
            "language": {
                "type": "string",
                "enum": ["python"],
                "default": "python",
                "description": "Programming language"
            },
            "session_name": {
                "type": "string",
                "description": "Optional session name. If not provided, uses default session (auto-created)."
            },
            "clear_context": {
                "type": "boolean",
                "default": False,
                "description": "Clear execution context before running"
            }
        },
        "required": ["code"]
    }
)
async def execute_code(args: dict[str, Any]) -> dict[str, Any]:
    """Execute code in a sandbox session."""
    tool_instance = _get_tool_instance()
    result = tool_instance.execute_code(
        code=args["code"],
        language=args.get("language", "python"),
        session_name=args.get("session_name"),
        clear_context=args.get("clear_context", False)
    )
    return {"content": result["content"]}


@tool(
    "execute_command",
    "Execute a shell command in the sandbox environment. Use for system operations like installing packages (pip install), running scripts, or file management.",
    {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute"
            },
            "session_name": {
                "type": "string",
                "description": "Optional session name"
            }
        },
        "required": ["command"]
    }
)
async def execute_command(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a shell command in a sandbox session."""
    tool_instance = _get_tool_instance()
    result = tool_instance.execute_command(
        command=args["command"],
        session_name=args.get("session_name")
    )
    return {"content": result["content"]}


@tool(
    "read_files",
    "Read the contents of one or more files from the sandbox file system.",
    {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to read"
            },
            "session_name": {
                "type": "string",
                "description": "Optional session name"
            }
        },
        "required": ["paths"]
    }
)
async def read_files(args: dict[str, Any]) -> dict[str, Any]:
    """Read files from a sandbox session."""
    tool_instance = _get_tool_instance()
    result = tool_instance.read_files(
        paths=args["paths"],
        session_name=args.get("session_name")
    )
    return {"content": result["content"]}


@tool(
    "write_files",
    "Create or update files in the sandbox file system.",
    {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "text": {"type": "string", "description": "File content"}
                    },
                    "required": ["path", "text"]
                },
                "description": "List of files to write with path and text content"
            },
            "session_name": {
                "type": "string",
                "description": "Optional session name"
            }
        },
        "required": ["files"]
    }
)
async def write_files(args: dict[str, Any]) -> dict[str, Any]:
    """Write files to a sandbox session."""
    tool_instance = _get_tool_instance()
    result = tool_instance.write_files(
        files=args["files"],
        session_name=args.get("session_name")
    )
    return {"content": result["content"]}


@tool(
    "list_files",
    "List files and directories in the sandbox file system.",
    {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "default": ".",
                "description": "Directory path to list (defaults to current directory)"
            },
            "session_name": {
                "type": "string",
                "description": "Optional session name"
            }
        }
    }
)
async def list_files(args: dict[str, Any]) -> dict[str, Any]:
    """List files in a sandbox session directory."""
    tool_instance = _get_tool_instance()
    result = tool_instance.list_files(
        path=args.get("path", "."),
        session_name=args.get("session_name")
    )
    return {"content": result["content"]}


@tool(
    "remove_files",
    "Delete files from the sandbox file system. Use with caution - this permanently removes files.",
    {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to remove"
            },
            "session_name": {
                "type": "string",
                "description": "Optional session name"
            }
        },
        "required": ["paths"]
    }
)
async def remove_files(args: dict[str, Any]) -> dict[str, Any]:
    """Remove files from a sandbox session."""
    tool_instance = _get_tool_instance()
    result = tool_instance.remove_files(
        paths=args["paths"],
        session_name=args.get("session_name")
    )
    return {"content": result["content"]}


@tool(
    "list_sessions",
    "List all active code interpreter sessions.",
    {}
)
async def list_sessions() -> dict[str, Any]:
    """List all sessions created by this tool instance."""
    tool_instance = _get_tool_instance()
    result = tool_instance.list_sessions()
    return {"content": result["content"]}


def create_code_interpreter_tools(
    region: str | None = None,
    server_name: str = "code_interpreter",
    server_version: str = "1.0.0",
) -> Any:
    """
    Create an Tool Configs all Code Interpreter tools.

    Args:
        region: AWS region for the code interpreter (default: from env or us-west-2)
        server_name: Name for the MCP server (default: "code_interpreter")
        server_version: Version for the MCP server (default: "1.0.0")

    Returns:
        MCP server instance with all code interpreter tools
    """
    # Initialize the global tool instance with region
    global _tool_instance
    _tool_instance = CodeInterpreterTool(region=region)

    # Create config with all tools
    return dict(
        name=server_name,
        version=server_version,
        tool_config = get_tool_configs([
            init_session,
            execute_code,
            execute_command,
            read_files,
            write_files,
            list_files,
            remove_files,
            list_sessions,
        ]),
        tools=dict(
            init_session=init_session,
            execute_code=execute_code,
            execute_command=execute_command,
            read_files=read_files,
            write_files=write_files,
            list_files=list_files,
            remove_files=remove_files,
            list_sessions=list_sessions,
        )
    )


# Convenience function to get individual tools
def get_code_interpreter_tools() -> list:
    """Get list of all code interpreter tool functions."""
    return [
        init_session,
        execute_code,
        execute_command,
        read_files,
        write_files,
        list_files,
        remove_files,
        list_sessions,
    ]
