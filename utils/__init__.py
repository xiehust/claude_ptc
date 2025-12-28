"""
Shared utilities for Claude tool use cookbooks.

This package contains reusable components for creating cookbook demonstrations:
- visualize: Rich terminal visualization for Claude API responses
- team_expense_api: Example mock API for team expense management demonstrations
"""

from .visualize import show_response, visualize
from .code_interpreter import (
    CodeInterpreterTool,
    create_code_interpreter_tools,
    execute_code,
)

__all__ = [
    "CodeInterpreterTool",
    "create_code_interpreter_tools",
    "visualize", 
    "show_response",
    "execute_code"
]
