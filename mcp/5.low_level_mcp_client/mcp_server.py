import os
import logging
from pathlib import Path
from fastmcp import FastMCP

# Suppress FastMCP banner and all output to stdout
os.environ["FASTMCP_LOG_LEVEL"] = "WARNING"  # suppress banner

# Suppress verbose FastMCP INFO logging
logging.getLogger("fastmcp").setLevel(logging.WARNING)

mcp = FastMCP("lab-server", log_level="WARNING")

BASE_DIR = Path.cwd()

@mcp.tool()
def echo(text: str) -> str:
    """Echo back the input text."""
    return f"Echo: {text}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    file_path = BASE_DIR / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Successfully wrote to {path}"

@mcp.resource("file://resources/{filename}")
def read_resource_file(filename: str) -> str:
    """Read a file from the resources directory."""
    file_path = BASE_DIR / "resources" / filename
    if not file_path.exists():
        return f"File not found: {filename}"
    return file_path.read_text(encoding="utf-8")

@mcp.prompt()
def review_file(filename: str) -> str:
    """Generate a prompt to review a file's contents."""
    return f"""Please review the file '{filename}' and provide:

A summary of its contents
Key points or sections
Any suggestions for improvement
Overall quality assessment

Use the appropriate tools to read the file if needed."""

# This ensures that the server runs and listens for MCP client connections via standard input/output (STDIO).
if __name__ == "__main__":
    mcp.run(transport="stdio")
