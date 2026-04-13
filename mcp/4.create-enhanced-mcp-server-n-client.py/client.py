import asyncio
import sys
import json
from urllib.parse import quote
from typing import Optional, Dict, Any, List, Union
from contextlib import AsyncExitStack

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult

# from anthropic import Anthropic # Use this if you have a Claude API key
from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # Load environment variables from .env file

MODEL_ID = "gpt-4o-mini" # OpenAI model identifier for API calls
# MODEL_ID = "claude-sonnet-4-5-20250929" # Use this if you have a Claude API key

class MCPClient:
    """MCP (Model Context Protocol) client for interacting with MCP servers and OpenAI.

    This client manages connections to MCP servers, handles tool execution,
    and provides an interactive interface for querying OpenAI with MCP tools.
    """

    def __init__(self):
        """Initialize the MCP client with session management and OpenAI API client.

        Sets up:
        - AsyncExitStack for managing async context managers
        - OpenAI client for API interactions
        """
        
        # AsyncExitStack tracks async context managers (like the MCP client connection).
        # Instead of wrapping everything in a fixed "async with" block, we register
        # connections dynamically via enter_async_context() and close all of them at
        # once by calling exit_stack.aclose() in cleanup().
        self.exit_stack = AsyncExitStack()

        # self.anthropic = Anthropic() # Use this if you have a Claude API key
        self.openai = OpenAI()
        
        self.client = None # its defined in connect_to_server

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server via stdio transport.

        Establishes a connection to an MCP server by launching the server script
        as a subprocess and communicating via stdin/stdout.

        Args:
            server_script_path: Path to the server script (.py, .js, or .ts file)

        Raises:
            ValueError: If server_script_path is not a .py, .js, or .ts file
        """
        # Determine script type based on file extension
        is_python = server_script_path.endswith('.py')
        is_ts = server_script_path.endswith('.ts')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_ts or is_js):
            raise ValueError("Server script must be a .py, .js, or .ts file")

        # FastMCP Client — handles the full MCP protocol (tools, resources, prompts).
        # Passing a file path tells it to use stdio transport internally (launches server as subprocess).
        self.client = Client(
            server_script_path,
            elicitation_handler=self.handle_elicitation, # called when server needs user input
            progress_handler=self.handle_progress, # called on progress updates
            message_handler=self.handle_message # called on server notifications
        )

        await self.exit_stack.enter_async_context(self.client)

    async def handle_elicitation(self, message: str, response_type: type, params, context):
        """Handle elicitation requests from the MCP server.

        When the server needs user input, this handler prompts the user,
        collects their response, and returns it in the expected format.

        Args:
            message: The question or prompt from the server
            response_type: Pydantic model defining the expected response structure
            params: Additional parameters for the elicitation
            context: Elicitation context information

        Returns:
            ElicitResult with action="decline" if no response, or response_type instance with user input
        """
        print(f"Server asks: {message}")

        user_data = {}
        for field_name, field_type in response_type.__annotations__.items():
            user_input = input(f"Enter value for '{field_name}' ({field_type.__name__}): ").strip()
            if not user_input:
                return ElicitResult(action="decline")

            user_data[field_name] = user_input

        # Return the structured response object
        return response_type(**user_data)

    async def handle_progress(self, progress: float, total: float | None, message: str | None) -> None:
        """Handle progress notifications from the MCP server.

        Displays progress updates to the user, showing percentage complete if total is provided.

        Args:
            progress: Current progress value
            total: Total expected progress value (None if unknown)
            message: Optional descriptive message about current progress
        """
        if total is not None:
            percentage = (progress / total) * 100
            print(f"Progress: {percentage:.1f}% - {message or ''}")
        else:
            print(f"Progress: {progress} - {message or ''}")

    async def handle_message(self, message):
        """Handle notification messages from the MCP server.

        Processes server notifications such as tool list changes or resource updates
        and displays appropriate messages to the user.

        Args:
            message: MCP notification message from the server
        """
        if hasattr(message, 'root'):
            method = message.root.method
            print(f"Received: {method}")

            if method == "notifications/tools/list_changed":
                print("Tools have changed - might want to refresh tool cache")
            elif method == "notifications/resources/list_changed":
                print("Resources have changed")


    async def _get_tools(self) -> List[Dict[str, Any]]:
        """Retrieve available tools from the MCP server.

        Fetches the list of tools exposed by the server and formats them
        for use with the OpenAI API.

        Returns:
            List of tool definitions with name, description, and input schema
        """
        tools_response = await self.client.list_tools()

        # Use this if you have a Claude API key
        # claude_tools = [
        #     {
        #         "name": tool.name,
        #         "description": tool.description or "MCP Tool",
        #         "input_schema": tool.inputSchema,
        #     }
        #     for tool in tools_response
        # ]

        # Format tools for OpenAI API compatibility
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "MCP Tool",
                    "parameters": tool.inputSchema,
                }
            }
            for tool in tools_response
        ]

        return openai_tools

    async def _get_prompts(self):
        """Retrieve available prompts from the MCP server.

        Returns:
            PromptsResponse containing available prompt templates
        """

        prompts_response = await self.client.list_prompts()

        return prompts_response

    # Currently unused — but useful if you want the client to auto-discover static resources
    # (e.g. "dir://.") instead of hardcoding URIs in read_dir(). Call this in menu() to let
    # the user pick from whatever resources the server exposes.
    async def _get_resources(self):
        """Retrieve available resources from the MCP server.

        Returns:
            ResourcesResponse containing available resources
        """
        resources_response = await self.client.list_resources()
        return resources_response

    # Currently unused — but useful if you want the client to auto-discover resource templates
    # (e.g. "file:///{file_name}") instead of hardcoding URIs in read_file(). Call this in menu()
    # to let the user pick a template and fill in the parameter dynamically.
    async def _get_resource_templates(self):
        """Retrieve available resource templates from the MCP server.

        Returns:
            ResourceTemplatesResponse containing available resource templates
        """
        resource_templates_response = await self.client.list_resource_templates()
        return resource_templates_response

    # Use this if you have a Claude API key
    # async def process_query(self, query: str) -> str:
    #     """Process a query using Claude with access to MCP server tools."""
    #     messages = [{"role": "user", "content": query}]
    #     available_tools = await self._get_tools()
    #     response = self.anthropic.messages.create(
    #         model=MODEL_ID, max_tokens=4096, messages=messages, tools=available_tools
    #     )
    #     while response.stop_reason == "tool_use":
    #         messages.append({"role": "assistant", "content": response.content})
    #         tool_results = []
    #         for content in response.content:
    #             if content.type == 'tool_use':
    #                 tool_name = content.name
    #                 tool_args = content.input
    #                 try:
    #                     result = await self.client.call_tool(tool_name, tool_args)
    #                     if isinstance(result.content, list):
    #                         result_text = "\n".join([c.text if hasattr(c, 'text') else str(c) for c in result.content])
    #                     else:
    #                         result_text = result.content
    #                     tool_results.append({"type": "tool_result", "tool_use_id": content.id, "content": result_text})
    #                 except Exception as e:
    #                     print(f"Error calling tool {tool_name}: {e}")
    #                     tool_results.append({"type": "tool_result", "tool_use_id": content.id, "content": f"Error: {str(e)}", "is_error": True})
    #         messages.append({"role": "user", "content": tool_results})
    #         response = self.anthropic.messages.create(
    #             model=MODEL_ID, max_tokens=4096, messages=messages, tools=available_tools
    #         )
    #     final_text = [content.text for content in response.content if hasattr(content, 'text')]
    #     return "\n".join(final_text)

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI with access to MCP server tools.

        Implements an agentic loop where OpenAI can use MCP tools to answer
        the query. The loop continues until the model provides a final response
        without requesting further tool use.

        Args:
            query: The user's query to process

        Returns:
            The final text response from OpenAI
        """
        # Initialize conversation with user query
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Fetch available tools from MCP server
        available_tools = await self._get_tools()

        # Initial OpenAI API call with tools
        response = self.openai.chat.completions.create(
            model=MODEL_ID,
            max_tokens=4096,
            messages=messages,
            tools=available_tools
        )

        # Agentic loop - continue while OpenAI requests tool use
        while response.choices[0].finish_reason == "tool_calls":
            assistant_message = response.choices[0].message

            # Add assistant's response (including tool call requests) to conversation
            messages.append(assistant_message)

            # Execute all requested tool calls
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                try:
                    # Call the tool via MCP session
                    result = await self.client.call_tool(tool_name, tool_args)

                    # Format result content
                    if isinstance(result.content, list):
                        result_text = "\n".join([
                            c.text if hasattr(c, 'text') else str(c)
                            for c in result.content
                        ])
                    else:
                        result_text = result.content

                    # OpenAI expects one message per tool result with role "tool"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })

                except Exception as e:
                    # Handle tool execution errors
                    print(f"Error calling tool {tool_name}: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {str(e)}"
                    })

            # Get next response from OpenAI
            response = self.openai.chat.completions.create(
                model=MODEL_ID,
                max_tokens=4096,
                messages=messages,
                tools=available_tools
            )

        # Extract final text response from OpenAI
        return response.choices[0].message.content or ""

    async def converse(self):
        """Start an interactive conversation mode with OpenAI.

        Allows the user to have a multi-turn conversation with OpenAI,
        where each query can trigger tool use. Exits when user types 'quit' or 'q'.
        """
        print("\nEntering conversation mode. Type 'quit' or 'q' to exit.")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() in ("quit", "q"):
                break  # Signal exit

            if not query:
                print("Please enter a query")
                continue

            try:
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"Error processing query: {e}")
        return

    async def prompt(self, prompt_name: str):
        """Execute a named prompt template from the MCP server.

        Retrieves a prompt template from the server, collects required arguments
        from the user, generates the prompt, and processes it with OpenAI.

        Flow:
            1. Fetch all available prompts from the server
            2. Find the requested prompt by name
            3. Collect any required arguments from the user (e.g. file_path for code_review)
            4. Call server to build the filled-in prompt text
            5. Pass the prompt to process_query() → OpenAI runs it using MCP tools if needed

        Args:
            prompt_name: Name of the prompt template to execute
        """
        try:
            # Fetch available prompts from server
            prompts_response = await self._get_prompts()
            prompt_obj = next(
                (p for p in prompts_response if p.name == prompt_name),
                None
            ) # next method has similar behaviour to js prompts_response.find

            if not prompt_obj:
                print(f"Prompt '{prompt_name}' not found")
                return

            print(prompt_obj)

            # Collect arguments for the prompt template
            arguments = {}
            if prompt_obj.arguments:
                for arg in prompt_obj.arguments:
                    required = "required" if arg.required else "optional"
                    user_input = input(f"{arg.name} ({required}): ").strip()

                    # Validate required arguments
                    if not user_input and arg.required:
                        print(f"Error: {arg.name} is required")
                        return

                    if user_input:
                        arguments[arg.name] = user_input

            # Generate the prompt with provided arguments
            prompt_result = await self.client.get_prompt(prompt_name, arguments=arguments)

            prompt = prompt_result.messages[0].content.text

            # Process the generated prompt with OpenAI
            response = await self.process_query(prompt)
            print(response)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}\n")
            return


    async def read_file(self):
        """Read the contents of a file via MCP resource.

        Prompts the user for a file path and retrieves the file content
        through the MCP server's file resource.
        """
        try:
            file_name = input("Enter file path: ").strip()
            # quote(): URL-encodes the file name so it's safe to embed in a URI.
            # e.g; if file_name is my file.txt, it would be converted to my%20file.txt
            # safe="" means even / gets encoded -- so a path like folder/file.txt becomes folder%2Ffile.txt
            encoded_file_name = quote(file_name, safe="")
            # Access file resource using file:/// URI scheme
            resource = await self.client.read_resource(f"file:///{encoded_file_name}")
            file_content = json.loads(resource[0].text)["file_content"]

            print(f"File Content:\n {file_content}")
            return file_content
        except Exception as e:
            print(f"Error reading file: {e}")

    def _print_dir_listing(self, items: list[dict]):
        """Format and print a directory listing.

        Args:
            items: List of directory items with metadata (type, size, modified, name)
        """
        print("\nDirectory Listing:\n")
        print(f"{'Type':<10} {'Size':>10} {'Modified':<25} {'Name'}")
        print("-" * 70)
        for item in items:
            # Add icon based on item type
            type_icon = "📁" if item["type"] == "directory" else "📄"
            size = f"{item['size']} B"
            print(f"{type_icon:<2} {item['type']:<8} {size:>10}  {item['modified']:<25} {item['name']}")


    async def read_dir(self):
        """List the contents of the current directory via MCP resource.

        Retrieves and displays directory contents through the MCP server's
        directory resource.
        """
        try:
            # Access directory resource using dir:// URI scheme
            resource = await self.client.read_resource(f"dir://.")
            dir_list = json.loads(resource[0].text)["items"]
            self._print_dir_listing(dir_list)
            return
        except Exception as e:
            print(f"Error reading directory: {e}")

    async def menu(self):
        """Run the main interactive chat loop with menu-driven interface.

        Presents a menu of options including prompt execution, file operations,
        and conversation mode. Continues until user selects quit.
        """
        print("\nMCP Client Started!")
        print("Select from the menu or 'quit'/'q' to exit.")

        # Map menu choices to async functions
        menu_actions = {
            "1": lambda: self.prompt("documentation_generator"),
            "2": lambda: self.prompt("code_review"),
            "3": self.read_file,
            "4": self.read_dir,
            "5": self.converse,
            "q": self.quit_action,
            # "quit": self.quit_action
        }

        while True:
            choice = input("""
            Select from the Menu
            1. Generate Documentation
            2. Review Code
            3. Read File
            4. Read Current Directory
            5. Converse with Agent
            q. Quit
            > """).strip()

            action = menu_actions.get(choice)

            if not action:
                print("Invalid choice. Please try again.")
                continue

            result = await action()
            if result == "quit":
                break

    async def quit_action(self):
        """Signal to exit the client.

        Returns:
            String "quit" to signal exit from chat loop
        """
        print("Exiting client...")
        return "quit"

    async def cleanup(self):
        """Clean up resources and close connections.

        Closes the async exit stack which manages all open connections
        and resources.
        """
        if self.exit_stack:
            await self.exit_stack.aclose()

async def main():
    # Check correct usage
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_path>")
        sys.exit(1)

    client = MCPClient()
    try:
        server_path = sys.argv[1]
        print(f"Connecting to server: {server_path}")

        await client.connect_to_server(server_path)

        await client.menu()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    finally:
        # Always cleanup at the end
        await client.cleanup()

# Runs main() only when executed directly (python client.py).
# If this file is imported by another module, this block is skipped —
# only the class and functions (MCPClient, main, etc.) are exported.
if __name__ == "__main__":
    asyncio.run(main())

# check run.ipynb for step by step options calling.