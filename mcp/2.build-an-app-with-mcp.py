# Building an MCP-Powered AI Agent
#
# Learn how to connect an AI agent to external tools via the Model Context Protocol (MCP).
# This script wires up a ReAct agent (LangGraph) to two MCP servers — one for library docs
# (Context7) and one for museum data (Met Museum) — and runs an interactive chat loop with
# persistent conversation memory.


import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient # Connects to MCP servers
from langgraph.prebuilt import create_react_agent # Creates ReAct-style agents
from langgraph.checkpoint.memory import InMemorySaver # Provides conversation memory
from langchain_openai import ChatOpenAI # OpenAI chat model integration

async def main():
    """
    Main function that sets up and runs an AI agent with access to multiple MCP servers.
    The agent can access Context7 library documentation and Met Museum collections.
    """

    # Configure MCP (Model Context Protocol) servers
    # These servers provide tools that the AI agent can use
    # MultiServerMCPClient can connect to, and update, multiple MCP servers at the same time while handling low-level transport details
    client = MultiServerMCPClient(
        {
            # Context7 server - provides access to library documentation
            "context7": {
                "url": "https://mcp.context7.com/mcp", # Server endpoint
                "transport": "streamable_http", # Communication protocol
            },
            # Met Museum server - provides access to museum collection data
            "met-museum": {
                "command": "npx", # Node.js package runner
                "args": ["-y", "metmuseum-mcp"], # Install and run met museum MCP
                "transport": "stdio", # Communication via stdin/stdout
            }
        }
    ) # The session and asynchronous context management is all handled internally in MultiServerMCPClient - Check end of file for better understanding of this statement

    # Initialize the OpenAI language model
    openai_model = ChatOpenAI(
        model="gpt-5-nano", 
    )

    # Retrieve all available tools from the configured MCP servers
    tools = await client.get_tools()

    # Set up conversation memory using InMemorySaver to remember previous messages in the conversation
    checkpointer = InMemorySaver()

	# Configuration for conversation persistence
    # The thread_id ensures all messages in this session are grouped together
    config = {"configurable": {"thread_id": "conversation_id"}}

    # Create the ReAct agent with all components
    agent = create_react_agent(
        model=openai_model,
        tools=tools,
        checkpointer=checkpointer # Memory system for conversation history
    )

    # Send initial message to introduce the agent and its capabilities
    response = await agent.ainvoke(
        {"messages": [
            {"role": "system", "content": "You are a smart, useful agent with tools to access code library documentation and the Met Museum collection."},
            {"role": "user", "content": "Give a brief introduction of what you do and the tools you can access."},
        ]},
        config=config # Use the conversation thread for memory persistence
    )

    print(response['messages'][-1].content) # Print the agent's response (last message in the conversation)

	# Command Line interaction loop - allows continuous conversation with the agent
    while True:
        # Display menu options to the user
        choice = input("""
        Menu:
        1. Ask the agent a question
        2. Quit
        Enter your choice (1 or 2): """)

        if choice == "1":
            print("Your question")
            query = input("> ")

            # Invoke Agent
            response = await agent.ainvoke(
                {"messages": query}, # User's current question
                config=config # Maintains conversation thread
            )
            # Display the agent's final response
            print(response['messages'][-1].content)
        else:
            # Exit the program for any choice other than "1"
            print("Goodbye!")
            break

# Entry point - runs the main function when script is executed directly
if __name__ == "__main__":
    # Use asyncio to run the async main function
    asyncio.run(main())


################################################################################################################
## ANOTHER WAY TO instantiate and invoke MCP Client

# # Create server parameters for stdio connection
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent

# server_params = StdioServerParameters(
#     command="python",
#     # Make sure to update to the full absolute path to your math_server.py file
#     args=["/path/to/math_server.py"],
# )

# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         # Initialize the connection
#         await session.initialize()

#         # Get tools
#         tools = await load_mcp_tools(session)

#         # Create and run the agent
#         agent = create_react_agent("openai:gpt-4.1", tools)
#         agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

## This is a lower-level approach that uses modules:
    # - StdioServerParameters: Parameter schema for a stdio server; command and args required
    # - stdio_client: Takes the transport parameters and creates an asynchronous client connected to the server with read/write streams so the client can receive JSON messages over the specified transport: standard I/O
    # - ClientSession: Handles all requests and messages over the input/ouput streams, this includes initializing the handshake, sending requests to the server, receiving responses, managing session states, and cleanup

##### The streamable HTTP version of this looks something like the following:

# # Use server from examples/servers/streamable-http-stateless/

# from mcp import ClientSession
# from mcp.client.streamable_http import streamablehttp_client

# from langgraph.prebuilt import create_react_agent
# from langchain_mcp_adapters.tools import load_mcp_tools

# async with streamablehttp_client("http://localhost:3000/mcp") as (read, write, _):
#     async with ClientSession(read, write) as session:
#         # Initialize the connection
#         await session.initialize()

#         # Get tools
#         tools = await load_mcp_tools(session)
#         agent = create_react_agent("openai:gpt-4.1", tools)
#         math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

#### The session and asynchronous context management is all handled internally in MultiServerMCPClient