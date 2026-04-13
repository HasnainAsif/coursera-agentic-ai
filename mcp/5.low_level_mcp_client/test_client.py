import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    print(f"Python: {sys.executable}")
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=None
    )
    
    print("Creating transport...")
    async with stdio_client(server_params) as (read, write):
        print("Transport created!")
        async with ClientSession(read, write) as session:
            print("Session created!")
            await session.initialize()
            print("✓ Connected!")
            
            tools = await session.list_tools()
            print(f"Tools: {[t.name for t in tools.tools]}")

asyncio.run(main())