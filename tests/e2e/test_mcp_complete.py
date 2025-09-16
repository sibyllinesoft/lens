#!/usr/bin/env python3
"""
Complete MCP Demo - Index and Search
Shows the full MCP workflow: indexing files first, then searching for content.
"""
import subprocess
import json
import time
import threading
import select
import sys
from datetime import datetime

class MCPClient:
    def __init__(self):
        self.process = None
        self.ready = False
        
    def start_server(self):
        """Start the MCP server process."""
        print("🚀 Starting MCP server...")
        self.process = subprocess.Popen(
            ['./rust-core/target/release/lens-core', '--mode', 'real', '--mcp'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0  # Unbuffered
        )
        # Wait for startup
        time.sleep(2)
        self.ready = True
        print("✅ MCP server started")
    
    def send_request(self, request):
        """Send a JSON-RPC request and wait for response."""
        if not self.ready or not self.process:
            return {"error": "Server not ready"}
            
        # Send request
        request_str = json.dumps(request) + '\n'
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Read response with timeout
        return self._read_json_response(timeout=10)
    
    def _read_json_response(self, timeout=10):
        """Read lines until we get a valid JSON response."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if self.process.poll() is not None:
                return {"error": "Server process died"}
            
            # Try to read a line with timeout
            try:
                if select.select([self.process.stdout], [], [], 0.1)[0]:
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        # Look for JSON-RPC responses (they start with {"jsonrpc")
                        if line.startswith('{"jsonrpc"'):
                            try:
                                return json.loads(line)
                            except json.JSONDecodeError:
                                continue
                else:
                    time.sleep(0.1)
            except:
                time.sleep(0.1)
                
        return {"error": "Timeout waiting for response"}
    
    def shutdown(self):
        """Shutdown the server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("🛑 MCP server stopped")

def main():
    print("=== Complete MCP Demo Session ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    client = MCPClient()
    
    try:
        # Start server
        client.start_server()
        
        # Test 1: Initialize
        print("\n=== Initialize MCP Server ===")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "complete-demo", "version": "1.0.0"}
            }
        }
        
        response = client.send_request(init_request)
        if "error" in response:
            print(f"❌ {response['error']}")
            return
        else:
            print("✅ Initialize successful!")
            print(f"🔧 Server: {response.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")
            
        print("-" * 60)
        
        # Test 2: List Tools
        print("\n=== List Available Tools ===")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = client.send_request(tools_request)
        if "error" in response:
            print(f"❌ {response['error']}")
            return
        else:
            print("✅ Tools list retrieved!")
            tools = response.get('result', {}).get('tools', [])
            print(f"📋 Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
                
        print("-" * 60)
        
        # Test 3: Index the MCP module we just created
        print("\n=== Index MCP Source Code ===")
        
        # Read the MCP source file
        try:
            with open('rust-core/src/mcp.rs', 'r') as f:
                mcp_content = f.read()
        except FileNotFoundError:
            print("❌ MCP source file not found")
            return
            
        index_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "lens_index",
                "arguments": {
                    "file_path": "rust-core/src/mcp.rs",
                    "content": mcp_content
                }
            }
        }
        
        response = client.send_request(index_request)
        if "error" in response:
            print(f"❌ Index failed: {response['error']}")
            return
        else:
            print("✅ MCP file indexed successfully!")
            content = response.get('result', {}).get('content', [])
            if content:
                result_text = content[0].get('text', 'No response text')
                print(f"📄 Index result: {result_text}")
                
        print("-" * 60)
        
        # Test 4: Index another file for more content
        print("\n=== Index Search Engine Code ===")
        
        try:
            with open('rust-core/src/search.rs', 'r') as f:
                search_content = f.read()
        except FileNotFoundError:
            print("❌ Search source file not found")
            return
            
        index_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "lens_index",
                "arguments": {
                    "file_path": "rust-core/src/search.rs",
                    "content": search_content
                }
            }
        }
        
        response = client.send_request(index_request)
        if "error" in response:
            print(f"❌ Index failed: {response['error']}")
        else:
            print("✅ Search file indexed successfully!")
            content = response.get('result', {}).get('content', [])
            if content:
                result_text = content[0].get('text', 'No response text')
                print(f"📄 Index result: {result_text}")
                
        print("-" * 60)
        
        # Test 5: Now search for MCP-related content
        print("\n=== Search for MCP-related code ===")
        search_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "lens_search",
                "arguments": {
                    "query": "MCP",
                    "limit": 5
                }
            }
        }
        
        response = client.send_request(search_request)
        if "error" in response:
            print(f"❌ Search failed: {response['error']}")
        else:
            print("✅ Search completed!")
            content = response.get('result', {}).get('content', [])
            if content:
                result_text = content[0].get('text', 'No search results')
                print(f"🔍 Search results:\n{result_text}")
            else:
                print("🔍 No results found")
                
        print("-" * 60)
        
        # Test 6: Search for SearchEngine 
        print("\n=== Search for SearchEngine ===")
        search_request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "lens_search",
                "arguments": {
                    "query": "SearchEngine",
                    "limit": 3
                }
            }
        }
        
        response = client.send_request(search_request)
        if "error" in response:
            print(f"❌ Search failed: {response['error']}")
        else:
            print("✅ Search completed!")
            content = response.get('result', {}).get('content', [])
            if content:
                result_text = content[0].get('text', 'No search results')
                print(f"🔍 Search results:\n{result_text}")
            else:
                print("🔍 No results found")
                
        print("-" * 60)
        
        # Test 7: Get server status
        print("\n=== Get Server Status ===")
        status_request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "lens_status",
                "arguments": {}
            }
        }
        
        response = client.send_request(status_request)
        if "error" in response:
            print(f"❌ Status failed: {response['error']}")
        else:
            print("✅ Status retrieved!")
            content = response.get('result', {}).get('content', [])
            if content:
                result_text = content[0].get('text', 'No status info')
                print(f"📊 Server status:\n{result_text}")
                
        print("-" * 60)
        print("\n🎉 Complete MCP demo finished successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        client.shutdown()

if __name__ == "__main__":
    main()