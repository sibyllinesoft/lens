#!/usr/bin/env python3
"""
End-to-End Tests for Lens MCP (Model Context Protocol) Server

This test suite validates the complete MCP server functionality including:
- JSON-RPC protocol compliance
- Tool discovery and execution
- File indexing and search operations
- Error handling and timeout management

Tests the Rust implementation of the MCP server running in production mode.
"""

import subprocess
import json
import time
import select
import os
import pytest
from datetime import datetime
from pathlib import Path


class MCPTestClient:
    """Test client for communicating with the Lens MCP server via STDIO."""
    
    def __init__(self):
        self.process = None
        self.ready = False
        
    def start_server(self):
        """Start the MCP server process."""
        # Get the project root directory (3 levels up from tests/e2e/)
        project_root = Path(__file__).parent.parent.parent
        binary_path = project_root / "rust-core" / "target" / "release" / "lens-core"
        
        if not binary_path.exists():
            raise FileNotFoundError(
                f"MCP server binary not found at {binary_path}. "
                "Run: cd rust-core && cargo build --release --features mcp"
            )
        
        self.process = subprocess.Popen(
            [str(binary_path), '--mode', 'real', '--mcp'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            cwd=str(project_root)  # Set working directory to project root
        )
        
        # Wait for startup
        time.sleep(2)
        self.ready = True
    
    def send_request(self, request, timeout=30):
        """Send a JSON-RPC request and wait for response."""
        if not self.ready or not self.process:
            raise RuntimeError("MCP server not ready")
            
        # Send request
        request_str = json.dumps(request) + '\n'
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Read response with timeout
        return self._read_json_response(timeout)
    
    def _read_json_response(self, timeout=30):
        """Read lines until we get a valid JSON response."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"MCP server process died. Stderr: {stderr_output}")
            
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
                
        raise TimeoutError(f"Timeout waiting for MCP server response after {timeout}s")
    
    def shutdown(self):
        """Shutdown the MCP server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.ready = False


@pytest.fixture
def mcp_client():
    """Pytest fixture providing an MCP client with server lifecycle management."""
    client = MCPTestClient()
    client.start_server()
    
    yield client
    
    client.shutdown()


class TestMCPServer:
    """Test suite for MCP server functionality."""
    
    def test_server_initialization(self, mcp_client):
        """Test MCP server initialization and capability discovery."""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        response = mcp_client.send_request(init_request)
        
        # Validate response structure
        assert "jsonrpc" in response
        assert response["jsonrpc"] == "2.0"
        assert "id" in response
        assert response["id"] == 1
        assert "result" in response
        
        result = response["result"]
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "lens-mcp-server"
        assert "capabilities" in result
        assert "tools" in result["capabilities"]
    
    def test_tools_discovery(self, mcp_client):
        """Test MCP tools discovery and metadata."""
        # Initialize first
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        mcp_client.send_request(init_request)
        
        # List tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = mcp_client.send_request(tools_request)
        
        assert "result" in response
        assert "tools" in response["result"]
        
        tools = response["result"]["tools"]
        assert len(tools) == 3
        
        tool_names = {tool["name"] for tool in tools}
        expected_tools = {"lens_search", "lens_index", "lens_status"}
        assert tool_names == expected_tools
        
        # Validate tool metadata
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
    
    def test_file_indexing_and_search_workflow(self, mcp_client):
        """Test the complete workflow: indexing files then searching them."""
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        mcp_client.send_request(init_request)
        
        # Read the MCP source file for indexing
        project_root = Path(__file__).parent.parent.parent
        mcp_source_path = project_root / "rust-core" / "src" / "mcp.rs"
        
        if not mcp_source_path.exists():
            pytest.skip(f"MCP source file not found at {mcp_source_path}")
        
        with open(mcp_source_path, 'r') as f:
            mcp_content = f.read()
        
        # Index the MCP source file
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
        
        response = mcp_client.send_request(index_request)
        
        assert "result" in response
        assert "content" in response["result"]
        content = response["result"]["content"][0]
        assert "Successfully indexed file" in content["text"]
        
        # Search for MCP-related content
        search_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "lens_search",
                "arguments": {
                    "query": "MCP",
                    "limit": 3
                }
            }
        }
        
        response = mcp_client.send_request(search_request)
        
        assert "result" in response
        assert "content" in response["result"]
        result_text = response["result"]["content"][0]["text"]
        
        # Validate search results
        assert "Found" in result_text
        assert "rust-core/src/mcp.rs" in result_text
        assert "Model Context Protocol" in result_text
    
    def test_search_engine_references(self, mcp_client):
        """Test searching for SearchEngine references after indexing."""
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        mcp_client.send_request(init_request)
        
        # Index MCP source
        project_root = Path(__file__).parent.parent.parent
        mcp_source_path = project_root / "rust-core" / "src" / "mcp.rs"
        
        with open(mcp_source_path, 'r') as f:
            mcp_content = f.read()
        
        index_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "lens_index",
                "arguments": {
                    "file_path": "rust-core/src/mcp.rs",
                    "content": mcp_content
                }
            }
        }
        mcp_client.send_request(index_request)
        
        # Search for SearchEngine
        search_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "lens_search",
                "arguments": {
                    "query": "SearchEngine",
                    "limit": 5
                }
            }
        }
        
        response = mcp_client.send_request(search_request)
        
        assert "result" in response
        result_text = response["result"]["content"][0]["text"]
        
        # Should find SearchEngine references
        assert "Found" in result_text
        assert "SearchEngine" in result_text
        assert "rust-core/src/mcp.rs" in result_text
    
    def test_server_status(self, mcp_client):
        """Test server status reporting."""
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        mcp_client.send_request(init_request)
        
        # Get status
        status_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "lens_status",
                "arguments": {}
            }
        }
        
        response = mcp_client.send_request(status_request)
        
        assert "result" in response
        result_text = response["result"]["content"][0]["text"]
        
        # Validate status information
        assert "Lens Search Engine Status" in result_text
        assert "Version:" in result_text
        assert "Mode: Production (MCP)" in result_text
        assert "MCP Server Info" in result_text
        assert "Protocol Version: 0.1.0" in result_text
    
    def test_search_without_indexing(self, mcp_client):
        """Test search behavior when no content is indexed."""
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        mcp_client.send_request(init_request)
        
        # Search without indexing anything
        search_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "lens_search",
                "arguments": {
                    "query": "nonexistent",
                    "limit": 5
                }
            }
        }
        
        response = mcp_client.send_request(search_request)
        
        assert "result" in response
        result_text = response["result"]["content"][0]["text"]
        
        # Should indicate no results
        assert "No results found" in result_text
    
    def test_invalid_tool_request(self, mcp_client):
        """Test error handling for invalid tool requests."""
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        mcp_client.send_request(init_request)
        
        # Request invalid tool
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            }
        }
        
        response = mcp_client.send_request(invalid_request)
        
        # Should return an error
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found


class TestMCPIntegration:
    """Integration tests for MCP server with larger workflows."""
    
    def test_multiple_file_indexing_and_cross_search(self, mcp_client):
        """Test indexing multiple files and searching across them."""
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        mcp_client.send_request(init_request)
        
        project_root = Path(__file__).parent.parent.parent
        
        # Index MCP source
        mcp_path = project_root / "rust-core" / "src" / "mcp.rs"
        if mcp_path.exists():
            with open(mcp_path, 'r') as f:
                mcp_content = f.read()
            
            index_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "lens_index",
                    "arguments": {
                        "file_path": "rust-core/src/mcp.rs",
                        "content": mcp_content
                    }
                }
            }
            mcp_client.send_request(index_request, timeout=60)
        
        # Index search engine source
        search_path = project_root / "rust-core" / "src" / "search.rs"
        if search_path.exists():
            with open(search_path, 'r') as f:
                search_content = f.read()
            
            index_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "lens_index",
                    "arguments": {
                        "file_path": "rust-core/src/search.rs",
                        "content": search_content
                    }
                }
            }
            mcp_client.send_request(index_request, timeout=60)
        
        # Search across both files
        search_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "lens_search",
                "arguments": {
                    "query": "SearchEngine",
                    "limit": 10
                }
            }
        }
        
        response = mcp_client.send_request(search_request)
        result_text = response["result"]["content"][0]["text"]
        
        # Should find references in both files
        assert "Found" in result_text
        assert "SearchEngine" in result_text


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])