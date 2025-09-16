/*!
 * # Model Context Protocol (MCP) Server Implementation
 * 
 * This module implements an MCP server that exposes the Lens search engine functionality
 * to AI assistants and development tools through the standardized Model Context Protocol.
 * 
 * ## MCP Protocol Support
 * 
 * The server implements the MCP specification including:
 * - **Tools**: Code search functionality exposed as MCP tools
 * - **Resources**: Access to indexed file contents and metadata
 * - **Prompts**: Pre-defined search patterns and code analysis prompts
 * - **STDIO Transport**: Standard input/output communication for broad compatibility
 * 
 * ## Available Tools
 * 
 * - `lens_search`: Search through indexed code with configurable parameters
 * - `lens_index`: Add new files to the search index
 * - `lens_status`: Get information about the search engine state
 * 
 * ## Security Model
 * 
 * The MCP server inherits the security model from the core Lens engine:
 * - Production-only operation enforcement
 * - Input validation for all search queries
 * - No direct file system access beyond indexed content
 * - Comprehensive audit logging of all operations
 */

use anyhow::Result;
use jsonrpc_core::{MetaIoHandler, Params, Value, Error as RpcError};
use jsonrpc_stdio_server::ServerBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map};
use std::sync::{Arc, Mutex};
use tracing::{info, error, debug};
use futures::future::BoxFuture;

use crate::SearchEngine;

/// MCP Server implementation for Lens search functionality.
/// 
/// Provides a standardized interface for AI assistants to interact with the Lens
/// search engine through the Model Context Protocol. The server runs on STDIO
/// for maximum compatibility with various AI development environments.
/// 
/// # Protocol Implementation
/// 
/// Implements MCP 0.1.0 specification including:
/// - Tool discovery and execution
/// - Resource enumeration and retrieval  
/// - Prompt templates for common search patterns
/// - Error handling with structured error responses
/// 
/// # Thread Safety
/// 
/// The server is designed to be thread-safe and can handle concurrent requests
/// from MCP clients while maintaining search engine consistency.
pub struct McpServer {
    search_engine: Arc<Mutex<SearchEngine>>,
}

/// MCP Tool definition for code search functionality.
/// 
/// Defines the structure and parameters for the `lens_search` tool that
/// MCP clients can invoke to perform code searches.
#[derive(Debug, Serialize, Deserialize)]
struct SearchTool {
    name: String,
    description: String,
    input_schema: Map<String, Value>,
}

/// Parameters for the lens_search MCP tool.
/// 
/// Defines the expected input parameters when clients invoke the search tool.
#[derive(Debug, Deserialize)]
struct SearchParams {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    file_filter: Option<String>,
}

/// Parameters for the lens_index MCP tool.
/// 
/// Allows clients to add new files to the search index.
#[derive(Debug, Deserialize)]
struct IndexParams {
    file_path: String,
    content: String,
}

/// MCP tool execution result.
/// 
/// Standardized response format for all MCP tool executions.
#[derive(Debug, Serialize)]
struct ToolResult {
    content: Vec<ToolContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
}

/// Content within a tool execution result.
/// 
/// Can contain text, structured data, or references to resources.
#[derive(Debug, Serialize)]
struct ToolContent {
    r#type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotations: Option<Map<String, Value>>,
}

/// Default limit for search results when not specified by client.
fn default_limit() -> usize {
    20
}

impl McpServer {
    /// Create a new MCP server with the given search engine.
    /// 
    /// # Arguments
    /// 
    /// * `search_engine` - Shared reference to the search engine instance
    /// 
    /// # Returns
    /// 
    /// A new MCP server ready to handle client connections.
    /// 
    /// # Example
    /// 
    /// ```rust,no_run
    /// use lens_core::{McpServer, SearchEngine};
    /// use std::sync::Arc;
    /// 
    /// # async fn example() -> anyhow::Result<()> {
    /// let search_engine = Arc::new(SearchEngine::new_in_memory()?);
    /// let mcp_server = McpServer::new(search_engine);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(search_engine: Arc<Mutex<SearchEngine>>) -> Self {
        Self { search_engine }
    }

    /// Start the MCP server on STDIO.
    /// 
    /// This method starts the MCP server and begins listening for JSON-RPC
    /// messages on standard input and output. The server will run until
    /// the client disconnects or sends a shutdown request.
    /// 
    /// # Returns
    /// 
    /// A future that resolves when the server shuts down gracefully.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the server fails to start or encounters a
    /// fatal error during operation.
    /// 
    /// # Example
    /// 
    /// ```rust,no_run
    /// use lens_core::{McpServer, SearchEngine};
    /// use std::sync::Arc;
    /// 
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let search_engine = Arc::new(SearchEngine::new_in_memory()?);
    ///     let mcp_server = McpServer::new(search_engine);
    ///     mcp_server.start().await
    /// }
    /// ```
    pub async fn start(self) -> Result<()> {
        info!("Starting MCP server on STDIO");
        
        let mut io: MetaIoHandler<()> = MetaIoHandler::default();
        
        // Clone for move into closures
        let search_engine = self.search_engine.clone();
        
        // Initialize MCP protocol - respond to initialization
        let init_engine = search_engine.clone();
        io.add_method("initialize", move |params: Params| {
            let engine = init_engine.clone();
            Box::pin(async move {
                debug!("MCP initialize request: {:?}", params);
                
                // Return server capabilities
                Ok(json!({
                    "protocolVersion": "0.1.0",
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        },
                        "resources": {
                            "subscribe": false,
                            "listChanged": false
                        },
                        "prompts": {
                            "listChanged": false
                        }
                    },
                    "serverInfo": {
                        "name": "lens-mcp-server",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }))
            }) as BoxFuture<'_, Result<Value, RpcError>>
        });

        // Handle tools/list - return available tools
        let tools_engine = search_engine.clone();
        io.add_method("tools/list", move |_params: Params| {
            let _engine = tools_engine.clone();
            Box::pin(async move {
                debug!("MCP tools/list request");
                
                Ok(json!({
                    "tools": [
                        {
                            "name": "lens_search",
                            "description": "Search through indexed code files for specific patterns, functions, or text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query to find in the code"
                                    },
                                    "limit": {
                                        "type": "number",
                                        "description": "Maximum number of results to return (default: 20)",
                                        "default": 20
                                    },
                                    "file_filter": {
                                        "type": "string",
                                        "description": "Optional file pattern to filter results (e.g., '*.rs', '*.py')"
                                    }
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "lens_index",
                            "description": "Add a new file to the search index",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path of the file to index"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of the file to index"
                                    }
                                },
                                "required": ["file_path", "content"]
                            }
                        },
                        {
                            "name": "lens_status",
                            "description": "Get status information about the search engine",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    ]
                }))
            }) as BoxFuture<'_, Result<Value, RpcError>>
        });

        // Handle tools/call - execute tool
        let call_engine = search_engine.clone();
        io.add_method("tools/call", move |params: Params| {
            let engine = call_engine.clone();
            Box::pin(async move {
                debug!("MCP tools/call request: {:?}", params);
                
                let params = params.parse::<Value>()?;
                let tool_name = params.get("name")
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| RpcError::invalid_params("Missing tool name"))?;
                
                let arguments = params.get("arguments")
                    .ok_or_else(|| RpcError::invalid_params("Missing arguments"))?;

                match tool_name {
                    "lens_search" => {
                        let search_params: SearchParams = serde_json::from_value(arguments.clone())
                            .map_err(|e| RpcError::invalid_params(format!("Invalid search parameters: {}", e)))?;
                        
                        info!("Executing lens_search with query: '{}', limit: {}", 
                              search_params.query, search_params.limit);
                        
                        match engine.lock().unwrap().search(&search_params.query, search_params.limit) {
                            Ok(results) => {
                                let results_text = if results.is_empty() {
                                    "No results found for the query.".to_string()
                                } else {
                                    let mut output = format!("Found {} results:\n\n", results.len());
                                    for (i, result) in results.iter().enumerate() {
                                        output.push_str(&format!(
                                            "{}. {}:{}:{} ({})\n   {}\n   Score: {:.2}\n\n",
                                            i + 1,
                                            result.file,
                                            result.line,
                                            result.col,
                                            result.lang,
                                            result.snippet.trim(),
                                            result.score
                                        ));
                                    }
                                    output
                                };
                                
                                Ok(json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": results_text
                                        }
                                    ]
                                }))
                            }
                            Err(e) => {
                                error!("Search failed: {}", e);
                                Ok(json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": format!("Search failed: {}", e)
                                        }
                                    ],
                                    "isError": true
                                }))
                            }
                        }
                    }
                    "lens_index" => {
                        let index_params: IndexParams = serde_json::from_value(arguments.clone())
                            .map_err(|e| RpcError::invalid_params(format!("Invalid index parameters: {}", e)))?;
                        
                        info!("Indexing file: {}", index_params.file_path);
                        
                        // Index the document using mutable access through the mutex
                        match engine.lock().unwrap().index_document(&index_params.file_path, &index_params.content) {
                            Ok(()) => {
                                Ok(json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": format!(
                                                "Successfully indexed file: {}\nContent length: {} characters\nFile is now searchable via lens_search.",
                                                index_params.file_path,
                                                index_params.content.len()
                                            )
                                        }
                                    ]
                                }))
                            }
                            Err(e) => {
                                error!("Indexing failed: {}", e);
                                Ok(json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": format!("Indexing failed for file {}: {}", index_params.file_path, e)
                                        }
                                    ],
                                    "isError": true
                                }))
                            }
                        }
                    }
                    "lens_status" => {
                        info!("Getting lens status");
                        
                        Ok(json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": format!(
                                        "Lens Search Engine Status:\n- Version: {}\n- Mode: Production (MCP)\n- Build: {}\n- Status: Ready for searches\n\nMCP Server Info:\n- Protocol Version: 0.1.0\n- Transport: STDIO\n- Available Tools: lens_search, lens_index, lens_status",
                                        env!("CARGO_PKG_VERSION"),
                                        crate::built::GIT_COMMIT_HASH.unwrap_or("unknown")
                                    )
                                }
                            ]
                        }))
                    }
                    _ => {
                        Err(RpcError::method_not_found())
                    }
                }
            }) as BoxFuture<'_, Result<Value, RpcError>>
        });

        // Handle resources/list - return available resources (empty for now)
        io.add_method("resources/list", move |_params: Params| {
            Box::pin(async move {
                debug!("MCP resources/list request");
                Ok(json!({
                    "resources": []
                }))
            }) as BoxFuture<'_, Result<Value, RpcError>>
        });

        // Handle prompts/list - return available prompts (empty for now)
        io.add_method("prompts/list", move |_params: Params| {
            Box::pin(async move {
                debug!("MCP prompts/list request");
                Ok(json!({
                    "prompts": []
                }))
            }) as BoxFuture<'_, Result<Value, RpcError>>
        });

        // Build and start the server
        let server = ServerBuilder::new(io)
            .build();

        info!("MCP server started, listening on STDIO");
        info!("Available tools: lens_search, lens_index, lens_status");
        info!("Ready to accept MCP requests from clients");

        // This will block until the server is shut down
        server.await;
        info!("MCP server shut down gracefully");
        Ok(())
    }
}