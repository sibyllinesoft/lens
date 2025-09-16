//! MCP (Model Context Protocol) server implementation
//! 
//! Provides a JSON-RPC server over stdio for code search capabilities
//! Compatible with MCP v2024-11-05 specification

use std::sync::Arc;
use anyhow::Result;
use jsonrpc_core::{IoHandler, Params, Value, MetaIoHandler};
use jsonrpc_stdio_server::ServerBuilder;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::search::{SearchEngine, SearchRequest, SearchResultType};

/// MCP tool specification for code search
#[derive(Debug, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// MCP search parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchParams {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub file_pattern: Option<String>,
}

fn default_limit() -> usize {
    20
}

/// MCP search result
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub score: f32,
    pub line_number: usize,
    pub code_snippet: String,
    pub match_reason: String,
}

/// Create and start the MCP server
pub async fn create_mcp_server(search_engine: Arc<SearchEngine>) -> Result<()> {
    let mut io = IoHandler::new();
    
    // Clone search engine for handlers
    let search_engine_init = search_engine.clone();
    let search_engine_tools = search_engine.clone();
    let search_engine_call = search_engine.clone();

    // Initialize handler - returns server capabilities
    io.add_method("initialize", move |params: Params| {
        let _search_engine = search_engine_init.clone();
        async move {
            tracing::info!("MCP server initializing");
            Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "lens-search",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }))
        }
    });

    // Tools list handler - returns available tools
    io.add_method("tools/list", move |_params: Params| {
        let _search_engine = search_engine_tools.clone();
        async move {
            Ok(json!({
                "tools": [{
                    "name": "search_code",
                    "description": "Search for code patterns, functions, or content in the indexed codebase",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (can be code patterns, function names, or content)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 20
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "Optional file pattern filter (e.g., '*.py', '*.ts')"
                            }
                        },
                        "required": ["query"]
                    }
                }]
            }))
        }
    });

    // Tool call handler - executes search
    io.add_method("tools/call", move |params: Params| {
        let search_engine = search_engine_call.clone();
        async move {
            // Parse parameters
            let tool_params: Value = params.parse()?;
            let tool_name = tool_params["name"].as_str()
                .ok_or_else(|| jsonrpc_core::Error::invalid_params("Missing tool name"))?;
            
            if tool_name != "search_code" {
                return Err(jsonrpc_core::Error::method_not_found());
            }

            let arguments = &tool_params["arguments"];
            let search_params: SearchParams = serde_json::from_value(arguments.clone())
                .map_err(|e| jsonrpc_core::Error::invalid_params(format!("Invalid parameters: {}", e)))?;

            // Perform search using the search engine's API
            let search_request = SearchRequest {
                query: search_params.query.clone(),
                file_path: search_params.file_pattern.clone(),
                language: None,
                max_results: search_params.limit,
                include_context: true,
                timeout_ms: 5000,
                enable_lsp: true,
                search_types: vec![SearchResultType::TextMatch, SearchResultType::Definition],
                search_method: None,
            };

            match search_engine.search_comprehensive(search_request).await {
                Ok(search_response) => {
                    let formatted_results: Vec<SearchResult> = search_response.results.into_iter()
                        .map(|result| SearchResult {
                            file_path: result.file_path,
                            score: result.score as f32,
                            line_number: result.line_number as usize,
                            code_snippet: result.content,
                            match_reason: format!("{:?}", result.result_type),
                        })
                        .collect();

                    // Format as markdown for better readability
                    let mut markdown_content = format!("# Code Search Results\n\n");
                    markdown_content.push_str(&format!("**Query:** `{}`\n", search_params.query));
                    markdown_content.push_str(&format!("**Found {} results**\n\n", formatted_results.len()));

                    for (i, result) in formatted_results.iter().enumerate() {
                        markdown_content.push_str(&format!("## Result {} (Score: {:.3})\n\n", i + 1, result.score));
                        markdown_content.push_str(&format!("**File:** `{}`\n", result.file_path));
                        markdown_content.push_str(&format!("**Line:** {}\n", result.line_number));
                        markdown_content.push_str(&format!("**Reason:** {}\n\n", result.match_reason));
                        markdown_content.push_str("```\n");
                        markdown_content.push_str(&result.code_snippet);
                        markdown_content.push_str("\n```\n\n");
                        markdown_content.push_str("---\n\n");
                    }

                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": markdown_content
                        }]
                    }))
                }
                Err(e) => {
                    tracing::error!("Search failed: {}", e);
                    Err(jsonrpc_core::Error::internal_error())
                }
            }
        }
    });

    // Resources list handler (optional)
    io.add_method("resources/list", |_params: Params| async move {
        Ok(json!({
            "resources": []
        }))
    });

    // Build and start the server
    tracing::info!("Starting MCP server on stdio");
    let server = ServerBuilder::new(io)
        .build();

    // Run the server
    server.await;
    
    Ok(())
}

/// Simple glob matching for file patterns
fn glob_match(pattern: &str, file_path: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    
    if pattern.starts_with("*.") {
        let extension = &pattern[2..];
        return file_path.ends_with(extension);
    }
    
    file_path.contains(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*.py", "test.py"));
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.py", "test.rs"));
        assert!(glob_match("*", "any_file.txt"));
        assert!(glob_match("test", "test_file.py"));
    }

    #[test]
    fn test_search_params_deserialization() {
        let json = json!({
            "query": "test function",
            "limit": 10,
            "file_pattern": "*.py"
        });
        
        let params: SearchParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.query, "test function");
        assert_eq!(params.limit, 10);
        assert_eq!(params.file_pattern, Some("*.py".to_string()));
    }

    #[test]
    fn test_search_params_defaults() {
        let json = json!({
            "query": "test function"
        });
        
        let params: SearchParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.query, "test function");
        assert_eq!(params.limit, 20); // Default value
        assert!(params.file_pattern.is_none());
    }
}