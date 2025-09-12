//! LSP Integration Module
//! 
//! Provides real Language Server Protocol integration for:
//! - tsserver (TypeScript/JavaScript) 
//! - pylsp (Python)
//! - rust-analyzer (Rust)
//! - gopls (Go)
//!
//! Key features:
//! - Bounded BFS traversal (depth≤2, K≤64) for def/ref/type/impl
//! - 24h TTL hint caching with invalidation
//! - 40-60% routing by intent with safety floors
//! - Real language servers with process management

pub mod client;
pub mod hint;
pub mod manager;
pub mod router;
pub mod server_process;

use anyhow::Result;
use lsp_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;

pub use client::LspClient;
pub use hint::{HintCache, SymbolHint, HintType};
pub use manager::LspManager;
pub use router::LspRouter;
pub use server_process::LspServerProcess;

/// Supported LSP server types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LspServerType {
    TypeScript,  // tsserver
    Python,      // pylsp
    Rust,        // rust-analyzer
    Go,          // gopls
    JavaScript,  // tsserver
}

impl LspServerType {
    pub fn from_file_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "ts" | "tsx" => Some(Self::TypeScript),
            "js" | "jsx" => Some(Self::JavaScript),
            "py" | "pyi" => Some(Self::Python),
            "rs" => Some(Self::Rust),
            "go" => Some(Self::Go),
            _ => None,
        }
    }

    pub fn server_command(&self) -> (&'static str, Vec<&'static str>) {
        match self {
            Self::TypeScript | Self::JavaScript => ("typescript-language-server", vec!["--stdio"]),
            Self::Python => ("pylsp", vec!["--verbose"]),
            Self::Rust => ("rust-analyzer", vec![]),
            Self::Go => ("gopls", vec!["serve"]),
        }
    }

    pub fn file_extensions(&self) -> Vec<&'static str> {
        match self {
            Self::TypeScript => vec!["ts", "tsx"],
            Self::JavaScript => vec!["js", "jsx"],
            Self::Python => vec!["py", "pyi"],
            Self::Rust => vec!["rs"],
            Self::Go => vec!["go"],
        }
    }
}

/// LSP query intent classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Find definition of symbol
    Definition,
    /// Find references to symbol  
    References,
    /// Find type information
    TypeDefinition,
    /// Find implementations
    Implementation,
    /// Goto declaration
    Declaration,
    /// Symbol search
    Symbol,
    /// Completion/autocomplete
    Completion,
    /// Hover information
    Hover,
    /// General text search (fallback to basic search)
    TextSearch,
}

impl QueryIntent {
    /// Classify query intent from search string
    pub fn classify(query: &str) -> Self {
        let query_lower = query.to_lowercase();
        
        // Pattern matching for intent classification
        if query_lower.contains("def ") || query_lower.contains("function ") || query_lower.contains("class ") {
            Self::Definition
        } else if query_lower.contains("ref ") || query_lower.contains("usage") || query_lower.contains("usages") {
            Self::References
        } else if query_lower.contains("type ") || query_lower.contains("interface ") {
            Self::TypeDefinition
        } else if query_lower.contains("impl ") || query_lower.contains("implement") {
            Self::Implementation
        } else if query_lower.starts_with("@") {
            Self::Symbol
        } else if query_lower.ends_with("?") {
            Self::Hover
        } else {
            Self::TextSearch
        }
    }

    /// Check if this intent is LSP-eligible
    pub fn is_lsp_eligible(&self) -> bool {
        !matches!(self, Self::TextSearch)
    }
    
    /// Check if this intent requires safety floor (monotone results)
    /// 
    /// Exact and structural queries must never return fewer results than baseline
    /// This implements the TODO.md safety requirement for exact/struct queries
    pub fn requires_safety_floor(&self) -> bool {
        matches!(self, Self::Definition | Self::Symbol | Self::TypeDefinition | Self::Implementation)
    }
    
    /// Check if this is an exact match query that must be monotone
    pub fn is_exact_query(&self) -> bool {
        matches!(self, Self::Definition | Self::Symbol)
    }
    
    /// Check if this is a structural query that must be monotone
    pub fn is_structural_query(&self) -> bool {
        matches!(self, Self::TypeDefinition | Self::Implementation)
    }
}

/// BFS traversal bounds for LSP queries
#[derive(Debug, Clone)]
pub struct TraversalBounds {
    pub max_depth: u8,
    pub max_results: u16,
    pub timeout_ms: u64,
}

impl Default for TraversalBounds {
    fn default() -> Self {
        Self {
            max_depth: 2,    // depth ≤ 2 per TODO.md
            max_results: 64, // K ≤ 64 per TODO.md
            timeout_ms: 5000, // 5 second default timeout
        }
    }
}

/// LSP search result with provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspSearchResult {
    pub file_path: String,
    pub line_number: u32,
    pub column: u32,
    pub content: String,
    pub hint_type: HintType,
    pub server_type: LspServerType,
    pub confidence: f64,
    pub context_lines: Option<Vec<String>>,
}

/// LSP-augmented search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspSearchResponse {
    pub lsp_results: Vec<LspSearchResult>,
    pub fallback_results: Vec<crate::search::SearchResult>,
    pub total_time_ms: u64,
    pub lsp_time_ms: u64,
    pub cache_hit_rate: f64,
    pub server_types_used: Vec<LspServerType>,
    pub intent: QueryIntent,
}

impl Default for LspSearchResponse {
    fn default() -> Self {
        Self {
            lsp_results: Vec::new(),
            fallback_results: Vec::new(),
            total_time_ms: 0,
            lsp_time_ms: 0,
            cache_hit_rate: 0.0,
            server_types_used: Vec::new(),
            intent: QueryIntent::TextSearch,
        }
    }
}

/// Configuration for LSP integration
#[derive(Debug, Clone)]
pub struct LspConfig {
    pub enabled: bool,
    pub server_timeout_ms: u64,
    pub cache_ttl_hours: u64,
    pub max_concurrent_requests: usize,
    pub routing_percentage: f64, // Target 40-60% per TODO.md
    pub traversal_bounds: TraversalBounds,
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            server_timeout_ms: 5000,
            cache_ttl_hours: 24,
            max_concurrent_requests: 10,
            routing_percentage: 0.5, // 50% routing target
            traversal_bounds: TraversalBounds::default(),
        }
    }
}

/// Global LSP state manager
pub struct LspState {
    manager: RwLock<Option<LspManager>>,
    config: LspConfig,
}

impl LspState {
    pub fn new(config: LspConfig) -> Self {
        Self {
            manager: RwLock::new(None),
            config,
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        let mut manager = self.manager.write().await;
        *manager = Some(LspManager::new(self.config.clone()).await?);
        Ok(())
    }

    pub async fn search(&self, query: &str, file_path: Option<&str>) -> Result<LspSearchResponse> {
        let manager = self.manager.read().await;
        match manager.as_ref() {
            Some(mgr) => mgr.search(query, file_path).await,
            None => {
                tracing::warn!("LSP not initialized, returning empty response");
                Ok(LspSearchResponse {
                    lsp_results: vec![],
                    fallback_results: vec![],
                    total_time_ms: 0,
                    lsp_time_ms: 0,
                    cache_hit_rate: 0.0,
                    server_types_used: vec![],
                    intent: QueryIntent::TextSearch,
                })
            }
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        let mut manager = self.manager.write().await;
        if let Some(mut mgr) = manager.take() {
            mgr.shutdown().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_type_from_extension() {
        assert_eq!(LspServerType::from_file_extension("ts"), Some(LspServerType::TypeScript));
        assert_eq!(LspServerType::from_file_extension("py"), Some(LspServerType::Python));
        assert_eq!(LspServerType::from_file_extension("rs"), Some(LspServerType::Rust));
        assert_eq!(LspServerType::from_file_extension("go"), Some(LspServerType::Go));
        assert_eq!(LspServerType::from_file_extension("xyz"), None);
    }

    #[test]
    fn test_query_intent_classification() {
        assert_eq!(QueryIntent::classify("def myFunction"), QueryIntent::Definition);
        assert_eq!(QueryIntent::classify("ref someVariable"), QueryIntent::References);
        assert_eq!(QueryIntent::classify("type MyInterface"), QueryIntent::TypeDefinition);
        assert_eq!(QueryIntent::classify("impl MyTrait"), QueryIntent::Implementation);
        assert_eq!(QueryIntent::classify("@symbolName"), QueryIntent::Symbol);
        assert_eq!(QueryIntent::classify("what is this?"), QueryIntent::Hover);
        assert_eq!(QueryIntent::classify("random text"), QueryIntent::TextSearch);
    }

    #[test]
    fn test_lsp_eligible_intent() {
        assert!(QueryIntent::Definition.is_lsp_eligible());
        assert!(QueryIntent::References.is_lsp_eligible());
        assert!(QueryIntent::Symbol.is_lsp_eligible());
        assert!(!QueryIntent::TextSearch.is_lsp_eligible());
    }
}