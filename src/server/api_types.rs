//! Rust API type definitions
//! 
//! Complete Rust equivalents of the TypeScript API types,
//! providing JSON serialization compatibility and validation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Search mode enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Lex,
    Struct,
    Hybrid,
}

/// Supported programming language enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SupportedLanguage {
    TypeScript,
    Python,
    Rust,
    Bash,
    Go,
    Java,
}

/// Symbol kind enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SymbolKind {
    Function,
    Class,
    Variable,
    Type,
    Interface,
    Constant,
    Enum,
    Method,
    Property,
}

/// Pattern type enumeration for structural searches
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PatternType {
    FunctionDef,
    ClassDef,
    Import,
    AsyncDef,
    Decorator,
    TryExcept,
    ForLoop,
    IfStatement,
}

/// Search request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Repository SHA identifier
    pub repo_sha: String,
    /// Search query
    pub q: String,
    /// Search mode
    pub mode: SearchMode,
    /// Fuzzy search edit distance (0-2)
    pub fuzzy: u32,
    /// Number of results to return (1-200)
    pub k: u32,
    /// Optional timeout in milliseconds
    pub timeout_ms: Option<u32>,
}

/// Structural search request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructRequest {
    /// Repository SHA identifier
    pub repo_sha: String,
    /// Search pattern
    pub pattern: String,
    /// Target language
    pub lang: SupportedLanguage,
    /// Maximum number of results
    pub max_results: Option<u32>,
}

/// Symbols near request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolsNearRequest {
    /// File path
    pub file: String,
    /// Line number
    pub line: u32,
    /// Search radius in lines
    pub radius: Option<u32>,
}

/// Individual search hit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// File path
    pub file: String,
    /// Line number (1-based)
    pub line: u32,
    /// Column number (0-based)
    pub col: u32,
    /// Language identifier
    pub lang: Option<String>,
    /// Code snippet
    pub snippet: Option<String>,
    /// Relevance score (0.0-1.0)
    pub score: f64,
    /// Match reasons
    pub why: Vec<String>,
    
    // Optional metadata
    /// AST path
    pub ast_path: Option<String>,
    /// Symbol kind
    pub symbol_kind: Option<SymbolKind>,
    /// Byte offset in file
    pub byte_offset: Option<u32>,
    /// Length of match span
    pub span_len: Option<u32>,
    /// Context before match
    pub context_before: Option<String>,
    /// Context after match
    pub context_after: Option<String>,
    /// Pattern type for structural matches
    pub pattern_type: Option<PatternType>,
    /// Symbol name
    pub symbol_name: Option<String>,
    /// Function/class signature
    pub signature: Option<String>,
}

/// Latency breakdown by stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    /// Stage A (lexical) latency in milliseconds
    pub stage_a: u32,
    /// Stage B (symbol/AST) latency in milliseconds
    pub stage_b: u32,
    /// Stage C (semantic) latency in milliseconds (optional)
    pub stage_c: Option<u32>,
    /// Total latency in milliseconds
    pub total: u32,
}

/// Search response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results
    pub hits: Vec<SearchHit>,
    /// Total number of potential results
    pub total: u32,
    /// Latency breakdown
    pub latency_ms: LatencyBreakdown,
    /// Request trace ID
    pub trace_id: String,
    /// API version
    pub api_version: String,
    /// Index version
    pub index_version: String,
    /// Policy version
    pub policy_version: String,
    /// Optional error message
    pub error: Option<String>,
    /// Optional status message
    pub message: Option<String>,
}

/// System health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Health status
    pub status: String, // "ok", "degraded", "down"
    /// Timestamp of health check
    pub timestamp: String,
    /// Number of healthy shards
    pub shards_healthy: u32,
}

/// Compatibility check request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityCheckRequest {
    /// Client API version
    pub api_version: String,
    /// Client index version
    pub index_version: String,
    /// Client policy version (optional)
    pub policy_version: Option<String>,
    /// Allow mismatched versions flag
    pub allow_compat: Option<bool>,
}

/// Compatibility check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityCheckResponse {
    /// Whether versions are compatible
    pub compatible: bool,
    /// Client API version
    pub api_version: String,
    /// Client index version
    pub index_version: String,
    /// Client policy version
    pub policy_version: Option<String>,
    /// Server API version
    pub server_api_version: String,
    /// Server index version
    pub server_index_version: String,
    /// Server policy version
    pub server_policy_version: String,
    /// Compatibility warnings
    pub warnings: Option<Vec<String>>,
    /// Compatibility errors
    pub errors: Option<Vec<String>>,
}

/// SPI search request (LSP interface)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiSearchRequest {
    /// Search query
    pub query: String,
    /// Repository context
    pub repo: Option<String>,
    /// Number of results
    pub k: Option<u32>,
    /// Search mode
    pub mode: Option<String>,
}

/// SPI search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiSearchResponse {
    /// Search hits
    pub hits: Vec<SearchHit>,
    /// Total results count
    pub total: u32,
    /// Request latency
    pub latency_ms: u32,
}

/// SPI health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiHealthResponse {
    /// Service status
    pub status: String,
    /// Health details
    pub details: Option<serde_json::Value>,
}

/// Resolve request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveRequest {
    /// File path to resolve
    pub file: String,
    /// Line number
    pub line: u32,
    /// Column number
    pub col: u32,
}

/// Resolve response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveResponse {
    /// Whether resolution was successful
    pub resolved: bool,
    /// Resolved location
    pub location: Option<SearchHit>,
}

/// Context request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRequest {
    /// File path
    pub file: String,
    /// Line number
    pub line: u32,
    /// Context radius
    pub radius: Option<u32>,
}

/// Context response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResponse {
    /// Context lines
    pub context: Vec<String>,
    /// Starting line number
    pub start_line: u32,
}

/// Cross-reference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrefRequest {
    /// Symbol to find references for
    pub symbol: String,
    /// File context
    pub file: Option<String>,
    /// Line context
    pub line: Option<u32>,
}

/// Cross-reference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrefResponse {
    /// Symbol references
    pub references: Vec<SearchHit>,
    /// Total reference count
    pub total: u32,
}

/// Symbols list request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolsListRequest {
    /// File path to get symbols for
    pub file: String,
}

/// Symbols list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolsListResponse {
    /// Symbols in file
    pub symbols: Vec<SearchHit>,
    /// Total symbol count
    pub total: u32,
}

/// LSP capabilities response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPCapabilitiesResponse {
    /// Available LSP capabilities
    pub capabilities: Vec<String>,
}

/// LSP diagnostics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPDiagnosticsResponse {
    /// Diagnostic messages
    pub diagnostics: Vec<serde_json::Value>,
}

/// LSP format response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPFormatResponse {
    /// Whether formatting was applied
    pub formatted: bool,
    /// Formatted content
    pub content: Option<String>,
}

/// LSP selection ranges response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPSelectionRangesResponse {
    /// Selection ranges
    pub ranges: Vec<serde_json::Value>,
}

/// LSP folding ranges response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPFoldingRangesResponse {
    /// Folding ranges
    pub ranges: Vec<serde_json::Value>,
}

/// LSP prepare rename response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPPrepareRenameResponse {
    /// Whether rename is possible
    pub can_rename: bool,
    /// Rename range
    pub range: Option<serde_json::Value>,
}

/// LSP rename response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPRenameResponse {
    /// Text changes for rename
    pub changes: Vec<serde_json::Value>,
}

/// LSP code actions response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPCodeActionsResponse {
    /// Available code actions
    pub actions: Vec<serde_json::Value>,
}

/// LSP hierarchy response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPHierarchyResponse {
    /// Symbol hierarchy
    pub hierarchy: Vec<serde_json::Value>,
}

/// Precision optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOptimizationConfig {
    /// Block A early exit configuration
    pub block_a_early_exit: Option<BlockAEarlyExitConfig>,
    /// Block A ANN configuration  
    pub block_a_ann: Option<BlockAAnnConfig>,
    /// Block A gate configuration
    pub block_a_gate: Option<BlockAGateConfig>,
    /// Block B dynamic top-N configuration
    pub block_b_dynamic_topn: Option<BlockBDynamicTopnConfig>,
    /// Block C deduplication configuration
    pub block_c_dedup: Option<BlockCDedupConfig>,
}

/// Block A early exit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAEarlyExitConfig {
    pub enabled: bool,
    pub margin: f64,
    pub min_probes: u32,
}

/// Block A ANN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAAnnConfig {
    pub k: u32,
    #[serde(rename = "efSearch")]
    pub ef_search: u32,
}

/// Block A gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAGateConfig {
    pub nl_threshold: f64,
    pub min_candidates: u32,
    pub confidence_cutoff: f64,
}

/// Block B dynamic top-N configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBDynamicTopnConfig {
    pub enabled: bool,
    pub score_threshold: f64,
    pub hard_cap: u32,
}

/// Block C deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockCDedupConfig {
    pub in_file: InFileDedupConfig,
    pub cross_file: CrossFileDedupConfig,
}

/// In-file deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InFileDedupConfig {
    pub simhash: SimhashConfig,
    pub keep: u32,
}

/// Simhash configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimhashConfig {
    pub k: u32,
    pub hamming_max: u32,
}

/// Cross-file deduplication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFileDedupConfig {
    pub vendor_deboost: f64,
}

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Experiment name
    pub name: String,
    /// Experiment description
    pub description: Option<String>,
    /// Configuration parameters
    pub config: HashMap<String, serde_json::Value>,
}

// Validation helper functions
impl SearchRequest {
    /// Validate the search request
    pub fn validate(&self) -> Result<(), String> {
        if self.repo_sha.is_empty() {
            return Err("repo_sha cannot be empty".to_string());
        }
        
        if self.repo_sha.len() > 64 {
            return Err("repo_sha cannot exceed 64 characters".to_string());
        }
        
        if self.q.is_empty() {
            return Err("Query cannot be empty".to_string());
        }
        
        if self.q.len() > 1000 {
            return Err("query cannot exceed 1000 characters".to_string());
        }
        
        if self.fuzzy > 2 {
            return Err("fuzzy distance cannot exceed 2".to_string());
        }
        
        if self.k == 0 {
            return Err("k must be greater than 0".to_string());
        }
        
        if self.k > 200 {
            return Err("k cannot exceed 200".to_string());
        }
        
        if let Some(timeout) = self.timeout_ms {
            if timeout < 100 || timeout > 5000 {
                return Err("timeout_ms must be between 100 and 5000".to_string());
            }
        }
        
        Ok(())
    }
}

impl StructRequest {
    /// Validate the structural search request
    pub fn validate(&self) -> Result<(), String> {
        if self.repo_sha.is_empty() {
            return Err("repo_sha cannot be empty".to_string());
        }
        
        if self.pattern.is_empty() {
            return Err("pattern cannot be empty".to_string());
        }
        
        if self.pattern.len() > 500 {
            return Err("pattern cannot exceed 500 characters".to_string());
        }
        
        if let Some(max_results) = self.max_results {
            if max_results == 0 || max_results > 100 {
                return Err("max_results must be between 1 and 100".to_string());
            }
        }
        
        Ok(())
    }
}

impl SymbolsNearRequest {
    /// Validate the symbols near request
    pub fn validate(&self) -> Result<(), String> {
        if self.file.is_empty() {
            return Err("file cannot be empty".to_string());
        }
        
        if self.line == 0 {
            return Err("line must be >= 1".to_string());
        }
        
        if let Some(radius) = self.radius {
            if radius == 0 || radius > 50 {
                return Err("radius must be between 1 and 50".to_string());
            }
        }
        
        Ok(())
    }
}