use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Main configuration for the Lens search engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensConfig {
    /// Server configuration
    pub server: ServerConfig,
    
    /// Search engine configuration
    pub search: SearchConfig,
    
    /// Context selection engine configuration
    pub context_engine: ContextEngineConfig,
    
    /// LSP manager configuration
    pub lsp: LspConfig,
    
    /// Pipeline configuration
    pub pipeline: PipelineConfig,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Metrics and monitoring
    pub metrics: MetricsConfig,
    
    /// Benchmarking configuration
    pub benchmark: BenchmarkConfig,
}

/// Server-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host
    pub host: String,
    
    /// gRPC server port
    pub grpc_port: u16,
    
    /// Metrics server port
    pub metrics_port: u16,
    
    /// Enable TLS
    pub enable_tls: bool,
    
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
}

/// Search engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Tantivy index path
    pub index_path: PathBuf,
    
    /// Maximum search results per query
    pub max_results: usize,
    
    /// Default query timeout in milliseconds
    pub default_timeout_ms: u64,
    
    /// Vector search configuration
    pub vector: VectorConfig,
    
    /// Text search configuration
    pub text: TextConfig,
    
    /// Result ranking configuration
    pub ranking: RankingConfig,
}

/// Vector search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    /// Enable vector search
    pub enabled: bool,
    
    /// Vector model path
    pub model_path: Option<String>,
    
    /// Vector dimensions
    pub dimensions: usize,
    
    /// HNSW ef_search parameter
    pub ef_search: usize,
    
    /// Maximum candidates for vector search
    pub max_candidates: usize,
}

/// Text search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    /// Enable fuzzy matching
    pub enable_fuzzy: bool,
    
    /// Fuzzy match threshold (0.0-1.0)
    pub fuzzy_threshold: f64,
    
    /// Enable phrase queries
    pub enable_phrase: bool,
    
    /// Enable boolean queries
    pub enable_boolean: bool,
    
    /// Stemming configuration
    pub stemming: StemmingConfig,
}

/// Stemming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemmingConfig {
    /// Enable stemming
    pub enabled: bool,
    
    /// Language for stemming
    pub language: String,
}

/// Result ranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// BM25 k1 parameter
    pub bm25_k1: f64,
    
    /// BM25 b parameter
    pub bm25_b: f64,
    
    /// TF-IDF boost factor
    pub tfidf_boost: f64,
    
    /// Recency boost factor
    pub recency_boost: f64,
    
    /// Language-specific boost factors
    pub language_boosts: HashMap<String, f64>,
}

/// LSP manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfig {
    /// Enable LSP integration
    pub enabled: bool,
    
    /// LSP server configurations
    pub servers: HashMap<String, LspServerConfig>,
    
    /// LSP request timeout in milliseconds
    pub request_timeout_ms: u64,
    
    /// LSP server startup timeout in milliseconds
    pub startup_timeout_ms: u64,
    
    /// Maximum concurrent LSP requests
    pub max_concurrent_requests: usize,
    
    /// BFS configuration for symbol traversal
    pub bfs: BfsConfig,
}

/// Individual LSP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerConfig {
    /// Command to start the LSP server
    pub command: String,
    
    /// Command arguments
    pub args: Vec<String>,
    
    /// Working directory
    pub working_dir: Option<PathBuf>,
    
    /// Environment variables
    pub env: HashMap<String, String>,
    
    /// Server initialization options
    pub init_options: serde_json::Value,
    
    /// File extensions handled by this server
    pub file_extensions: Vec<String>,
}

/// BFS configuration for symbol traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfsConfig {
    /// Maximum BFS depth (≤2 per TODO.md)
    pub max_depth: u32,
    
    /// Maximum nodes to explore (≤64 per TODO.md)
    pub max_nodes: u32,
    
    /// Enable definition traversal
    pub enable_definitions: bool,
    
    /// Enable reference traversal
    pub enable_references: bool,
    
    /// Enable type traversal
    pub enable_types: bool,
    
    /// Enable implementation traversal
    pub enable_implementations: bool,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Enable zero-copy optimization
    pub zero_copy: bool,
    
    /// Pipeline stage buffer sizes
    pub buffer_sizes: HashMap<String, usize>,
    
    /// Enable async overlap between stages
    pub async_overlap: bool,
    
    /// Memory pool configuration
    pub memory_pool: MemoryPoolConfig,
    
    /// Stage fusion configuration
    pub fusion: FusionConfig,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size in MB
    pub initial_size_mb: usize,
    
    /// Maximum pool size in MB
    pub max_size_mb: usize,
    
    /// Buffer size in bytes
    pub buffer_size: usize,
    
    /// Enable buffer reuse
    pub enable_reuse: bool,
}

/// Stage fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Enable stage fusion
    pub enabled: bool,
    
    /// Fusion window size
    pub window_size: usize,
    
    /// Enable predictive termination
    pub enable_prediction: bool,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable hint caching
    pub enabled: bool,
    
    /// Maximum cache entries
    pub max_entries: usize,
    
    /// Cache TTL in hours (24h per TODO.md)
    pub ttl_hours: u64,
    
    /// Enable cache compression
    pub enable_compression: bool,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable Prometheus metrics
    pub enabled: bool,
    
    /// Metrics endpoint path
    pub endpoint: String,
    
    /// Export interval in seconds
    pub export_interval_s: u64,
    
    /// Enable detailed metrics
    pub detailed: bool,
}

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Enable benchmarking endpoints
    pub enabled: bool,
    
    /// Golden dataset path
    pub golden_dataset_path: PathBuf,
    
    /// Benchmark results output path
    pub results_path: PathBuf,
    
    /// Enable fraud-resistant attestation
    pub enable_attestation: bool,
}

/// Context selection engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEngineConfig {
    /// Enable hero defaults from promoted configurations
    pub enable_hero_defaults: bool,
    
    /// Hero configuration lock file path
    pub hero_lock_path: PathBuf,
    
    /// Context selection strategy
    pub strategy: ContextStrategy,
    
    /// Hero parameters for optimized context selection
    pub hero_params: HeroParams,
}

/// Context selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextStrategy {
    /// Use hero defaults for optimal performance
    Hero,
    /// Use adaptive configuration based on query type
    Adaptive,
    /// Use legacy configuration for backwards compatibility
    Legacy,
}

/// Hero parameters derived from promoted configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroParams {
    /// Fusion strategy (e.g., "aggressive_milvus")
    pub fusion: String,
    
    /// Chunk policy (e.g., "ce_large")
    pub chunk_policy: String,
    
    /// Chunk length in tokens
    pub chunk_len: u32,
    
    /// Overlap between chunks
    pub overlap: u32,
    
    /// Retrieval K parameter 
    pub retrieval_k: u32,
    
    /// RRF K0 parameter for ranking fusion
    pub rrf_k0: u32,
    
    /// Reranker type (e.g., "cross_encoder")
    pub reranker: String,
    
    /// Router type (e.g., "ml_v2")
    pub router: String,
    
    /// Maximum chunks per file
    pub max_chunks_per_file: u32,
    
    /// Symbol boost factor
    pub symbol_boost: f64,
    
    /// Graph expansion hops
    pub graph_expand_hops: u32,
    
    /// Graph added tokens cap
    pub graph_added_tokens_cap: u32,
}

impl Default for LensConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                grpc_port: 50051,
                metrics_port: 9090,
                enable_tls: false,
                request_timeout_ms: 5000,
                max_concurrent_requests: 1000,
            },
            search: SearchConfig {
                index_path: PathBuf::from("./data/index"),
                max_results: 50,
                default_timeout_ms: 2000,
                vector: VectorConfig {
                    enabled: false,
                    model_path: None,
                    dimensions: 768,
                    ef_search: 256,
                    max_candidates: 1000,
                },
                text: TextConfig {
                    enable_fuzzy: true,
                    fuzzy_threshold: 0.8,
                    enable_phrase: true,
                    enable_boolean: true,
                    stemming: StemmingConfig {
                        enabled: true,
                        language: "english".to_string(),
                    },
                },
                ranking: RankingConfig {
                    bm25_k1: 1.5,
                    bm25_b: 0.75,
                    tfidf_boost: 1.0,
                    recency_boost: 0.1,
                    language_boosts: {
                        let mut boosts = HashMap::new();
                        boosts.insert("rust".to_string(), 1.2);
                        boosts.insert("typescript".to_string(), 1.1);
                        boosts.insert("python".to_string(), 1.1);
                        boosts.insert("javascript".to_string(), 1.0);
                        boosts
                    },
                },
            },
            context_engine: ContextEngineConfig {
                enable_hero_defaults: true,
                hero_lock_path: PathBuf::from("./release/hero.lock.json"),
                strategy: ContextStrategy::Hero,
                hero_params: HeroParams {
                    fusion: "aggressive_milvus".to_string(),
                    chunk_policy: "ce_large".to_string(),
                    chunk_len: 384,
                    overlap: 128,
                    retrieval_k: 20,
                    rrf_k0: 60,
                    reranker: "cross_encoder".to_string(),
                    router: "ml_v2".to_string(),
                    max_chunks_per_file: 50,
                    symbol_boost: 1.2,
                    graph_expand_hops: 2,
                    graph_added_tokens_cap: 256,
                },
            },
            lsp: LspConfig {
                enabled: true,
                servers: Self::default_lsp_servers(),
                request_timeout_ms: 1000,
                startup_timeout_ms: 10000,
                max_concurrent_requests: 100,
                bfs: BfsConfig {
                    max_depth: 2,        // ≤2 per TODO.md
                    max_nodes: 64,       // ≤64 per TODO.md
                    enable_definitions: true,
                    enable_references: true,
                    enable_types: true,
                    enable_implementations: true,
                },
            },
            pipeline: PipelineConfig {
                zero_copy: true,
                buffer_sizes: {
                    let mut sizes = HashMap::new();
                    sizes.insert("query_parse".to_string(), 1024);
                    sizes.insert("index_search".to_string(), 4096);
                    sizes.insert("result_merge".to_string(), 2048);
                    sizes
                },
                async_overlap: true,
                memory_pool: MemoryPoolConfig {
                    initial_size_mb: 64,
                    max_size_mb: 512,
                    buffer_size: 4096,
                    enable_reuse: true,
                },
                fusion: FusionConfig {
                    enabled: true,
                    window_size: 4,
                    enable_prediction: true,
                },
            },
            cache: CacheConfig {
                enabled: true,
                max_entries: 10000,
                ttl_hours: 24,       // 24h per TODO.md
                enable_compression: true,
            },
            metrics: MetricsConfig {
                enabled: true,
                endpoint: "/metrics".to_string(),
                export_interval_s: 60,
                detailed: false,
            },
            benchmark: BenchmarkConfig {
                enabled: true,
                golden_dataset_path: PathBuf::from("./golden-dataset.json"),
                results_path: PathBuf::from("./benchmark-results"),
                enable_attestation: true,
            },
        }
    }
}

impl ContextEngineConfig {
    /// Load hero parameters from hero lock file
    pub async fn load_hero_params(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.enable_hero_defaults {
            return Ok(());
        }

        let content = tokio::fs::read_to_string(&self.hero_lock_path).await?;
        let hero_data: serde_json::Value = serde_json::from_str(&content)?;
        
        if let Some(params) = hero_data.get("params") {
            if let Some(fusion) = params.get("fusion").and_then(|v| v.as_str()) {
                self.hero_params.fusion = fusion.to_string();
            }
            if let Some(chunk_policy) = params.get("chunk_policy").and_then(|v| v.as_str()) {
                self.hero_params.chunk_policy = chunk_policy.to_string();
            }
            if let Some(chunk_len) = params.get("chunk_len").and_then(|v| v.as_u64()) {
                self.hero_params.chunk_len = chunk_len as u32;
            }
            if let Some(overlap) = params.get("overlap").and_then(|v| v.as_u64()) {
                self.hero_params.overlap = overlap as u32;
            }
            if let Some(retrieval_k) = params.get("retrieval_k").and_then(|v| v.as_u64()) {
                self.hero_params.retrieval_k = retrieval_k as u32;
            }
            if let Some(rrf_k0) = params.get("rrf_k0").and_then(|v| v.as_u64()) {
                self.hero_params.rrf_k0 = rrf_k0 as u32;
            }
            if let Some(reranker) = params.get("reranker").and_then(|v| v.as_str()) {
                self.hero_params.reranker = reranker.to_string();
            }
            if let Some(router) = params.get("router").and_then(|v| v.as_str()) {
                self.hero_params.router = router.to_string();
            }
            if let Some(max_chunks_per_file) = params.get("max_chunks_per_file").and_then(|v| v.as_u64()) {
                self.hero_params.max_chunks_per_file = max_chunks_per_file as u32;
            }
            if let Some(symbol_boost) = params.get("symbol_boost").and_then(|v| v.as_f64()) {
                self.hero_params.symbol_boost = symbol_boost;
            }
            if let Some(graph_expand_hops) = params.get("graph_expand_hops").and_then(|v| v.as_u64()) {
                self.hero_params.graph_expand_hops = graph_expand_hops as u32;
            }
            if let Some(graph_added_tokens_cap) = params.get("graph_added_tokens_cap").and_then(|v| v.as_u64()) {
                self.hero_params.graph_added_tokens_cap = graph_added_tokens_cap as u32;
            }
        }

        Ok(())
    }

    /// Get the current configuration strategy label for metrics
    pub fn strategy_label(&self) -> &str {
        match self.strategy {
            ContextStrategy::Hero => "hero",
            ContextStrategy::Adaptive => "adaptive", 
            ContextStrategy::Legacy => "legacy",
        }
    }
}

impl LensConfig {
    /// Default LSP server configurations
    fn default_lsp_servers() -> HashMap<String, LspServerConfig> {
        let mut servers = HashMap::new();

        // TypeScript/JavaScript server
        servers.insert("tsserver".to_string(), LspServerConfig {
            command: "typescript-language-server".to_string(),
            args: vec!["--stdio".to_string()],
            working_dir: None,
            env: HashMap::new(),
            init_options: serde_json::json!({
                "preferences": {
                    "includeInlayParameterNameHints": "all",
                    "includeInlayVariableTypeHints": true
                }
            }),
            file_extensions: vec![".ts".to_string(), ".tsx".to_string(), ".js".to_string(), ".jsx".to_string()],
        });

        // Python server
        servers.insert("pylsp".to_string(), LspServerConfig {
            command: "pylsp".to_string(),
            args: vec![],
            working_dir: None,
            env: HashMap::new(),
            init_options: serde_json::json!({
                "settings": {
                    "pylsp": {
                        "plugins": {
                            "pycodestyle": {"enabled": false},
                            "mccabe": {"enabled": false}
                        }
                    }
                }
            }),
            file_extensions: vec![".py".to_string()],
        });

        // Rust server
        servers.insert("rust-analyzer".to_string(), LspServerConfig {
            command: "rust-analyzer".to_string(),
            args: vec![],
            working_dir: None,
            env: HashMap::new(),
            init_options: serde_json::json!({
                "cargo": {
                    "buildScripts": {
                        "enable": true
                    }
                }
            }),
            file_extensions: vec![".rs".to_string()],
        });

        // Go server
        servers.insert("gopls".to_string(), LspServerConfig {
            command: "gopls".to_string(),
            args: vec![],
            working_dir: None,
            env: HashMap::new(),
            init_options: serde_json::json!({}),
            file_extensions: vec![".go".to_string()],
        });

        servers
    }

    /// Load configuration from file
    pub async fn load_from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub async fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.server.grpc_port == self.server.metrics_port {
            return Err("gRPC and metrics ports cannot be the same".to_string());
        }

        if self.lsp.enabled && self.lsp.servers.is_empty() {
            return Err("LSP is enabled but no servers are configured".to_string());
        }

        if self.lsp.bfs.max_depth > 2 {
            return Err("BFS max depth must be ≤2 per TODO.md requirements".to_string());
        }

        if self.lsp.bfs.max_nodes > 64 {
            return Err("BFS max nodes must be ≤64 per TODO.md requirements".to_string());
        }

        if self.cache.ttl_hours > 48 {
            return Err("Cache TTL should not exceed 48 hours for memory management".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = LensConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_bfs_limits() {
        let mut config = LensConfig::default();
        
        // Test max depth limit
        config.lsp.bfs.max_depth = 3;
        assert!(config.validate().is_err());
        
        config.lsp.bfs.max_depth = 2;
        config.lsp.bfs.max_nodes = 65;
        assert!(config.validate().is_err());
        
        config.lsp.bfs.max_nodes = 64;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_port_conflict() {
        let mut config = LensConfig::default();
        config.server.metrics_port = config.server.grpc_port;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_lsp_enabled_without_servers() {
        let mut config = LensConfig::default();
        config.lsp.enabled = true;
        config.lsp.servers.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cache_ttl_limits() {
        let mut config = LensConfig::default();
        config.cache.ttl_hours = 49;
        assert!(config.validate().is_err());
        
        config.cache.ttl_hours = 24;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig {
            host: "127.0.0.1".to_string(),
            grpc_port: 50051,
            metrics_port: 9090,
            enable_tls: false,
            request_timeout_ms: 5000,
            max_concurrent_requests: 1000,
        };
        
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.grpc_port, 50051);
        assert_eq!(config.metrics_port, 9090);
        assert!(!config.enable_tls);
    }

    #[test]
    fn test_search_config_defaults() {
        let config = SearchConfig {
            index_path: PathBuf::from("./data/index"),
            max_results: 50,
            default_timeout_ms: 2000,
            vector: VectorConfig {
                enabled: false,
                model_path: None,
                dimensions: 768,
                ef_search: 256,
                max_candidates: 1000,
            },
            text: TextConfig {
                enable_fuzzy: true,
                fuzzy_threshold: 0.8,
                enable_phrase: true,
                enable_boolean: true,
                stemming: StemmingConfig {
                    enabled: true,
                    language: "english".to_string(),
                },
            },
            ranking: RankingConfig {
                bm25_k1: 1.5,
                bm25_b: 0.75,
                tfidf_boost: 1.0,
                recency_boost: 0.1,
                language_boosts: HashMap::new(),
            },
        };
        
        assert_eq!(config.max_results, 50);
        assert_eq!(config.default_timeout_ms, 2000);
        assert!(!config.vector.enabled);
        assert!(config.text.enable_fuzzy);
        assert_eq!(config.text.fuzzy_threshold, 0.8);
    }

    #[test]
    fn test_vector_config() {
        let config = VectorConfig {
            enabled: true,
            model_path: Some("./model.bin".to_string()),
            dimensions: 384,
            ef_search: 128,
            max_candidates: 500,
        };
        
        assert!(config.enabled);
        assert_eq!(config.model_path, Some("./model.bin".to_string()));
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.ef_search, 128);
    }

    #[test]
    fn test_text_config() {
        let stemming = StemmingConfig {
            enabled: false,
            language: "german".to_string(),
        };
        
        let config = TextConfig {
            enable_fuzzy: false,
            fuzzy_threshold: 0.9,
            enable_phrase: false,
            enable_boolean: false,
            stemming,
        };
        
        assert!(!config.enable_fuzzy);
        assert_eq!(config.fuzzy_threshold, 0.9);
        assert!(!config.enable_phrase);
        assert!(!config.stemming.enabled);
        assert_eq!(config.stemming.language, "german");
    }

    #[test]
    fn test_ranking_config() {
        let mut language_boosts = HashMap::new();
        language_boosts.insert("rust".to_string(), 2.0);
        language_boosts.insert("go".to_string(), 1.5);
        
        let config = RankingConfig {
            bm25_k1: 1.2,
            bm25_b: 0.8,
            tfidf_boost: 1.5,
            recency_boost: 0.2,
            language_boosts,
        };
        
        assert_eq!(config.bm25_k1, 1.2);
        assert_eq!(config.bm25_b, 0.8);
        assert_eq!(config.tfidf_boost, 1.5);
        assert_eq!(config.language_boosts.get("rust"), Some(&2.0));
    }

    #[test]
    fn test_lsp_config() {
        let servers = HashMap::new();
        let bfs = BfsConfig {
            max_depth: 1,
            max_nodes: 32,
            enable_definitions: false,
            enable_references: false,
            enable_types: true,
            enable_implementations: true,
        };
        
        let config = LspConfig {
            enabled: false,
            servers,
            request_timeout_ms: 500,
            startup_timeout_ms: 5000,
            max_concurrent_requests: 50,
            bfs,
        };
        
        assert!(!config.enabled);
        assert_eq!(config.request_timeout_ms, 500);
        assert_eq!(config.bfs.max_depth, 1);
        assert_eq!(config.bfs.max_nodes, 32);
        assert!(!config.bfs.enable_definitions);
        assert!(config.bfs.enable_types);
    }

    #[test]
    fn test_lsp_server_config() {
        let mut env = HashMap::new();
        env.insert("LANG".to_string(), "en_US.UTF-8".to_string());
        
        let config = LspServerConfig {
            command: "node".to_string(),
            args: vec!["--inspect".to_string(), "server.js".to_string()],
            working_dir: Some(PathBuf::from("/opt/lsp")),
            env,
            init_options: serde_json::json!({"debug": true}),
            file_extensions: vec![".js".to_string(), ".ts".to_string()],
        };
        
        assert_eq!(config.command, "node");
        assert_eq!(config.args.len(), 2);
        assert_eq!(config.working_dir, Some(PathBuf::from("/opt/lsp")));
        assert_eq!(config.file_extensions.len(), 2);
    }

    #[test]
    fn test_bfs_config() {
        let config = BfsConfig {
            max_depth: 2,
            max_nodes: 64,
            enable_definitions: true,
            enable_references: true,
            enable_types: false,
            enable_implementations: false,
        };
        
        assert_eq!(config.max_depth, 2);
        assert_eq!(config.max_nodes, 64);
        assert!(config.enable_definitions);
        assert!(config.enable_references);
        assert!(!config.enable_types);
        assert!(!config.enable_implementations);
    }

    #[test]
    fn test_pipeline_config() {
        let mut buffer_sizes = HashMap::new();
        buffer_sizes.insert("stage1".to_string(), 2048);
        buffer_sizes.insert("stage2".to_string(), 4096);
        
        let memory_pool = MemoryPoolConfig {
            initial_size_mb: 32,
            max_size_mb: 256,
            buffer_size: 2048,
            enable_reuse: false,
        };
        
        let fusion = FusionConfig {
            enabled: false,
            window_size: 2,
            enable_prediction: false,
        };
        
        let config = PipelineConfig {
            zero_copy: false,
            buffer_sizes,
            async_overlap: false,
            memory_pool,
            fusion,
        };
        
        assert!(!config.zero_copy);
        assert!(!config.async_overlap);
        assert_eq!(config.buffer_sizes.len(), 2);
        assert_eq!(config.memory_pool.initial_size_mb, 32);
        assert!(!config.fusion.enabled);
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig {
            initial_size_mb: 128,
            max_size_mb: 1024,
            buffer_size: 8192,
            enable_reuse: true,
        };
        
        assert_eq!(config.initial_size_mb, 128);
        assert_eq!(config.max_size_mb, 1024);
        assert_eq!(config.buffer_size, 8192);
        assert!(config.enable_reuse);
    }

    #[test]
    fn test_fusion_config() {
        let config = FusionConfig {
            enabled: true,
            window_size: 8,
            enable_prediction: true,
        };
        
        assert!(config.enabled);
        assert_eq!(config.window_size, 8);
        assert!(config.enable_prediction);
    }

    #[test]
    fn test_cache_config() {
        let config = CacheConfig {
            enabled: false,
            max_entries: 5000,
            ttl_hours: 12,
            enable_compression: false,
        };
        
        assert!(!config.enabled);
        assert_eq!(config.max_entries, 5000);
        assert_eq!(config.ttl_hours, 12);
        assert!(!config.enable_compression);
    }

    #[test]
    fn test_metrics_config() {
        let config = MetricsConfig {
            enabled: true,
            endpoint: "/health".to_string(),
            export_interval_s: 30,
            detailed: true,
        };
        
        assert!(config.enabled);
        assert_eq!(config.endpoint, "/health");
        assert_eq!(config.export_interval_s, 30);
        assert!(config.detailed);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig {
            enabled: false,
            golden_dataset_path: PathBuf::from("/data/golden.json"),
            results_path: PathBuf::from("/output/results"),
            enable_attestation: false,
        };
        
        assert!(!config.enabled);
        assert_eq!(config.golden_dataset_path, PathBuf::from("/data/golden.json"));
        assert_eq!(config.results_path, PathBuf::from("/output/results"));
        assert!(!config.enable_attestation);
    }

    #[test]
    fn test_default_lsp_servers() {
        let servers = LensConfig::default_lsp_servers();
        
        assert!(servers.contains_key("tsserver"));
        assert!(servers.contains_key("pylsp"));
        assert!(servers.contains_key("rust-analyzer"));
        assert!(servers.contains_key("gopls"));
        
        let ts_server = servers.get("tsserver").unwrap();
        assert_eq!(ts_server.command, "typescript-language-server");
        assert!(ts_server.file_extensions.contains(&".ts".to_string()));
        assert!(ts_server.file_extensions.contains(&".js".to_string()));
        
        let py_server = servers.get("pylsp").unwrap();
        assert_eq!(py_server.command, "pylsp");
        assert!(py_server.file_extensions.contains(&".py".to_string()));
        
        let rust_server = servers.get("rust-analyzer").unwrap();
        assert_eq!(rust_server.command, "rust-analyzer");
        assert!(rust_server.file_extensions.contains(&".rs".to_string()));
        
        let go_server = servers.get("gopls").unwrap();
        assert_eq!(go_server.command, "gopls");
        assert!(go_server.file_extensions.contains(&".go".to_string()));
    }

    #[test]
    fn test_lens_config_default() {
        let config = LensConfig::default();
        
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.grpc_port, 50051);
        assert_eq!(config.search.max_results, 50);
        assert!(config.lsp.enabled);
        assert!(config.cache.enabled);
        assert!(config.metrics.enabled);
        assert!(config.benchmark.enabled);
        
        // Test language boosts
        let rust_boost = config.search.ranking.language_boosts.get("rust");
        assert_eq!(rust_boost, Some(&1.2));
        
        let ts_boost = config.search.ranking.language_boosts.get("typescript");
        assert_eq!(ts_boost, Some(&1.1));
    }

    #[test]
    fn test_serialization_deserialization() {
        let config = LensConfig::default();
        
        // Test serialization
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());
        
        // Test deserialization
        let deserialized: LensConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.server.host, config.server.host);
        assert_eq!(deserialized.search.max_results, config.search.max_results);
    }

    #[test]
    fn test_edge_cases() {
        let mut config = LensConfig::default();
        
        // Test with zero ports (different values to avoid conflict)
        config.server.grpc_port = 0;
        config.server.metrics_port = 1;
        
        // Test empty LSP servers when disabled
        config.lsp.enabled = false;
        config.lsp.servers.clear();
        assert!(config.validate().is_ok()); // Should be OK when LSP is disabled
        
        // Test maximum allowed values
        config.lsp.bfs.max_depth = 2;
        config.lsp.bfs.max_nodes = 64;
        config.cache.ttl_hours = 48;
        assert!(config.validate().is_ok());
        
        // Test minimum values
        config.lsp.bfs.max_depth = 0;
        config.lsp.bfs.max_nodes = 0;
        config.cache.ttl_hours = 1;
        assert!(config.validate().is_ok());
    }

    #[tokio::test]
    async fn test_config_file_operations() {
        use tempfile::TempDir;
        
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("lens.toml");
        
        let original_config = LensConfig::default();
        
        // Test saving config to file
        let save_result = original_config.save_to_file(&config_path).await;
        assert!(save_result.is_ok());
        
        // Test loading config from file
        let loaded_config = LensConfig::load_from_file(&config_path).await;
        assert!(loaded_config.is_ok());
        
        let loaded = loaded_config.unwrap();
        assert_eq!(loaded.server.host, original_config.server.host);
        assert_eq!(loaded.search.max_results, original_config.search.max_results);
    }

    #[test]
    fn test_context_engine_config() {
        let hero_params = HeroParams {
            fusion: "aggressive_milvus".to_string(),
            chunk_policy: "ce_large".to_string(),
            chunk_len: 384,
            overlap: 128,
            retrieval_k: 20,
            rrf_k0: 60,
            reranker: "cross_encoder".to_string(),
            router: "ml_v2".to_string(),
            max_chunks_per_file: 50,
            symbol_boost: 1.2,
            graph_expand_hops: 2,
            graph_added_tokens_cap: 256,
        };

        let config = ContextEngineConfig {
            enable_hero_defaults: true,
            hero_lock_path: PathBuf::from("./release/hero.lock.json"),
            strategy: ContextStrategy::Hero,
            hero_params,
        };

        assert!(config.enable_hero_defaults);
        assert_eq!(config.hero_lock_path, PathBuf::from("./release/hero.lock.json"));
        assert_eq!(config.strategy_label(), "hero");
        assert_eq!(config.hero_params.fusion, "aggressive_milvus");
        assert_eq!(config.hero_params.chunk_len, 384);
        assert_eq!(config.hero_params.symbol_boost, 1.2);
    }

    #[test]
    fn test_context_strategy_labels() {
        let mut config = ContextEngineConfig {
            enable_hero_defaults: true,
            hero_lock_path: PathBuf::from("./release/hero.lock.json"),
            strategy: ContextStrategy::Hero,
            hero_params: HeroParams {
                fusion: "aggressive_milvus".to_string(),
                chunk_policy: "ce_large".to_string(),
                chunk_len: 384,
                overlap: 128,
                retrieval_k: 20,
                rrf_k0: 60,
                reranker: "cross_encoder".to_string(),
                router: "ml_v2".to_string(),
                max_chunks_per_file: 50,
                symbol_boost: 1.2,
                graph_expand_hops: 2,
                graph_added_tokens_cap: 256,
            },
        };

        assert_eq!(config.strategy_label(), "hero");
        
        config.strategy = ContextStrategy::Adaptive;
        assert_eq!(config.strategy_label(), "adaptive");
        
        config.strategy = ContextStrategy::Legacy;
        assert_eq!(config.strategy_label(), "legacy");
    }

    #[test]
    fn test_hero_params_serialization() {
        let hero_params = HeroParams {
            fusion: "aggressive_milvus".to_string(),
            chunk_policy: "ce_large".to_string(),
            chunk_len: 384,
            overlap: 128,
            retrieval_k: 20,
            rrf_k0: 60,
            reranker: "cross_encoder".to_string(),
            router: "ml_v2".to_string(),
            max_chunks_per_file: 50,
            symbol_boost: 1.2,
            graph_expand_hops: 2,
            graph_added_tokens_cap: 256,
        };

        // Test serialization
        let serialized = serde_json::to_string(&hero_params).unwrap();
        assert!(!serialized.is_empty());
        assert!(serialized.contains("aggressive_milvus"));
        assert!(serialized.contains("cross_encoder"));

        // Test deserialization
        let deserialized: HeroParams = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.fusion, hero_params.fusion);
        assert_eq!(deserialized.chunk_len, hero_params.chunk_len);
        assert_eq!(deserialized.symbol_boost, hero_params.symbol_boost);
    }

    #[test]
    fn test_default_config_includes_context_engine() {
        let config = LensConfig::default();
        
        assert!(config.context_engine.enable_hero_defaults);
        assert_eq!(config.context_engine.hero_lock_path, PathBuf::from("./release/hero.lock.json"));
        assert_eq!(config.context_engine.strategy_label(), "hero");
        assert_eq!(config.context_engine.hero_params.fusion, "aggressive_milvus");
        assert_eq!(config.context_engine.hero_params.retrieval_k, 20);
        assert_eq!(config.context_engine.hero_params.rrf_k0, 60);
    }
}