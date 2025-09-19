//! Lens LSP Server - Real Language Server Protocol implementation
//!
//! This is a complete LSP server implementation that provides code search
//! capabilities through the Language Server Protocol. This replaces any
//! simulation or mock LSP functionality with a real, production-ready server.

use anyhow::Result;
use dashmap::DashMap;
use lens_search_engine::{QueryBuilder, SearchEngine};
use lsp_types::*;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result as JsonRpcResult;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::{debug, error, info, warn};

pub mod capabilities;
pub mod handlers;
pub mod workspace;

pub use capabilities::*;
pub use handlers::*;
pub use workspace::*;

/// Real LSP server implementation for Lens
pub struct LensLspServer {
    client: Client,
    search_engine: Arc<SearchEngine>,
    workspace: Arc<RwLock<Workspace>>,
    document_cache: Arc<DashMap<Url, TextDocumentItem>>,
    config: LspServerConfig,
}

/// Configuration for the LSP server
#[derive(Debug, Clone)]
pub struct LspServerConfig {
    /// Maximum number of search results to return
    pub max_search_results: usize,
    /// Whether to enable fuzzy search by default
    pub enable_fuzzy_search: bool,
    /// Whether to enable semantic search features
    pub enable_semantic_search: bool,
    /// Debounce delay for search requests (milliseconds)
    pub search_debounce_ms: u64,
    /// Whether to cache search results
    pub enable_result_caching: bool,
    /// Glob patterns excluded from workspace indexing and watching
    pub workspace_exclude_patterns: Vec<String>,
}

impl Default for LspServerConfig {
    fn default() -> Self {
        Self {
            max_search_results: 50,
            enable_fuzzy_search: true,
            enable_semantic_search: false,
            search_debounce_ms: 300,
            enable_result_caching: true,
            workspace_exclude_patterns: vec![
                "**/node_modules/**".to_string(),
                "**/target/**".to_string(),
                "**/.git/**".to_string(),
                "**/dist/**".to_string(),
                "**/build/**".to_string(),
                "**/__pycache__/**".to_string(),
            ],
        }
    }
}

impl LensLspServer {
    /// Create a new LSP server with search engine
    pub fn new(client: Client, search_engine: Arc<SearchEngine>) -> Self {
        Self::with_config(client, search_engine, LspServerConfig::default())
    }

    /// Create a new LSP server with custom configuration
    pub fn with_config(
        client: Client,
        search_engine: Arc<SearchEngine>,
        config: LspServerConfig,
    ) -> Self {
        let workspace_config = match WorkspaceConfig::default()
            .with_exclude_patterns(config.workspace_exclude_patterns.clone())
        {
            Ok(cfg) => cfg,
            Err(err) => {
                warn!(
                    "Failed to apply workspace exclude patterns ({}); falling back to defaults",
                    err
                );
                WorkspaceConfig::default()
            }
        };

        Self {
            client,
            search_engine,
            workspace: Arc::new(RwLock::new(Workspace::with_config(workspace_config))),
            document_cache: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Convert search results to LSP locations
    async fn search_results_to_locations(
        &self,
        results: &lens_search_engine::SearchResults,
    ) -> Vec<Location> {
        let mut locations = Vec::new();

        for result in &results.results {
            // Convert file path to URI
            let uri = match Url::from_file_path(&result.file_path) {
                Ok(uri) => uri,
                Err(_) => {
                    warn!("Failed to convert file path to URI: {}", result.file_path);
                    continue;
                }
            };

            // Create LSP range (0-based coordinates)
            let range = Range {
                start: Position {
                    line: result.line_number.saturating_sub(1),
                    character: result.column.saturating_sub(1),
                },
                end: Position {
                    line: result.line_number.saturating_sub(1),
                    character: result.column.saturating_sub(1) + result.content.len() as u32,
                },
            };

            locations.push(Location { uri, range });
        }

        locations
    }

    /// Perform search based on workspace symbol request
    async fn search_workspace_symbols(&self, query: &str) -> Result<Vec<SymbolInformation>> {
        info!("Searching workspace symbols for: {}", query);

        // Build search query
        let search_query = QueryBuilder::new(query)
            .symbol()
            .limit(self.config.max_search_results)
            .build();

        // Execute search
        let results = self.search_engine.search(&search_query).await?;

        // Convert to LSP symbol information
        let mut symbols = Vec::new();
        for result in results.results {
            // Convert file path to URI
            let uri = match Url::from_file_path(&result.file_path) {
                Ok(uri) => uri,
                Err(_) => continue,
            };

            // Determine symbol kind based on result type
            let kind = match result.result_type {
                lens_search_engine::SearchResultType::Text => SymbolKind::VARIABLE,
                lens_search_engine::SearchResultType::Function => SymbolKind::FUNCTION,
                lens_search_engine::SearchResultType::Class => SymbolKind::CLASS,
                lens_search_engine::SearchResultType::Variable => SymbolKind::VARIABLE,
                lens_search_engine::SearchResultType::Import => SymbolKind::MODULE,
                lens_search_engine::SearchResultType::Comment => SymbolKind::STRING,
                lens_search_engine::SearchResultType::Symbol => SymbolKind::VARIABLE,
            };

            // Create location
            let location = Location {
                uri,
                range: Range {
                    start: Position {
                        line: result.line_number.saturating_sub(1),
                        character: result.column.saturating_sub(1),
                    },
                    end: Position {
                        line: result.line_number.saturating_sub(1),
                        character: result.column.saturating_sub(1) + result.content.len() as u32,
                    },
                },
            };

            #[allow(deprecated)]
            symbols.push(SymbolInformation {
                name: result.content.clone(),
                kind,
                tags: None,
                deprecated: None,
                location,
                container_name: Some(result.file_path.clone()),
            });
        }

        Ok(symbols)
    }

    /// Search for text in workspace
    async fn search_text_in_workspace(&self, query: &str) -> Result<Vec<Location>> {
        debug!("Searching text in workspace: {}", query);

        // Build search query based on configuration
        let search_query = if self.config.enable_fuzzy_search {
            QueryBuilder::new(query)
                .fuzzy()
                .limit(self.config.max_search_results)
                .build()
        } else {
            QueryBuilder::new(query)
                .limit(self.config.max_search_results)
                .build()
        };

        // Execute search
        let results = self.search_engine.search(&search_query).await?;

        // Convert to locations
        let locations = self.search_results_to_locations(&results).await;

        Ok(locations)
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for LensLspServer {
    async fn initialize(&self, params: InitializeParams) -> JsonRpcResult<InitializeResult> {
        info!("Initializing Lens LSP Server");

        // Store workspace folders
        if let Some(workspace_folders) = params.workspace_folders {
            let mut workspace = self.workspace.write().await;
            for folder in workspace_folders {
                workspace.add_folder(folder).await;
            }
        } else if let Some(root_uri) = params.root_uri {
            let mut workspace = self.workspace.write().await;
            workspace.add_root_uri(root_uri).await;
        }

        Ok(InitializeResult {
            capabilities: create_server_capabilities(),
            server_info: Some(ServerInfo {
                name: "Lens LSP Server".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
            offset_encoding: None,
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        info!("Lens LSP Server initialized");

        // Send initial status message
        self.client
            .log_message(MessageType::INFO, "Lens LSP Server is ready")
            .await;

        // Index workspace if any folders are present
        let workspace = self.workspace.read().await;
        if !workspace.folders.is_empty() {
            drop(workspace);

            // Spawn indexing task
            let search_engine = self.search_engine.clone();
            let workspace_clone = self.workspace.clone();
            let client = self.client.clone();

            tokio::spawn(async move {
                let workspace = workspace_clone.read().await;
                for folder in &workspace.folders {
                    client
                        .log_message(
                            MessageType::INFO,
                            format!("Indexing workspace: {}", folder.uri),
                        )
                        .await;

                    if let Ok(path) = folder.uri.to_file_path() {
                        match search_engine.index_directory(&path).await {
                            Ok(stats) => {
                                client
                                    .log_message(
                                        MessageType::INFO,
                                        format!(
                                            "Indexed {} files in {:?}",
                                            stats.files_indexed, stats.indexing_duration
                                        ),
                                    )
                                    .await;
                            }
                            Err(e) => {
                                client
                                    .log_message(
                                        MessageType::ERROR,
                                        format!("Failed to index workspace: {}", e),
                                    )
                                    .await;
                            }
                        }

                        // Start watching the directory for changes
                        if let Err(e) = search_engine.start_watching_directory(&path).await {
                            client
                                .log_message(
                                    MessageType::ERROR,
                                    format!("Failed to start watching directory {:?}: {}", path, e),
                                )
                                .await;
                        } else {
                            client
                                .log_message(
                                    MessageType::INFO,
                                    format!("Started watching directory: {:?}", path),
                                )
                                .await;
                        }
                    }
                }
            });
        }
    }

    async fn shutdown(&self) -> JsonRpcResult<()> {
        info!("Shutting down Lens LSP Server");
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        debug!("Document opened: {}", params.text_document.uri);

        // Cache the document
        let doc = TextDocumentItem {
            uri: params.text_document.uri.clone(),
            language_id: params.text_document.language_id.clone(),
            version: params.text_document.version,
            text: params.text_document.text.clone(),
        };

        self.document_cache.insert(params.text_document.uri, doc);
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        debug!("Document changed: {}", params.text_document.uri);

        // Update cached document
        if let Some(mut doc) = self.document_cache.get_mut(&params.text_document.uri) {
            doc.version = params.text_document.version;

            // Apply changes (simplified - assumes full document replacement)
            for change in params.content_changes {
                if change.range.is_none() {
                    // Full document update
                    doc.text = change.text;
                }
            }
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        debug!("Document closed: {}", params.text_document.uri);
        self.document_cache.remove(&params.text_document.uri);
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> JsonRpcResult<Option<Vec<SymbolInformation>>> {
        debug!("Workspace symbol search: {}", params.query);

        match self.search_workspace_symbols(&params.query).await {
            Ok(symbols) => Ok(Some(symbols)),
            Err(e) => {
                error!("Workspace symbol search failed: {}", e);
                Ok(Some(Vec::new()))
            }
        }
    }

    async fn references(&self, params: ReferenceParams) -> JsonRpcResult<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        debug!(
            "Finding references at {}:{}:{}",
            uri, position.line, position.character
        );

        // Get the word at the position
        if let Some(doc) = self.document_cache.get(uri) {
            let word = extract_word_at_position(&doc.text, position);

            if !word.is_empty() {
                match self.search_text_in_workspace(&word).await {
                    Ok(locations) => return Ok(Some(locations)),
                    Err(e) => {
                        error!("Reference search failed: {}", e);
                    }
                }
            }
        }

        Ok(Some(Vec::new()))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> JsonRpcResult<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        debug!(
            "Finding definition at {}:{}:{}",
            uri, position.line, position.character
        );

        // Get the document text
        if let Some(doc) = self.document_cache.get(uri) {
            match handle_goto_definition(
                &self.search_engine,
                &doc.text,
                uri,
                position,
                self.config.max_search_results,
            )
            .await
            {
                Ok(locations) => {
                    if !locations.is_empty() {
                        return Ok(Some(GotoDefinitionResponse::Array(locations)));
                    }
                }
                Err(e) => {
                    error!("Enhanced definition search failed: {}", e);
                }
            }
        }

        Ok(None)
    }

    async fn completion(
        &self,
        params: CompletionParams,
    ) -> JsonRpcResult<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        debug!(
            "Completion request at {}:{}:{}",
            uri, position.line, position.character
        );

        // Get partial word for completion
        if let Some(doc) = self.document_cache.get(uri) {
            let partial_word = extract_partial_word_at_position(&doc.text, position);

            if partial_word.len() >= 2 {
                // Only complete after 2+ characters
                // Search for matching symbols
                let search_query = QueryBuilder::new(&partial_word).fuzzy().limit(20).build();

                match self.search_engine.search(&search_query).await {
                    Ok(results) => {
                        let mut completion_items = Vec::new();

                        for result in results.results.iter().take(10) {
                            // Extract symbol from content
                            let symbol = extract_symbol_from_content(&result.content);

                            if !symbol.is_empty() && symbol.starts_with(&partial_word) {
                                let kind = match result.result_type {
                                    lens_search_engine::SearchResultType::Function => {
                                        CompletionItemKind::FUNCTION
                                    }
                                    lens_search_engine::SearchResultType::Class => {
                                        CompletionItemKind::CLASS
                                    }
                                    lens_search_engine::SearchResultType::Variable => {
                                        CompletionItemKind::VARIABLE
                                    }
                                    _ => CompletionItemKind::TEXT,
                                };

                                completion_items.push(CompletionItem {
                                    label: symbol.clone(),
                                    kind: Some(kind),
                                    detail: Some(format!(
                                        "{}:{}",
                                        result.file_path, result.line_number
                                    )),
                                    documentation: Some(Documentation::String(
                                        result.content.clone(),
                                    )),
                                    insert_text: Some(symbol),
                                    ..Default::default()
                                });
                            }
                        }

                        if !completion_items.is_empty() {
                            return Ok(Some(CompletionResponse::Array(completion_items)));
                        }
                    }
                    Err(e) => {
                        error!("Completion search failed: {}", e);
                    }
                }
            }
        }

        Ok(None)
    }

    async fn hover(&self, params: HoverParams) -> JsonRpcResult<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        debug!(
            "Hover request at {}:{}:{}",
            uri, position.line, position.character
        );

        // Get the document text
        if let Some(doc) = self.document_cache.get(uri) {
            match handle_hover(
                &self.search_engine,
                &doc.text,
                uri,
                position,
                self.config.max_search_results,
            )
            .await
            {
                Ok(hover_result) => return Ok(hover_result),
                Err(e) => {
                    error!("Enhanced hover failed: {}", e);
                }
            }
        }

        Ok(None)
    }

    async fn execute_command(&self, params: ExecuteCommandParams) -> JsonRpcResult<Option<Value>> {
        debug!("Executing command: {}", params.command);

        match params.command.as_str() {
            "lens.search" => {
                if let Some(query) = params.arguments.first().and_then(|v| v.as_str()) {
                    match self.search_text_in_workspace(query).await {
                        Ok(locations) => {
                            // Send results to client
                            self.client
                                .log_message(
                                    MessageType::INFO,
                                    format!("Found {} results for '{}'", locations.len(), query),
                                )
                                .await;

                            return Ok(Some(serde_json::to_value(locations).unwrap_or_default()));
                        }
                        Err(e) => {
                            self.client
                                .log_message(MessageType::ERROR, format!("Search failed: {}", e))
                                .await;
                        }
                    }
                }
            }
            "lens.reindex" => {
                // Re-index the workspace
                let workspace = self.workspace.read().await;
                for folder in &workspace.folders {
                    if let Ok(path) = folder.uri.to_file_path() {
                        let search_engine = self.search_engine.clone();
                        let client = self.client.clone();
                        let path_clone = path.clone();

                        tokio::spawn(async move {
                            client
                                .log_message(MessageType::INFO, "Re-indexing workspace...")
                                .await;

                            match search_engine.index_directory(&path_clone).await {
                                Ok(stats) => {
                                    client
                                        .log_message(
                                            MessageType::INFO,
                                            format!("Re-indexed {} files", stats.files_indexed),
                                        )
                                        .await;
                                }
                                Err(e) => {
                                    client
                                        .log_message(
                                            MessageType::ERROR,
                                            format!("Re-indexing failed: {}", e),
                                        )
                                        .await;
                                }
                            }
                        });
                    }
                }
            }
            _ => {
                self.client
                    .log_message(
                        MessageType::WARNING,
                        format!("Unknown command: {}", params.command),
                    )
                    .await;
            }
        }

        Ok(None)
    }
}

/// Start LSP server over TCP
pub async fn start_lsp_tcp_server(
    search_engine: Arc<SearchEngine>,
    config: LspServerConfig,
    port: u16,
) -> Result<()> {
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse()?;
    let listener = TcpListener::bind(&addr).await?;

    info!("LSP server listening on {}", addr);

    let config = Arc::new(config);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        info!("LSP client connected from {}", peer_addr);

        let search_engine = search_engine.clone();
        let server_config = config.clone();

        tokio::spawn(async move {
            let (read, write) = tokio::io::split(stream);
            let engine = search_engine.clone();
            let config = server_config.as_ref().clone();
            let (service, socket) = LspService::build(move |client| {
                LensLspServer::with_config(client, engine.clone(), config.clone())
            })
            .finish();

            Server::new(read, write, socket).serve(service).await;

            info!("LSP client {} disconnected", peer_addr);
        });
    }
}

/// Extract word at the given position from text
fn extract_word_at_position(text: &str, position: Position) -> String {
    let lines: Vec<&str> = text.lines().collect();

    if position.line as usize >= lines.len() {
        return String::new();
    }

    let line = lines[position.line as usize];
    let char_pos = position.character as usize;

    if char_pos >= line.len() {
        return String::new();
    }

    // Find word boundaries
    let chars: Vec<char> = line.chars().collect();
    let mut start = char_pos;
    let mut end = char_pos;

    // Find start of word
    while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
        start -= 1;
    }

    // Find end of word
    while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
        end += 1;
    }

    chars[start..end].iter().collect()
}

/// Extract partial word for completion
fn extract_partial_word_at_position(text: &str, position: Position) -> String {
    let lines: Vec<&str> = text.lines().collect();

    if position.line as usize >= lines.len() {
        return String::new();
    }

    let line = lines[position.line as usize];
    let char_pos = position.character as usize;

    if char_pos > line.len() {
        return String::new();
    }

    // Find start of partial word
    let chars: Vec<char> = line.chars().collect();
    let mut start = char_pos;

    while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
        start -= 1;
    }

    chars[start..char_pos].iter().collect()
}

/// Extract symbol name from code content
fn extract_symbol_from_content(content: &str) -> String {
    // Simple extraction - look for identifiers
    let words: Vec<&str> = content
        .split_whitespace()
        .filter(|word| {
            !word.is_empty()
                && word.chars().next().unwrap_or('_').is_alphabetic()
                && word.chars().all(|c| c.is_alphanumeric() || c == '_')
        })
        .collect();

    if let Some(first_word) = words.first() {
        first_word.to_string()
    } else {
        String::new()
    }
}

/// Create and start the LSP server over stdio
pub async fn start_lsp_server(
    search_engine: Arc<SearchEngine>,
    config: LspServerConfig,
) -> Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let engine = search_engine.clone();
    let server_config = config.clone();
    let exclude_count = config.workspace_exclude_patterns.len();
    let (service, socket) = LspService::build(move |client| {
        LensLspServer::with_config(client, engine.clone(), server_config.clone())
    })
    .finish();

    info!(
        "Starting Lens LSP Server (exclude patterns: {})",
        exclude_count
    );
    Server::new(stdin, stdout, socket).serve(service).await;

    Ok(())
}
