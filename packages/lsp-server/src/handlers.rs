//! LSP request handlers
//!
//! This module contains real implementations of LSP request handlers,
//! providing actual search and navigation functionality.

use anyhow::Result;
use lens_common::ProgrammingLanguage;
use lens_search_engine::{QueryBuilder, SearchEngine};
use lsp_types::*;
use std::path::Path;
use tracing::{debug, info};

/// Handler for workspace symbol requests
pub async fn handle_workspace_symbol(
    search_engine: &SearchEngine,
    params: WorkspaceSymbolParams,
    max_results: usize,
) -> Result<Vec<SymbolInformation>> {
    debug!("Handling workspace symbol request: {}", params.query);

    if params.query.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Build symbol search query
    let search_query = QueryBuilder::new(&params.query)
        .symbol()
        .limit(max_results)
        .build();

    // Execute search
    let results = search_engine.search(&search_query).await?;

    // Convert to LSP symbols
    let mut symbols = Vec::new();
    for result in results.results {
        if let Ok(uri) = Url::from_file_path(&result.file_path) {
            let kind = determine_symbol_kind(&result.content, &result.result_type);
            let file_path = Path::new(&result.file_path);

            // Use enhanced Tree-sitter-based symbol name extraction
            let symbol_name = extract_symbol_name_enhanced(
                search_engine,
                &result.content,
                file_path,
                result.line_number,
            )
            .await
            .unwrap_or_else(|_| extract_symbol_name(&result.content));

            if !symbol_name.is_empty() {
                let location = Location {
                    uri,
                    range: Range {
                        start: Position {
                            line: result.line_number.saturating_sub(1),
                            character: result.column.saturating_sub(1),
                        },
                        end: Position {
                            line: result.line_number.saturating_sub(1),
                            character: result.column.saturating_sub(1) + symbol_name.len() as u32,
                        },
                    },
                };

                #[allow(deprecated)]
                symbols.push(SymbolInformation {
                    name: symbol_name,
                    kind,
                    tags: None,
                    deprecated: None,
                    location,
                    container_name: Some(extract_file_name(&result.file_path)),
                });
            }
        }
    }

    info!(
        "Found {} workspace symbols for query '{}'",
        symbols.len(),
        params.query
    );
    Ok(symbols)
}

/// Handler for go-to-definition requests
pub async fn handle_goto_definition(
    search_engine: &SearchEngine,
    document_text: &str,
    document_uri: &Url,
    position: Position,
    max_results: usize,
) -> Result<Vec<Location>> {
    debug!(
        "Handling goto definition at {}:{}",
        position.line, position.character
    );

    // Extract word at position using enhanced Tree-sitter extraction
    let file_path = document_uri
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URI"))?;
    let word =
        extract_word_at_position_enhanced(search_engine, document_text, &file_path, position)
            .await?;

    if word.is_empty() {
        return Ok(Vec::new());
    }

    debug!("Looking for definition of: {}", word);

    // Search for symbol definitions
    let search_query = QueryBuilder::new(&word).symbol().limit(max_results).build();

    let results = search_engine.search(&search_query).await?;

    // Filter for likely definitions and convert to locations using enhanced detection
    let mut locations = Vec::new();
    for result in results.results {
        let file_path = Path::new(&result.file_path);

        // Use enhanced Tree-sitter-based definition detection
        let is_definition = is_likely_definition_enhanced(
            search_engine,
            &result.content,
            file_path,
            result.line_number,
            &word,
        )
        .await
        .unwrap_or_else(|_| is_likely_definition(&result.content, &word));

        if is_definition {
            if let Ok(uri) = Url::from_file_path(&result.file_path) {
                let range = create_range_for_symbol(&result.content, &word, result.line_number);
                locations.push(Location { uri, range });
            }
        }
    }

    info!("Found {} definitions for '{}'", locations.len(), word);
    Ok(locations)
}

/// Handler for find references requests
pub async fn handle_find_references(
    search_engine: &SearchEngine,
    document_text: &str,
    position: Position,
    include_declaration: bool,
    max_results: usize,
) -> Result<Vec<Location>> {
    debug!(
        "Handling find references at {}:{}",
        position.line, position.character
    );

    // Extract word at position
    let word = extract_word_at_position(document_text, position);
    if word.is_empty() {
        return Ok(Vec::new());
    }

    debug!("Looking for references to: {}", word);

    // Search for all occurrences
    let search_query = QueryBuilder::new(&word).limit(max_results).build();

    let results = search_engine.search(&search_query).await?;

    // Convert to locations, filtering by include_declaration flag
    let mut locations = Vec::new();
    for result in results.results {
        let is_declaration = is_likely_definition(&result.content, &word);

        if include_declaration || !is_declaration {
            if let Ok(uri) = Url::from_file_path(&result.file_path) {
                let range = create_range_for_symbol(&result.content, &word, result.line_number);
                locations.push(Location { uri, range });
            }
        }
    }

    info!("Found {} references to '{}'", locations.len(), word);
    Ok(locations)
}

/// Handler for completion requests
pub async fn handle_completion(
    search_engine: &SearchEngine,
    document_text: &str,
    position: Position,
    language_id: &str,
    max_results: usize,
) -> Result<Vec<CompletionItem>> {
    debug!(
        "Handling completion at {}:{} for {}",
        position.line, position.character, language_id
    );

    // Extract partial word for completion
    let partial_word = extract_partial_word_at_position(document_text, position);
    if partial_word.len() < 2 {
        return Ok(Vec::new()); // Only complete after 2+ characters
    }

    debug!("Completing partial word: {}", partial_word);

    // Build fuzzy search query
    let mut query_builder = QueryBuilder::new(&partial_word)
        .fuzzy()
        .limit(max_results * 2);

    // Add language filter if possible
    if let Some(lang) = map_language_id_to_programming_language(language_id) {
        query_builder = query_builder.language(lang);
    }

    let search_query = query_builder.build();
    let results = search_engine.search(&search_query).await?;

    // Extract completion items
    let mut completion_items = Vec::new();
    let mut seen_symbols = std::collections::HashSet::new();

    for result in results.results.iter().take(max_results) {
        let symbols = extract_completion_symbols(&result.content, &partial_word);

        for symbol in symbols {
            if seen_symbols.insert(symbol.clone()) {
                let kind = determine_completion_kind(&result.content, &symbol);
                let detail = format!(
                    "{}:{}",
                    extract_file_name(&result.file_path),
                    result.line_number
                );

                completion_items.push(CompletionItem {
                    label: symbol.clone(),
                    kind: Some(kind),
                    detail: Some(detail),
                    documentation: Some(Documentation::String(result.content.trim().to_string())),
                    insert_text: Some(symbol),
                    sort_text: Some(format!("{:04}", completion_items.len())),
                    filter_text: Some(partial_word.clone()),
                    ..Default::default()
                });

                if completion_items.len() >= max_results {
                    break;
                }
            }
        }

        if completion_items.len() >= max_results {
            break;
        }
    }

    info!(
        "Generated {} completion items for '{}'",
        completion_items.len(),
        partial_word
    );
    Ok(completion_items)
}

/// Handler for hover requests
pub async fn handle_hover(
    search_engine: &SearchEngine,
    document_text: &str,
    document_uri: &Url,
    position: Position,
    max_results: usize,
) -> Result<Option<Hover>> {
    debug!("Handling hover at {}:{}", position.line, position.character);

    // Extract word at position using enhanced Tree-sitter extraction
    let file_path = document_uri
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URI"))?;
    let word =
        extract_word_at_position_enhanced(search_engine, document_text, &file_path, position)
            .await?;

    if word.is_empty() {
        return Ok(None);
    }

    debug!("Looking for hover info for: {}", word);

    // Search for symbol information
    let search_query = QueryBuilder::new(&word).symbol().limit(max_results).build();

    let results = search_engine.search(&search_query).await?;

    if results.results.is_empty() {
        return Ok(None);
    }

    // Create hover content from search results
    let mut hover_lines = Vec::new();
    hover_lines.push(format!("**{}**", word));
    hover_lines.push(String::new());

    // Add definitions using enhanced Tree-sitter-based detection
    for result in results.results.iter().take(3) {
        let file_path = Path::new(&result.file_path);

        // Use enhanced Tree-sitter-based definition detection
        let is_definition = is_likely_definition_enhanced(
            search_engine,
            &result.content,
            file_path,
            result.line_number,
            &word,
        )
        .await
        .unwrap_or_else(|_| is_likely_definition(&result.content, &word));

        if is_definition {
            let file_name = extract_file_name(&result.file_path);
            hover_lines.push(format!("*{}:{}*", file_name, result.line_number));
            hover_lines.push(format!("```\n{}\n```", result.content.trim()));
            hover_lines.push(String::new());
        }
    }

    // Add reference count
    let ref_count = results.results.len();
    if ref_count > 1 {
        hover_lines.push(format!("*{} references found*", ref_count));
    }

    let hover_content = hover_lines.join("\n");

    let range = Range {
        start: Position {
            line: position.line,
            character: position.character.saturating_sub(word.len() as u32 / 2),
        },
        end: Position {
            line: position.line,
            character: position.character + word.len() as u32 / 2,
        },
    };

    Ok(Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: hover_content,
        }),
        range: Some(range),
    }))
}

/// Utility functions for LSP handlers

/// Extract word at the given position
fn extract_word_at_position(text: &str, position: Position) -> String {
    let lines: Vec<&str> = text.lines().collect();

    if position.line as usize >= lines.len() {
        return String::new();
    }

    let line = lines[position.line as usize];
    let char_pos = position.character as usize;

    if char_pos > line.len() {
        return String::new();
    }

    // Find word boundaries
    let chars: Vec<char> = line.chars().collect();
    let mut start = char_pos;
    let mut end = char_pos;

    // Find start of word
    while start > 0 && is_word_char(chars.get(start - 1).copied().unwrap_or(' ')) {
        start -= 1;
    }

    // Find end of word
    while end < chars.len() && is_word_char(chars.get(end).copied().unwrap_or(' ')) {
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

    while start > 0 && is_word_char(chars.get(start - 1).copied().unwrap_or(' ')) {
        start -= 1;
    }

    chars[start..char_pos].iter().collect()
}

/// Check if character is part of a word
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

/// Determine symbol kind from content and result type
fn determine_symbol_kind(
    content: &str,
    result_type: &lens_search_engine::SearchResultType,
) -> SymbolKind {
    match result_type {
        lens_search_engine::SearchResultType::Function => SymbolKind::FUNCTION,
        lens_search_engine::SearchResultType::Class => SymbolKind::CLASS,
        lens_search_engine::SearchResultType::Variable => SymbolKind::VARIABLE,
        lens_search_engine::SearchResultType::Import => SymbolKind::MODULE,
        _ => {
            // Try to infer from content
            if content.contains("fn ") || content.contains("function ") || content.contains("def ")
            {
                SymbolKind::FUNCTION
            } else if content.contains("class ")
                || content.contains("struct ")
                || content.contains("interface ")
            {
                SymbolKind::CLASS
            } else if content.contains("const ")
                || content.contains("let ")
                || content.contains("var ")
            {
                SymbolKind::VARIABLE
            } else {
                SymbolKind::VARIABLE
            }
        }
    }
}

/// Extract symbol name from content
fn extract_symbol_name(content: &str) -> String {
    // Look for common patterns to extract symbol names
    let words: Vec<&str> = content.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        if matches!(
            *word,
            "fn" | "function" | "def" | "class" | "struct" | "interface" | "const" | "let" | "var"
        ) {
            if let Some(next_word) = words.get(i + 1) {
                // Remove common suffixes like parentheses, colons, etc.
                let clean_name =
                    next_word.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_');

                if !clean_name.is_empty()
                    && clean_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                {
                    return clean_name.to_string();
                }
            }
        }
    }

    // Fallback: try to find the first identifier
    for word in words {
        if !word.is_empty()
            && word.chars().next().unwrap_or('_').is_alphabetic()
            && word.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            return word.to_string();
        }
    }

    String::new()
}

/// Check if content is likely a definition
fn is_likely_definition(content: &str, symbol: &str) -> bool {
    let content_lower = content.to_lowercase();
    let symbol_lower = symbol.to_lowercase();

    // Check for definition keywords before the symbol
    let def_keywords = [
        "fn ",
        "function ",
        "def ",
        "class ",
        "struct ",
        "interface ",
        "const ",
        "let ",
        "var ",
    ];

    for keyword in def_keywords {
        if let Some(keyword_pos) = content_lower.find(keyword) {
            if let Some(symbol_pos) = content_lower.find(&symbol_lower) {
                if symbol_pos > keyword_pos && symbol_pos - keyword_pos < 50 {
                    return true;
                }
            }
        }
    }

    false
}

/// Enhanced Tree-sitter-based symbol extraction at position
async fn extract_word_at_position_enhanced(
    search_engine: &SearchEngine,
    document_text: &str,
    file_path: &Path,
    position: Position,
) -> Result<String> {
    let line_number = position.line as usize;

    // Try Tree-sitter first
    if let Ok(Some(symbol)) = search_engine
        .get_symbol_at_position(document_text, file_path, line_number)
        .await
    {
        return Ok(symbol);
    }

    // Fallback to simple extraction
    Ok(extract_word_at_position(document_text, position))
}

/// Enhanced Tree-sitter-based definition detection
async fn is_likely_definition_enhanced(
    search_engine: &SearchEngine,
    content: &str,
    file_path: &Path,
    line_number: u32,
    symbol: &str,
) -> Result<bool> {
    let line_index = line_number.saturating_sub(1) as usize;

    // Try Tree-sitter first
    if let Ok(is_def) = search_engine
        .is_definition_at_line(content, file_path, line_index, symbol)
        .await
    {
        return Ok(is_def);
    }

    // Fallback to simple heuristic
    Ok(is_likely_definition(content, symbol))
}

/// Enhanced Tree-sitter-based symbol name extraction
async fn extract_symbol_name_enhanced(
    search_engine: &SearchEngine,
    content: &str,
    file_path: &Path,
    line_number: u32,
) -> Result<String> {
    let line_index = line_number.saturating_sub(1) as usize;

    // Try Tree-sitter first
    if let Ok(Some(symbol)) = search_engine
        .get_symbol_at_position(content, file_path, line_index)
        .await
    {
        return Ok(symbol);
    }

    // Fallback to simple extraction
    Ok(extract_symbol_name(content))
}

/// Create a range for a symbol in content
fn create_range_for_symbol(content: &str, symbol: &str, line_number: u32) -> Range {
    let symbol_pos = content.find(symbol).unwrap_or(0);
    let start_char = symbol_pos as u32;
    let end_char = start_char + symbol.len() as u32;

    Range {
        start: Position {
            line: line_number.saturating_sub(1),
            character: start_char,
        },
        end: Position {
            line: line_number.saturating_sub(1),
            character: end_char,
        },
    }
}

/// Extract completion symbols from content
fn extract_completion_symbols(content: &str, partial_word: &str) -> Vec<String> {
    let mut symbols = Vec::new();

    // Split content into words and find matches
    for word in content.split_whitespace() {
        let clean_word = word
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
            .to_string();

        if clean_word.len() >= partial_word.len()
            && clean_word.starts_with(partial_word)
            && clean_word.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            symbols.push(clean_word);
        }
    }

    symbols.sort();
    symbols.dedup();
    symbols
}

/// Determine completion item kind
fn determine_completion_kind(content: &str, _symbol: &str) -> CompletionItemKind {
    let content_around_symbol = content.to_lowercase();

    if content_around_symbol.contains("fn ")
        || content_around_symbol.contains("function ")
        || content_around_symbol.contains("def ")
    {
        CompletionItemKind::FUNCTION
    } else if content_around_symbol.contains("class ") || content_around_symbol.contains("struct ")
    {
        CompletionItemKind::CLASS
    } else if content_around_symbol.contains("const ") {
        CompletionItemKind::CONSTANT
    } else if content_around_symbol.contains("let ") || content_around_symbol.contains("var ") {
        CompletionItemKind::VARIABLE
    } else {
        CompletionItemKind::TEXT
    }
}

/// Extract file name from path
fn extract_file_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_string()
}

/// Map LSP language ID to programming language
fn map_language_id_to_programming_language(language_id: &str) -> Option<ProgrammingLanguage> {
    match language_id {
        "rust" => Some(ProgrammingLanguage::Rust),
        "python" => Some(ProgrammingLanguage::Python),
        "typescript" => Some(ProgrammingLanguage::TypeScript),
        "javascript" => Some(ProgrammingLanguage::JavaScript),
        "go" => Some(ProgrammingLanguage::Go),
        "java" => Some(ProgrammingLanguage::Java),
        "cpp" | "c++" => Some(ProgrammingLanguage::Cpp),
        "c" => Some(ProgrammingLanguage::C),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_word_at_position() {
        let text = "fn hello_world() {";
        let position = Position {
            line: 0,
            character: 5,
        };
        let word = extract_word_at_position(text, position);
        assert_eq!(word, "hello_world");
    }

    #[test]
    fn test_extract_partial_word() {
        let text = "fn hello_wor";
        let position = Position {
            line: 0,
            character: 12,
        };
        let partial = extract_partial_word_at_position(text, position);
        assert_eq!(partial, "hello_wor");
    }

    #[test]
    fn test_extract_symbol_name() {
        assert_eq!(extract_symbol_name("fn hello_world() {"), "hello_world");
        assert_eq!(extract_symbol_name("class MyClass {"), "MyClass");
        assert_eq!(extract_symbol_name("const PI = 3.14;"), "PI");
        assert_eq!(extract_symbol_name("def calculate():"), "calculate");
    }

    #[test]
    fn test_is_likely_definition() {
        assert!(is_likely_definition("fn hello_world() {", "hello_world"));
        assert!(is_likely_definition("class MyClass {", "MyClass"));
        assert!(!is_likely_definition("hello_world();", "hello_world"));
        assert!(!is_likely_definition("print(hello_world)", "hello_world"));
    }

    #[test]
    fn test_determine_symbol_kind() {
        use lens_search_engine::SearchResultType;

        assert_eq!(
            determine_symbol_kind("fn test()", &SearchResultType::Function),
            SymbolKind::FUNCTION
        );
        assert_eq!(
            determine_symbol_kind("class Test", &SearchResultType::Class),
            SymbolKind::CLASS
        );
        assert_eq!(
            determine_symbol_kind("let x = 5", &SearchResultType::Variable),
            SymbolKind::VARIABLE
        );
    }
}
