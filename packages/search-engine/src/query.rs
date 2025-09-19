//! Query types and parsing for the search engine
//!
//! This module defines real query structures and parsing logic,
//! replacing any simulation or mock query handling.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use lens_common::ProgrammingLanguage;

/// Types of search queries supported
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryType {
    /// Exact text match
    Exact,
    /// Fuzzy text search with edit distance
    Fuzzy,
    /// General text search
    Text,
    /// Symbol-specific search (functions, classes, etc.)
    Symbol,
}

/// Search query with all parameters
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SearchQuery {
    /// The search text
    pub text: String,
    /// Type of query to perform
    pub query_type: QueryType,
    /// Maximum number of results to return
    pub limit: Option<usize>,
    /// Number of results to skip
    pub offset: Option<usize>,
    /// Filter by programming language
    pub language_filter: Option<ProgrammingLanguage>,
    /// Filter by file path substring/regex
    pub file_filter: Option<String>,
}

impl SearchQuery {
    /// Create a new text search query
    pub fn new_text<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            query_type: QueryType::Text,
            limit: None,
            offset: None,
            language_filter: None,
            file_filter: None,
        }
    }

    /// Create a new exact match query
    pub fn new_exact<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            query_type: QueryType::Exact,
            limit: None,
            offset: None,
            language_filter: None,
            file_filter: None,
        }
    }

    /// Create a new fuzzy search query
    pub fn new_fuzzy<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            query_type: QueryType::Fuzzy,
            limit: None,
            offset: None,
            language_filter: None,
            file_filter: None,
        }
    }

    /// Create a new symbol search query
    pub fn new_symbol<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            query_type: QueryType::Symbol,
            limit: None,
            offset: None,
            language_filter: None,
            file_filter: None,
        }
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set result offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Set language filter
    pub fn with_language(mut self, language: ProgrammingLanguage) -> Self {
        self.language_filter = Some(language);
        self
    }

    /// Set file filter
    pub fn with_file_filter<S: Into<String>>(mut self, pattern: S) -> Self {
        self.file_filter = Some(pattern.into());
        self
    }

    /// Generate a cache key for this query
    pub fn cache_key(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        format!("query_{:x}", hasher.finish())
    }

    /// Check if this query is valid
    pub fn is_valid(&self) -> bool {
        !self.text.trim().is_empty() && self.text.len() <= 1000
    }

    /// Normalize the query text for better matching
    pub fn normalized_text(&self) -> String {
        self.text.trim().to_lowercase()
    }

    /// Get the effective limit (with default)
    pub fn effective_limit(&self, default_limit: usize) -> usize {
        self.limit.unwrap_or(default_limit)
    }

    /// Get the effective offset (with default)
    pub fn effective_offset(&self, default_offset: usize) -> usize {
        self.offset.unwrap_or(default_offset)
    }
}

/// Query builder for constructing complex queries
pub struct QueryBuilder {
    text: String,
    query_type: QueryType,
    limit: Option<usize>,
    offset: Option<usize>,
    language_filter: Option<ProgrammingLanguage>,
    file_filter: Option<String>,
}

impl QueryBuilder {
    /// Start building a new query
    pub fn new<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            query_type: QueryType::Text,
            limit: None,
            offset: None,
            language_filter: None,
            file_filter: None,
        }
    }

    /// Set the query type
    pub fn query_type(mut self, query_type: QueryType) -> Self {
        self.query_type = query_type;
        self
    }

    /// Set exact match
    pub fn exact(mut self) -> Self {
        self.query_type = QueryType::Exact;
        self
    }

    /// Set fuzzy search
    pub fn fuzzy(mut self) -> Self {
        self.query_type = QueryType::Fuzzy;
        self
    }

    /// Set symbol search
    pub fn symbol(mut self) -> Self {
        self.query_type = QueryType::Symbol;
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Skip number of results
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Filter by language
    pub fn language(mut self, language: ProgrammingLanguage) -> Self {
        self.language_filter = Some(language);
        self
    }

    /// Filter by file path (substring or regex)
    pub fn file_pattern<S: Into<String>>(mut self, pattern: S) -> Self {
        self.file_filter = Some(pattern.into());
        self
    }

    /// Build the final query
    pub fn build(self) -> SearchQuery {
        SearchQuery {
            text: self.text,
            query_type: self.query_type,
            limit: self.limit,
            offset: self.offset,
            language_filter: self.language_filter,
            file_filter: self.file_filter,
        }
    }
}

/// Parse a query string into a SearchQuery
pub fn parse_query_string(query_str: &str) -> Result<SearchQuery> {
    parse_core_query(query_str)
}

/// Parse language from query string (e.g., "lang:rust hello world")
pub fn extract_language_filter(query_str: &str) -> (Option<ProgrammingLanguage>, String) {
    if let Some(lang_start) = query_str.find("lang:") {
        let after_lang = &query_str[lang_start + 5..];
        if let Some(space_pos) = after_lang.find(' ') {
            let lang_str = &after_lang[..space_pos];
            let remaining_query = format!(
                "{}{}",
                &query_str[..lang_start],
                &after_lang[space_pos + 1..]
            )
            .trim()
            .to_string();

            let language = match lang_str.to_lowercase().as_str() {
                "rust" | "rs" => Some(ProgrammingLanguage::Rust),
                "python" | "py" => Some(ProgrammingLanguage::Python),
                "typescript" | "ts" => Some(ProgrammingLanguage::TypeScript),
                "javascript" | "js" => Some(ProgrammingLanguage::JavaScript),
                "go" => Some(ProgrammingLanguage::Go),
                "java" => Some(ProgrammingLanguage::Java),
                "cpp" | "c++" => Some(ProgrammingLanguage::Cpp),
                "c" => Some(ProgrammingLanguage::C),
                _ => None,
            };

            return (language, remaining_query);
        } else {
            // Language is at the end
            let lang_str = after_lang;
            let remaining_query = query_str[..lang_start].trim().to_string();

            let language = match lang_str.to_lowercase().as_str() {
                "rust" | "rs" => Some(ProgrammingLanguage::Rust),
                "python" | "py" => Some(ProgrammingLanguage::Python),
                "typescript" | "ts" => Some(ProgrammingLanguage::TypeScript),
                "javascript" | "js" => Some(ProgrammingLanguage::JavaScript),
                "go" => Some(ProgrammingLanguage::Go),
                "java" => Some(ProgrammingLanguage::Java),
                "cpp" | "c++" => Some(ProgrammingLanguage::Cpp),
                "c" => Some(ProgrammingLanguage::C),
                _ => None,
            };

            return (language, remaining_query);
        }
    }

    (None, query_str.to_string())
}

fn extract_file_filter(query_str: &str) -> (Option<String>, String) {
    if let Some(path_start) = query_str.find("path:") {
        let after_path = &query_str[path_start + 5..];
        if after_path.is_empty() {
            return (None, query_str.to_string());
        }

        if let Some(space_pos) = after_path.find(' ') {
            let pattern = after_path[..space_pos].to_string();
            let remaining = format!(
                "{}{}",
                &query_str[..path_start],
                &after_path[space_pos + 1..]
            )
            .trim()
            .to_string();
            return (Some(pattern), remaining);
        } else {
            let pattern = after_path.trim().to_string();
            let remaining = query_str[..path_start].trim().to_string();
            return (Some(pattern), remaining);
        }
    }

    (None, query_str.to_string())
}

fn extract_numeric_filter(query_str: &str, prefix: &str) -> (Option<usize>, String) {
    if let Some(start) = query_str.find(prefix) {
        let after = &query_str[start + prefix.len()..];
        if after.is_empty() {
            return (None, query_str.to_string());
        }

        let (number_str, remaining) = if let Some(space_pos) = after.find(' ') {
            let value = after[..space_pos].trim();
            let remaining = format!("{}{}", &query_str[..start], &after[space_pos + 1..])
                .trim()
                .to_string();
            (value.to_string(), remaining)
        } else {
            let value = after.trim().to_string();
            let remaining = query_str[..start].trim().to_string();
            (value, remaining)
        };

        if let Ok(parsed) = number_str.parse::<usize>() {
            return (Some(parsed), remaining);
        } else {
            return (None, query_str.to_string());
        }
    }

    (None, query_str.to_string())
}

/// Parse a complete query with all features
pub fn parse_full_query(query_str: &str) -> Result<SearchQuery> {
    let mut working = query_str.trim().to_string();
    let (language_filter, rest) = extract_language_filter(&working);
    working = rest;

    let (file_filter, rest) = extract_file_filter(&working);
    working = rest;

    let (offset_filter, rest) = extract_numeric_filter(&working, "offset:");
    working = rest;

    let (limit_filter, rest) = extract_numeric_filter(&working, "limit:");
    working = rest;

    let mut query = parse_core_query(&working)?;

    if let Some(language) = language_filter {
        query.language_filter = Some(language);
    }
    if let Some(path_pattern) = file_filter {
        query.file_filter = Some(path_pattern);
    }
    if let Some(offset) = offset_filter {
        query.offset = Some(offset);
    }
    if let Some(limit) = limit_filter {
        query.limit = Some(limit);
    }

    Ok(query)
}

fn parse_core_query(query_str: &str) -> Result<SearchQuery> {
    let trimmed = query_str.trim();

    if trimmed.is_empty() {
        return Ok(SearchQuery::new_text(""));
    }

    if let Some(stripped) = trimmed.strip_prefix("exact:") {
        return Ok(SearchQuery::new_exact(stripped.trim()));
    }

    if let Some(stripped) = trimmed.strip_prefix("fuzzy:") {
        return Ok(SearchQuery::new_fuzzy(stripped.trim()));
    }

    if let Some(stripped) = trimmed.strip_prefix("symbol:") {
        return Ok(SearchQuery::new_symbol(stripped.trim()));
    }

    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() > 1 {
        let quoted_content = &trimmed[1..trimmed.len() - 1];
        return Ok(SearchQuery::new_exact(quoted_content));
    }

    Ok(SearchQuery::new_text(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_parsing() {
        // Test basic text query
        let query = parse_query_string("hello world").unwrap();
        assert_eq!(query.text, "hello world");
        assert_eq!(query.query_type, QueryType::Text);

        // Test exact query
        let query = parse_query_string("exact:hello world").unwrap();
        assert_eq!(query.text, "hello world");
        assert_eq!(query.query_type, QueryType::Exact);

        // Test quoted exact query
        let query = parse_query_string("\"hello world\"").unwrap();
        assert_eq!(query.text, "hello world");
        assert_eq!(query.query_type, QueryType::Exact);

        // Test fuzzy query
        let query = parse_query_string("fuzzy:hello world").unwrap();
        assert_eq!(query.text, "hello world");
        assert_eq!(query.query_type, QueryType::Fuzzy);

        // Test symbol query
        let query = parse_query_string("symbol:function_name").unwrap();
        assert_eq!(query.text, "function_name");
        assert_eq!(query.query_type, QueryType::Symbol);
    }

    #[test]
    fn test_language_extraction() {
        // Test language at beginning
        let (lang, remaining) = extract_language_filter("lang:rust hello world");
        assert_eq!(lang, Some(ProgrammingLanguage::Rust));
        assert_eq!(remaining, "hello world");

        // Test language at end
        let (lang, remaining) = extract_language_filter("hello world lang:python");
        assert_eq!(lang, Some(ProgrammingLanguage::Python));
        assert_eq!(remaining, "hello world");

        // Test no language
        let (lang, remaining) = extract_language_filter("hello world");
        assert_eq!(lang, None);
        assert_eq!(remaining, "hello world");
    }

    #[test]
    fn test_full_query_parsing() {
        let query = parse_full_query("lang:rust symbol:main").unwrap();
        assert_eq!(query.text, "main");
        assert_eq!(query.query_type, QueryType::Symbol);
        assert_eq!(query.language_filter, Some(ProgrammingLanguage::Rust));
        assert_eq!(query.file_filter, None);

        let query = parse_full_query("path:src/search exact:builder").unwrap();
        assert_eq!(query.text, "builder");
        assert_eq!(query.query_type, QueryType::Exact);
        assert_eq!(query.file_filter.as_deref(), Some("src/search"));

        let query = parse_full_query("lang:python path:tests fuzzy:runner").unwrap();
        assert_eq!(query.text, "runner");
        assert_eq!(query.query_type, QueryType::Fuzzy);
        assert_eq!(query.language_filter, Some(ProgrammingLanguage::Python));
        assert_eq!(query.file_filter.as_deref(), Some("tests"));
    }

    #[test]
    fn test_file_filter_extraction() {
        let (pattern, remaining) = extract_file_filter("path:src/lib.rs exact:query");
        assert_eq!(pattern.as_deref(), Some("src/lib.rs"));
        assert_eq!(remaining, "exact:query");

        let (pattern, remaining) = extract_file_filter("symbol:main path:src");
        assert_eq!(pattern.as_deref(), Some("src"));
        assert_eq!(remaining, "symbol:main");

        let (pattern, remaining) = extract_file_filter("no filter here");
        assert!(pattern.is_none());
        assert_eq!(remaining, "no filter here");
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new("test")
            .fuzzy()
            .limit(10)
            .language(ProgrammingLanguage::Python)
            .file_pattern("src/")
            .build();

        assert_eq!(query.text, "test");
        assert_eq!(query.query_type, QueryType::Fuzzy);
        assert_eq!(query.limit, Some(10));
        assert_eq!(query.language_filter, Some(ProgrammingLanguage::Python));
        assert_eq!(query.file_filter.as_deref(), Some("src/"));
    }

    #[test]
    fn test_numeric_filters() {
        let query = parse_full_query("limit:20 offset:5 fuzzy:runner").unwrap();
        assert_eq!(query.limit, Some(20));
        assert_eq!(query.offset, Some(5));
        assert_eq!(query.query_type, QueryType::Fuzzy);
        assert_eq!(query.text, "runner");
    }

    #[test]
    fn test_query_validation() {
        assert!(SearchQuery::new_text("valid query").is_valid());
        assert!(!SearchQuery::new_text("").is_valid());
        assert!(!SearchQuery::new_text("   ").is_valid());

        // Test very long query
        let long_text = "a".repeat(1001);
        assert!(!SearchQuery::new_text(long_text).is_valid());
    }

    #[test]
    fn test_cache_key() {
        let query1 = SearchQuery::new_text("test");
        let query2 = SearchQuery::new_text("test");
        let query3 = SearchQuery::new_text("different");

        assert_eq!(query1.cache_key(), query2.cache_key());
        assert_ne!(query1.cache_key(), query3.cache_key());
    }
}
