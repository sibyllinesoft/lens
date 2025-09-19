//! Language detection and parsing using Tree-sitter
//!
//! This module provides real language detection and parsing capabilities
//! to replace any simulation code for language-aware indexing.

use anyhow::{anyhow, Result};
use lens_common::{ParsedContent, ProgrammingLanguage};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::Parser;

/// Real language detector using Tree-sitter parsers
pub struct LanguageDetector {
    languages: HashMap<ProgrammingLanguage, tree_sitter::Language>,
}

impl LanguageDetector {
    /// Create a new language detector with Tree-sitter parsers
    pub fn new() -> Result<Self> {
        let mut languages = HashMap::new();

        // Store languages for supported parsers
        languages.insert(ProgrammingLanguage::Rust, tree_sitter_rust::LANGUAGE.into());
        languages.insert(
            ProgrammingLanguage::Python,
            tree_sitter_python::LANGUAGE.into(),
        );
        languages.insert(
            ProgrammingLanguage::TypeScript,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        );
        languages.insert(
            ProgrammingLanguage::JavaScript,
            tree_sitter_javascript::LANGUAGE.into(),
        );
        languages.insert(ProgrammingLanguage::Go, tree_sitter_go::LANGUAGE.into());

        Ok(Self { languages })
    }

    /// Detect language from file extension and content
    pub async fn detect_language<P: AsRef<Path>>(
        &self,
        path: P,
        content: &str,
    ) -> Result<ProgrammingLanguage> {
        // First try file extension
        if let Some(extension) = path.as_ref().extension() {
            if let Some(ext_str) = extension.to_str() {
                let language = ProgrammingLanguage::from_extension(ext_str);
                if language != ProgrammingLanguage::Unknown {
                    return Ok(language);
                }
            }
        }

        // Fallback to content-based detection
        self.detect_language_from_content(content).await
    }

    /// Detect language from content patterns
    async fn detect_language_from_content(&self, content: &str) -> Result<ProgrammingLanguage> {
        let lines: Vec<&str> = content.lines().take(50).collect(); // Check first 50 lines

        // Rust patterns
        if lines.iter().any(|line| {
            line.contains("fn ")
                || line.contains("impl ")
                || line.contains("struct ")
                || line.contains("enum ")
                || line.contains("use ")
                || line.contains("pub fn")
                || line.contains("let mut")
                || line.contains("println!")
        }) {
            return Ok(ProgrammingLanguage::Rust);
        }

        // Python patterns
        if lines.iter().any(|line| {
            line.contains("def ")
                || line.contains("class ")
                || line.contains("import ")
                || line.contains("from ")
                || line.contains("if __name__ == '__main__':")
        }) {
            return Ok(ProgrammingLanguage::Python);
        }

        // TypeScript patterns
        if lines.iter().any(|line| {
            line.contains("interface ")
                || line.contains("type ")
                || line.contains(": string")
                || line.contains(": number")
                || line.contains("export ")
        }) {
            return Ok(ProgrammingLanguage::TypeScript);
        }

        // JavaScript patterns
        if lines.iter().any(|line| {
            line.contains("function ")
                || line.contains("const ")
                || line.contains("require(")
                || line.contains("module.exports")
        }) {
            return Ok(ProgrammingLanguage::JavaScript);
        }

        // Go patterns
        if lines.iter().any(|line| {
            line.contains("package ")
                || line.contains("func ")
                || line.contains("import (")
                || line.contains("type ") && line.contains("struct")
        }) {
            return Ok(ProgrammingLanguage::Go);
        }

        Ok(ProgrammingLanguage::Unknown)
    }

    /// Parse content with Tree-sitter to extract symbols and structures
    pub async fn parse_content(
        &self,
        content: &str,
        language: &ProgrammingLanguage,
    ) -> Result<ParsedContent> {
        let mut parsed = ParsedContent {
            language: language.clone(),
            functions: HashMap::new(),
            classes: HashMap::new(),
            imports: Vec::new(),
            symbols: HashMap::new(),
        };

        // Use Tree-sitter for supported languages
        if self.languages.contains_key(language) {
            self.extract_symbols_tree_sitter(content, language, &mut parsed)
                .await?;
        } else {
            // Fallback to regex for unsupported languages
            self.extract_symbols_regex(content, &mut parsed).await?;
        }

        Ok(parsed)
    }

    /// Extract symbols using Tree-sitter AST parsing
    async fn extract_symbols_tree_sitter(
        &self,
        content: &str,
        language: &ProgrammingLanguage,
        parsed: &mut ParsedContent,
    ) -> Result<()> {
        let ts_language = self
            .languages
            .get(language)
            .ok_or_else(|| anyhow!("No language available for: {:?}", language))?;

        // Create a parser and set the language
        let mut parser = Parser::new();
        parser.set_language(ts_language)?;

        // Parse the content into an AST
        let tree = parser
            .parse(content, None)
            .ok_or_else(|| anyhow!("Failed to parse content with Tree-sitter"))?;

        let root_node = tree.root_node();

        // Walk the AST and extract symbols
        self.walk_tree_node(&root_node, content, language, parsed)
            .await?;

        Ok(())
    }

    /// Recursively walk Tree-sitter AST nodes to extract symbols
    async fn walk_tree_node(
        &self,
        node: &tree_sitter::Node<'_>,
        source: &str,
        language: &ProgrammingLanguage,
        parsed: &mut ParsedContent,
    ) -> Result<()> {
        let node_type = node.kind();
        let start_point = node.start_position();
        let line_number = start_point.row;

        // Extract symbols based on node type and language
        match language {
            ProgrammingLanguage::Rust => match node_type {
                "function_item" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let function_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !function_name.is_empty() {
                            parsed.functions.insert(line_number, function_name.clone());
                            self.add_symbol_to_line(parsed, line_number, function_name);
                        }
                    }
                }
                "struct_item" | "enum_item" | "union_item" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let class_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !class_name.is_empty() {
                            parsed.classes.insert(line_number, class_name.clone());
                            self.add_symbol_to_line(parsed, line_number, class_name);
                        }
                    }
                }
                "use_declaration" => {
                    parsed.imports.push(line_number);
                }
                "identifier" => {
                    let identifier = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    if !identifier.is_empty() && !self.is_keyword(&identifier, language) {
                        self.add_symbol_to_line(parsed, line_number, identifier);
                    }
                }
                _ => {}
            },
            ProgrammingLanguage::Python => match node_type {
                "function_definition" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let function_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !function_name.is_empty() {
                            parsed.functions.insert(line_number, function_name.clone());
                            self.add_symbol_to_line(parsed, line_number, function_name);
                        }
                    }
                }
                "class_definition" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let class_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !class_name.is_empty() {
                            parsed.classes.insert(line_number, class_name.clone());
                            self.add_symbol_to_line(parsed, line_number, class_name);
                        }
                    }
                }
                "import_statement" | "import_from_statement" => {
                    parsed.imports.push(line_number);
                }
                "identifier" => {
                    let identifier = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    if !identifier.is_empty() && !self.is_keyword(&identifier, language) {
                        self.add_symbol_to_line(parsed, line_number, identifier);
                    }
                }
                _ => {}
            },
            ProgrammingLanguage::TypeScript => match node_type {
                "function_declaration" | "method_definition" | "function_expression" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let function_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !function_name.is_empty() {
                            parsed.functions.insert(line_number, function_name.clone());
                            self.add_symbol_to_line(parsed, line_number, function_name);
                        }
                    }
                }
                "class_declaration" | "interface_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let class_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !class_name.is_empty() {
                            parsed.classes.insert(line_number, class_name.clone());
                            self.add_symbol_to_line(parsed, line_number, class_name);
                        }
                    }
                }
                "import_statement" => {
                    parsed.imports.push(line_number);
                }
                "identifier" => {
                    let identifier = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    if !identifier.is_empty() && !self.is_keyword(&identifier, language) {
                        self.add_symbol_to_line(parsed, line_number, identifier);
                    }
                }
                _ => {}
            },
            ProgrammingLanguage::JavaScript => match node_type {
                "function_declaration" | "method_definition" | "function_expression" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let function_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !function_name.is_empty() {
                            parsed.functions.insert(line_number, function_name.clone());
                            self.add_symbol_to_line(parsed, line_number, function_name);
                        }
                    }
                }
                "class_declaration" => {
                    if let Some(name_node) = node.child_by_field_name("name") {
                        let class_name = name_node
                            .utf8_text(source.as_bytes())
                            .unwrap_or("")
                            .to_string();
                        if !class_name.is_empty() {
                            parsed.classes.insert(line_number, class_name.clone());
                            self.add_symbol_to_line(parsed, line_number, class_name);
                        }
                    }
                }
                "import_statement" => {
                    parsed.imports.push(line_number);
                }
                "identifier" => {
                    let identifier = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    if !identifier.is_empty() && !self.is_keyword(&identifier, language) {
                        self.add_symbol_to_line(parsed, line_number, identifier);
                    }
                }
                _ => {}
            },
            ProgrammingLanguage::Go => {
                match node_type {
                    "function_declaration" | "method_declaration" => {
                        if let Some(name_node) = node.child_by_field_name("name") {
                            let function_name = name_node
                                .utf8_text(source.as_bytes())
                                .unwrap_or("")
                                .to_string();
                            if !function_name.is_empty() {
                                parsed.functions.insert(line_number, function_name.clone());
                                self.add_symbol_to_line(parsed, line_number, function_name);
                            }
                        }
                    }
                    "type_declaration" => {
                        // Look for struct types
                        if let Some(spec) = node.child_by_field_name("type_spec") {
                            if let Some(name_node) = spec.child_by_field_name("name") {
                                let type_name = name_node
                                    .utf8_text(source.as_bytes())
                                    .unwrap_or("")
                                    .to_string();
                                if !type_name.is_empty() {
                                    parsed.classes.insert(line_number, type_name.clone());
                                    self.add_symbol_to_line(parsed, line_number, type_name);
                                }
                            }
                        }
                    }
                    "import_declaration" => {
                        parsed.imports.push(line_number);
                    }
                    "identifier" => {
                        let identifier =
                            node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                        if !identifier.is_empty() && !self.is_keyword(&identifier, language) {
                            self.add_symbol_to_line(parsed, line_number, identifier);
                        }
                    }
                    _ => {}
                }
            }
            _ => {
                // For unsupported languages, still extract identifiers
                if node_type == "identifier" {
                    let identifier = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    if !identifier.is_empty() {
                        self.add_symbol_to_line(parsed, line_number, identifier);
                    }
                }
            }
        }

        // Recursively process child nodes
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                Box::pin(self.walk_tree_node(&child, source, language, parsed)).await?;
            }
        }

        Ok(())
    }

    /// Helper function to add a symbol to a specific line
    fn add_symbol_to_line(&self, parsed: &mut ParsedContent, line_number: usize, symbol: String) {
        parsed.symbols.entry(line_number).or_default().push(symbol);

        // Deduplicate symbols on this line
        if let Some(symbols) = parsed.symbols.get_mut(&line_number) {
            symbols.sort();
            symbols.dedup();
        }
    }

    /// Extract symbols using regex patterns (fallback for unsupported languages)
    async fn extract_symbols_regex(&self, content: &str, parsed: &mut ParsedContent) -> Result<()> {
        use regex::Regex;

        // Function patterns by language
        let function_regex = match parsed.language {
            ProgrammingLanguage::Rust => Regex::new(r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\(")?,
            ProgrammingLanguage::Python => Regex::new(r"^\s*def\s+(\w+)\s*\(")?,
            ProgrammingLanguage::TypeScript | ProgrammingLanguage::JavaScript => Regex::new(
                r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(|^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
            )?,
            ProgrammingLanguage::Go => {
                Regex::new(r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(")?
            }
            _ => Regex::new(r"^\s*\w+\s+(\w+)\s*\(")?, // Generic pattern
        };

        // Class patterns by language
        let class_regex = match parsed.language {
            ProgrammingLanguage::Rust => Regex::new(r"^\s*(?:pub\s+)?struct\s+(\w+)")?,
            ProgrammingLanguage::Python => Regex::new(r"^\s*class\s+(\w+)\s*(?:\(|:)")?,
            ProgrammingLanguage::TypeScript | ProgrammingLanguage::JavaScript => {
                Regex::new(r"^\s*(?:export\s+)?class\s+(\w+)")?
            }
            ProgrammingLanguage::Go => Regex::new(r"^\s*type\s+(\w+)\s+struct")?,
            _ => Regex::new(r"^\s*class\s+(\w+)")?, // Generic pattern
        };

        // Import patterns by language
        let import_regex = match parsed.language {
            ProgrammingLanguage::Rust => Regex::new(r"^\s*use\s+")?,
            ProgrammingLanguage::Python => Regex::new(r"^\s*(?:import\s+|from\s+)")?,
            ProgrammingLanguage::TypeScript | ProgrammingLanguage::JavaScript => {
                Regex::new(r"^\s*(?:import\s+|const\s+.*=\s*require\()")?
            }
            ProgrammingLanguage::Go => Regex::new(r"^\s*import\s+")?,
            _ => Regex::new(r"^\s*(?:import|#include)")?, // Generic pattern
        };

        let identifier_regex = Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")?;

        // Extract symbols from each line
        for (line_number, line) in content.lines().enumerate() {
            let mut symbols = Vec::new();

            // Check for functions
            if let Some(captures) = function_regex.captures(line) {
                if let Some(function_name) = captures.get(1).or_else(|| captures.get(2)) {
                    parsed
                        .functions
                        .insert(line_number, function_name.as_str().to_string());
                    symbols.push(function_name.as_str().to_string());
                }
            }

            // Check for classes/structs
            if let Some(captures) = class_regex.captures(line) {
                if let Some(class_name) = captures.get(1) {
                    parsed
                        .classes
                        .insert(line_number, class_name.as_str().to_string());
                    symbols.push(class_name.as_str().to_string());
                }
            }

            // Check for imports
            if import_regex.is_match(line) {
                parsed.imports.push(line_number);
            }

            // Extract general identifiers
            for capture in identifier_regex.captures_iter(line) {
                if let Some(identifier) = capture.get(0) {
                    let id = identifier.as_str();
                    // Filter out common keywords
                    if !self.is_keyword(id, &parsed.language) {
                        symbols.push(id.to_string());
                    }
                }
            }

            if !symbols.is_empty() {
                // Deduplicate symbols
                symbols.sort();
                symbols.dedup();
                parsed.symbols.insert(line_number, symbols);
            }
        }

        Ok(())
    }

    /// Check if a word is a language keyword
    fn is_keyword(&self, word: &str, language: &ProgrammingLanguage) -> bool {
        let keywords = match language {
            ProgrammingLanguage::Rust => vec![
                "fn", "let", "mut", "const", "static", "if", "else", "match", "for", "while",
                "loop", "break", "continue", "return", "pub", "use", "mod", "struct", "enum",
                "impl", "trait", "where", "type", "async", "await", "move", "ref", "in", "as",
                "crate", "super", "self",
            ],
            ProgrammingLanguage::Python => vec![
                "def", "class", "if", "else", "elif", "for", "while", "return", "import", "from",
                "try", "except", "finally", "with", "as", "pass", "break", "continue", "and", "or",
                "not", "in", "is", "lambda", "yield", "global", "nonlocal", "assert", "del",
                "raise",
            ],
            ProgrammingLanguage::TypeScript | ProgrammingLanguage::JavaScript => vec![
                "function",
                "const",
                "let",
                "var",
                "if",
                "else",
                "for",
                "while",
                "return",
                "import",
                "export",
                "class",
                "interface",
                "type",
                "extends",
                "implements",
                "public",
                "private",
                "protected",
                "static",
                "async",
                "await",
                "try",
                "catch",
                "finally",
                "throw",
                "new",
            ],
            ProgrammingLanguage::Go => vec![
                "func",
                "var",
                "const",
                "if",
                "else",
                "for",
                "range",
                "return",
                "import",
                "package",
                "type",
                "struct",
                "interface",
                "map",
                "chan",
                "go",
                "defer",
                "select",
                "switch",
                "case",
                "default",
                "fallthrough",
                "break",
                "continue",
                "goto",
            ],
            _ => vec![], // No keywords for unknown languages
        };

        keywords.contains(&word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_language_detection() {
        let detector = LanguageDetector::new().unwrap();

        // Test Rust detection
        let rust_content = "fn main() {\n    println!(\"Hello, world!\");\n}";
        let language = detector
            .detect_language_from_content(rust_content)
            .await
            .unwrap();
        assert_eq!(language, ProgrammingLanguage::Rust);

        // Test Python detection
        let python_content = "def hello():\n    print(\"Hello, world!\")\n";
        let language = detector
            .detect_language_from_content(python_content)
            .await
            .unwrap();
        assert_eq!(language, ProgrammingLanguage::Python);
    }

    #[tokio::test]
    async fn test_symbol_extraction() {
        let detector = LanguageDetector::new().unwrap();
        let rust_content = "fn hello_world() {\n    println!(\"Hello, world!\");\n}";

        let parsed = detector
            .parse_content(rust_content, &ProgrammingLanguage::Rust)
            .await
            .unwrap();

        assert!(parsed.functions.contains_key(&0));
        assert_eq!(parsed.functions.get(&0), Some(&"hello_world".to_string()));
    }
}
