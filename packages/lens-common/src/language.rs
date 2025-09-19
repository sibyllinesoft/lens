//! Programming language detection and parsing types
//!
//! Shared types for language detection and AST parsing

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};

/// Supported programming languages
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProgrammingLanguage {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    Cpp,
    C,
    Unknown,
}

impl fmt::Display for ProgrammingLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            ProgrammingLanguage::Rust => "rust",
            ProgrammingLanguage::Python => "python",
            ProgrammingLanguage::TypeScript => "typescript",
            ProgrammingLanguage::JavaScript => "javascript",
            ProgrammingLanguage::Go => "go",
            ProgrammingLanguage::Java => "java",
            ProgrammingLanguage::Cpp => "cpp",
            ProgrammingLanguage::C => "c",
            ProgrammingLanguage::Unknown => "unknown",
        };

        f.write_str(value)
    }
}

impl ProgrammingLanguage {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "py" | "pyw" => Self::Python,
            "ts" => Self::TypeScript,
            "js" | "jsx" => Self::JavaScript,
            "go" => Self::Go,
            "java" => Self::Java,
            "cpp" | "cc" | "cxx" | "c++" => Self::Cpp,
            "c" | "h" => Self::C,
            _ => Self::Unknown,
        }
    }

    /// Get file extensions for this language
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Rust => &["rs"],
            Self::Python => &["py", "pyw"],
            Self::TypeScript => &["ts"],
            Self::JavaScript => &["js", "jsx"],
            Self::Go => &["go"],
            Self::Java => &["java"],
            Self::Cpp => &["cpp", "cc", "cxx", "c++"],
            Self::C => &["c", "h"],
            Self::Unknown => &[],
        }
    }
}

/// Parsed content with extracted symbols and structures
#[derive(Debug, Clone)]
pub struct ParsedContent {
    pub language: ProgrammingLanguage,
    pub functions: HashMap<usize, String>, // line_number -> function_name
    pub classes: HashMap<usize, String>,   // line_number -> class_name
    pub imports: Vec<usize>,               // line_numbers of import statements
    pub symbols: HashMap<usize, Vec<String>>, // line_number -> symbols
}

impl ParsedContent {
    /// Create new empty parsed content
    pub fn new(language: ProgrammingLanguage) -> Self {
        Self {
            language,
            functions: HashMap::new(),
            classes: HashMap::new(),
            imports: Vec::new(),
            symbols: HashMap::new(),
        }
    }

    /// Add a function at the given line
    pub fn add_function(&mut self, line: usize, name: String) {
        self.functions.insert(line, name.clone());
        self.symbols.entry(line).or_default().push(name);
    }

    /// Add a class at the given line
    pub fn add_class(&mut self, line: usize, name: String) {
        self.classes.insert(line, name.clone());
        self.symbols.entry(line).or_default().push(name);
    }

    /// Add an import at the given line
    pub fn add_import(&mut self, line: usize) {
        self.imports.push(line);
    }

    /// Add a symbol at the given line
    pub fn add_symbol(&mut self, line: usize, symbol: String) {
        self.symbols.entry(line).or_default().push(symbol);
    }

    /// Get all symbols at a specific line (returns a clone)
    pub fn symbols_at_line(&self, line: usize) -> Vec<String> {
        self.symbols.get(&line).cloned().unwrap_or_default()
    }

    /// Get all symbols at a specific line (returns a reference)
    pub fn symbols_at_line_ref(&self, line: usize) -> Option<&Vec<String>> {
        self.symbols.get(&line)
    }

    /// Check if a line contains a function definition
    pub fn is_function_definition(&self, line: usize) -> bool {
        self.functions.contains_key(&line)
    }

    /// Check if a line contains a class definition
    pub fn is_class_definition(&self, line: usize) -> bool {
        self.classes.contains_key(&line)
    }

    /// Check if a line contains an import
    pub fn is_import(&self, line: usize) -> bool {
        self.imports.contains(&line)
    }

    /// Get function name at a specific line
    pub fn functions_at_line(&self, line: usize) -> Option<String> {
        self.functions.get(&line).cloned()
    }

    /// Get class name at a specific line  
    pub fn classes_at_line(&self, line: usize) -> Option<String> {
        self.classes.get(&line).cloned()
    }

    /// Check if a line contains an import (alias for is_import)
    pub fn is_import_line(&self, line: usize) -> bool {
        self.imports.contains(&line)
    }
}
