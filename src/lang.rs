use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Language detection and classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    CSharp,
    Cpp,
    C,
    Ruby,
    Php,
    Swift,
    Kotlin,
    Scala,
    Clojure,
    Haskell,
    Unknown,
}

impl Language {
    /// Detect language from file extension
    pub fn from_extension(extension: &str) -> Self {
        match extension.to_lowercase().as_str() {
            "rs" => Language::Rust,
            "ts" => Language::TypeScript,
            "tsx" => Language::TypeScript,
            "js" => Language::JavaScript,
            "jsx" => Language::JavaScript,
            "mjs" => Language::JavaScript,
            "py" => Language::Python,
            "pyi" => Language::Python,
            "go" => Language::Go,
            "java" => Language::Java,
            "cs" => Language::CSharp,
            "cpp" | "cxx" | "cc" => Language::Cpp,
            "c" => Language::C,
            "h" | "hpp" => Language::Cpp, // Header files default to C++
            "rb" => Language::Ruby,
            "php" => Language::Php,
            "swift" => Language::Swift,
            "kt" | "kts" => Language::Kotlin,
            "scala" | "sc" => Language::Scala,
            "clj" | "cljs" | "cljc" => Language::Clojure,
            "hs" => Language::Haskell,
            _ => Language::Unknown,
        }
    }

    /// Get file extensions for this language
    pub fn extensions(&self) -> Vec<&'static str> {
        match self {
            Language::Rust => vec!["rs"],
            Language::TypeScript => vec!["ts", "tsx"],
            Language::JavaScript => vec!["js", "jsx", "mjs"],
            Language::Python => vec!["py", "pyi"],
            Language::Go => vec!["go"],
            Language::Java => vec!["java"],
            Language::CSharp => vec!["cs"],
            Language::Cpp => vec!["cpp", "cxx", "cc", "hpp", "h"],
            Language::C => vec!["c", "h"],
            Language::Ruby => vec!["rb"],
            Language::Php => vec!["php"],
            Language::Swift => vec!["swift"],
            Language::Kotlin => vec!["kt", "kts"],
            Language::Scala => vec!["scala", "sc"],
            Language::Clojure => vec!["clj", "cljs", "cljc"],
            Language::Haskell => vec!["hs"],
            Language::Unknown => vec![],
        }
    }

    /// Get LSP server name for this language
    pub fn lsp_server(&self) -> Option<&'static str> {
        match self {
            Language::TypeScript | Language::JavaScript => Some("tsserver"),
            Language::Python => Some("pylsp"),
            Language::Rust => Some("rust-analyzer"),
            Language::Go => Some("gopls"),
            Language::Java => Some("jdtls"),
            Language::CSharp => Some("omnisharp"),
            Language::Cpp | Language::C => Some("clangd"),
            _ => None,
        }
    }

    /// Get language display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Language::Rust => "Rust",
            Language::TypeScript => "TypeScript",
            Language::JavaScript => "JavaScript",
            Language::Python => "Python",
            Language::Go => "Go",
            Language::Java => "Java",
            Language::CSharp => "C#",
            Language::Cpp => "C++",
            Language::C => "C",
            Language::Ruby => "Ruby",
            Language::Php => "PHP",
            Language::Swift => "Swift",
            Language::Kotlin => "Kotlin",
            Language::Scala => "Scala",
            Language::Clojure => "Clojure",
            Language::Haskell => "Haskell",
            Language::Unknown => "Unknown",
        }
    }

    /// Check if language supports LSP integration
    pub fn supports_lsp(&self) -> bool {
        self.lsp_server().is_some()
    }

    /// Get default search boost for this language
    pub fn search_boost(&self) -> f64 {
        match self {
            Language::Rust => 1.2,        // Boost Rust files
            Language::TypeScript => 1.1,  // Boost TypeScript files
            Language::Python => 1.1,      // Boost Python files
            Language::JavaScript => 1.0,  // Neutral
            Language::Go => 1.05,         // Slight boost
            Language::Java => 1.0,        // Neutral
            _ => 0.9,                     // Slight penalty for less common languages
        }
    }

    /// Get common identifier patterns for this language
    pub fn identifier_patterns(&self) -> Vec<&'static str> {
        match self {
            Language::Rust => vec![
                r"fn\s+(\w+)",           // Functions
                r"struct\s+(\w+)",       // Structs
                r"enum\s+(\w+)",         // Enums
                r"trait\s+(\w+)",        // Traits
                r"impl\s+(?:\w+\s+for\s+)?(\w+)", // Implementations
                r"mod\s+(\w+)",          // Modules
                r"use\s+(?:.+::)?(\w+)", // Imports
            ],
            Language::TypeScript | Language::JavaScript => vec![
                r"function\s+(\w+)",     // Functions
                r"class\s+(\w+)",        // Classes
                r"interface\s+(\w+)",    // Interfaces (TS)
                r"type\s+(\w+)",         // Type aliases (TS)
                r"const\s+(\w+)",        // Constants
                r"let\s+(\w+)",          // Variables
                r"var\s+(\w+)",          // Variables
                r"export\s+(?:function|class|interface|type|const|let|var)\s+(\w+)", // Exports
            ],
            Language::Python => vec![
                r"def\s+(\w+)",          // Functions
                r"class\s+(\w+)",        // Classes
                r"import\s+(\w+)",       // Imports
                r"from\s+\w+\s+import\s+(\w+)", // From imports
                r"(\w+)\s*=\s*", // Assignments (variables)
            ],
            Language::Go => vec![
                r"func\s+(\w+)",         // Functions
                r"type\s+(\w+)",         // Types
                r"var\s+(\w+)",          // Variables
                r"const\s+(\w+)",        // Constants
                r"package\s+(\w+)",      // Package
                r#"import\s+(?:\w+\s+)?"(?:.+/)?(\w+)""#, // Imports
            ],
            Language::Java => vec![
                r"public\s+(?:static\s+)?(?:class|interface|enum)\s+(\w+)", // Classes/Interfaces/Enums
                r"public\s+(?:static\s+)?(?:\w+\s+)*(\w+)\s*\(", // Methods
                r"private\s+(?:static\s+)?(?:\w+\s+)*(\w+)\s*\(", // Private methods
                r"(?:public|private|protected)\s+(?:static\s+)?(\w+)\s+(\w+);", // Fields
            ],
            _ => vec![], // Default: no patterns
        }
    }

    /// Check if file should be indexed based on language
    pub fn should_index(&self) -> bool {
        !matches!(self, Language::Unknown)
    }
}

/// Language statistics for corpus analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
    pub language: Language,
    pub file_count: usize,
    pub total_lines: usize,
    pub total_bytes: u64,
    pub average_file_size: f64,
}

/// Language detector that can analyze file content
pub struct LanguageDetector {
    // Extension-based detection patterns
    extension_map: HashMap<String, Language>,
}

impl LanguageDetector {
    /// Create a new language detector
    pub fn new() -> Self {
        let mut extension_map = HashMap::new();
        
        // Build comprehensive extension mapping
        for lang in [
            Language::Rust,
            Language::TypeScript,
            Language::JavaScript,
            Language::Python,
            Language::Go,
            Language::Java,
            Language::CSharp,
            Language::Cpp,
            Language::C,
            Language::Ruby,
            Language::Php,
            Language::Swift,
            Language::Kotlin,
            Language::Scala,
            Language::Clojure,
            Language::Haskell,
        ] {
            for ext in lang.extensions() {
                extension_map.insert(ext.to_string(), lang.clone());
            }
        }

        Self { extension_map }
    }

    /// Detect language from file path
    pub fn detect_from_path(&self, path: &std::path::Path) -> Language {
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                return self.extension_map
                    .get(&ext_str.to_lowercase())
                    .cloned()
                    .unwrap_or(Language::Unknown);
            }
        }
        Language::Unknown
    }

    /// Detect language from file content (fallback method)
    pub fn detect_from_content(&self, content: &str) -> Language {
        // Simple heuristic-based detection for common patterns
        if content.contains("fn main()") || content.contains("use std::") {
            Language::Rust
        } else if content.contains("interface ") || content.contains(": string") {
            Language::TypeScript
        } else if content.contains("def ") || content.contains("import ") || content.contains("print(") {
            Language::Python
        } else if content.contains("func main()") || content.contains("package main") {
            Language::Go
        } else if content.contains("public class") || content.contains("import java.") {
            Language::Java
        } else {
            Language::Unknown
        }
    }

    /// Get all supported languages
    pub fn supported_languages(&self) -> Vec<Language> {
        let mut languages: Vec<_> = self.extension_map.values().cloned().collect();
        languages.sort_by_key(|lang| lang.display_name());
        languages.dedup();
        languages
    }

    /// Check if extension is supported
    pub fn is_supported_extension(&self, extension: &str) -> bool {
        self.extension_map.contains_key(&extension.to_lowercase())
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("unknown"), Language::Unknown);
    }

    #[test]
    fn test_language_detector() {
        let detector = LanguageDetector::new();
        
        assert_eq!(detector.detect_from_path(Path::new("main.rs")), Language::Rust);
        assert_eq!(detector.detect_from_path(Path::new("app.ts")), Language::TypeScript);
        assert_eq!(detector.detect_from_path(Path::new("script.py")), Language::Python);
        assert_eq!(detector.detect_from_path(Path::new("unknown.xyz")), Language::Unknown);
    }

    #[test]
    fn test_lsp_support() {
        assert!(Language::Rust.supports_lsp());
        assert!(Language::TypeScript.supports_lsp());
        assert!(Language::Python.supports_lsp());
        assert!(Language::Go.supports_lsp());
        assert!(!Language::Unknown.supports_lsp());
    }

    #[test]
    fn test_content_detection() {
        let detector = LanguageDetector::new();
        
        let rust_content = "fn main() { println!(\"Hello\"); }";
        assert_eq!(detector.detect_from_content(rust_content), Language::Rust);
        
        let ts_content = "interface User { name: string; }";
        assert_eq!(detector.detect_from_content(ts_content), Language::TypeScript);
        
        let python_content = "def hello():\n    print('Hello')";
        assert_eq!(detector.detect_from_content(python_content), Language::Python);
    }

    #[test]
    fn test_search_boost() {
        assert_eq!(Language::Rust.search_boost(), 1.2);
        assert_eq!(Language::TypeScript.search_boost(), 1.1);
        assert_eq!(Language::JavaScript.search_boost(), 1.0);
        assert!(Language::Unknown.search_boost() < 1.0);
    }

    #[test]
    fn test_supported_languages() {
        let detector = LanguageDetector::new();
        let languages = detector.supported_languages();
        
        assert!(!languages.is_empty());
        assert!(languages.contains(&Language::Rust));
        assert!(languages.contains(&Language::TypeScript));
        assert!(!languages.contains(&Language::Unknown));
    }
}