//! # Language-Specific Tokenization
//!
//! Dedicated tokenizers for JS/Go/Java beyond existing TS/Python/Rust.
//! Language-specific keyword recognition and syntax-aware scoring.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// Tokenization configuration for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationConfig {
    /// Language identifier
    pub language: String,
    /// Language-specific keywords
    pub keywords: Vec<String>,
    /// Syntax delimiters and operators
    pub syntax_tokens: Vec<String>,
    /// Comment patterns
    pub comment_patterns: Vec<String>,
    /// String literal patterns
    pub string_patterns: Vec<String>,
    /// Identifier patterns (regex)
    pub identifier_patterns: Vec<String>,
    /// Token importance weights
    pub token_weights: HashMap<String, f32>,
}

/// Language-specific tokenizer
#[derive(Debug, Clone)]
pub struct LanguageTokenizer {
    config: TokenizationConfig,
    /// Compiled keyword set for fast lookup
    keyword_set: HashSet<String>,
    /// Syntax token weights
    syntax_weights: HashMap<String, f32>,
    /// Token frequency statistics
    token_stats: TokenStatistics,
}

/// Token statistics for scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStatistics {
    /// Total tokens processed
    pub total_tokens: usize,
    /// Keyword frequency
    pub keyword_frequency: HashMap<String, usize>,
    /// Identifier frequency
    pub identifier_frequency: HashMap<String, usize>,
    /// Average tokens per query
    pub avg_tokens_per_query: f32,
}

/// Tokenization result with language-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationResult {
    /// Raw tokens
    pub tokens: Vec<String>,
    /// Language-specific features
    pub features: LanguageFeatures,
    /// Token scores for ranking
    pub token_scores: HashMap<String, f32>,
    /// Detected language patterns
    pub patterns: Vec<DetectedPattern>,
}

/// Language-specific features extracted from tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageFeatures {
    /// Keyword density (keywords / total tokens)
    pub keyword_density: f32,
    /// Identifier count
    pub identifier_count: usize,
    /// Syntax complexity score
    pub syntax_complexity: f32,
    /// Comment ratio
    pub comment_ratio: f32,
    /// String literal count
    pub string_literal_count: usize,
    /// Average token length
    pub avg_token_length: f32,
    /// Language-specific pattern matches
    pub pattern_matches: usize,
}

/// Detected language pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Matched text
    pub matched_text: String,
    /// Confidence score
    pub confidence: f32,
    /// Position in text
    pub position: usize,
}

/// Types of language patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Function definition
    FunctionDefinition,
    /// Class definition
    ClassDefinition,
    /// Interface definition
    InterfaceDefinition,
    /// Type definition
    TypeDefinition,
    /// Import statement
    ImportStatement,
    /// Variable declaration
    VariableDeclaration,
    /// Method call
    MethodCall,
    /// Control flow (if, for, while)
    ControlFlow,
    /// Lambda/Anonymous function
    Lambda,
    /// Generics usage
    Generics,
}

impl TokenizationConfig {
    /// Create configuration for a specific language
    pub fn for_language(language: &str) -> Self {
        match language.to_lowercase().as_str() {
            "javascript" | "js" => Self::javascript_config(),
            "go" => Self::go_config(),
            "java" => Self::java_config(),
            "typescript" | "ts" => Self::typescript_config(),
            "python" | "py" => Self::python_config(),
            "rust" | "rs" => Self::rust_config(),
            _ => Self::default_config(language),
        }
    }

    /// JavaScript-specific tokenization configuration
    fn javascript_config() -> Self {
        Self {
            language: "javascript".to_string(),
            keywords: vec![
                "function".to_string(), "const".to_string(), "let".to_string(), "var".to_string(),
                "class".to_string(), "extends".to_string(), "import".to_string(), "export".to_string(),
                "async".to_string(), "await".to_string(), "promise".to_string(), "then".to_string(),
                "catch".to_string(), "try".to_string(), "throw".to_string(), "new".to_string(),
                "this".to_string(), "prototype".to_string(), "bind".to_string(), "call".to_string(),
                "apply".to_string(), "arrow".to_string(), "spread".to_string(), "destructuring".to_string(),
            ],
            syntax_tokens: vec![
                "=>".to_string(), "...".to_string(), "?.".to_string(), "??".to_string(),
                "===".to_string(), "!==".to_string(), "&&".to_string(), "||".to_string(),
                "++".to_string(), "--".to_string(), "+=".to_string(), "-=".to_string(),
            ],
            comment_patterns: vec!["//".to_string(), "/*".to_string(), "*/".to_string()],
            string_patterns: vec!["\"".to_string(), "'".to_string(), "`".to_string()],
            identifier_patterns: vec![
                r"function\s+(\w+)".to_string(),
                r"const\s+(\w+)".to_string(),
                r"class\s+(\w+)".to_string(),
                r"(\w+)\s*=>".to_string(),
            ],
            token_weights: [
                ("function", 2.0), ("class", 2.0), ("async", 1.5), ("await", 1.5),
                ("import", 1.8), ("export", 1.8), ("const", 1.2), ("let", 1.2),
            ].iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    /// Go-specific tokenization configuration
    fn go_config() -> Self {
        Self {
            language: "go".to_string(),
            keywords: vec![
                "func".to_string(), "package".to_string(), "import".to_string(), "var".to_string(),
                "const".to_string(), "type".to_string(), "struct".to_string(), "interface".to_string(),
                "go".to_string(), "goroutine".to_string(), "channel".to_string(), "select".to_string(),
                "defer".to_string(), "panic".to_string(), "recover".to_string(), "make".to_string(),
                "new".to_string(), "append".to_string(), "len".to_string(), "cap".to_string(),
                "range".to_string(), "map".to_string(), "slice".to_string(),
            ],
            syntax_tokens: vec![
                ":=".to_string(), "<-".to_string(), "->".to_string(), "...".to_string(),
                "==".to_string(), "!=".to_string(), "<=".to_string(), ">=".to_string(),
                "&&".to_string(), "||".to_string(), "++".to_string(), "--".to_string(),
            ],
            comment_patterns: vec!["//".to_string(), "/*".to_string(), "*/".to_string()],
            string_patterns: vec!["\"".to_string(), "`".to_string()],
            identifier_patterns: vec![
                r"func\s+(\w+)".to_string(),
                r"type\s+(\w+)".to_string(),
                r"var\s+(\w+)".to_string(),
                r"package\s+(\w+)".to_string(),
            ],
            token_weights: [
                ("func", 2.0), ("type", 2.0), ("struct", 1.8), ("interface", 1.8),
                ("goroutine", 2.5), ("channel", 2.0), ("defer", 1.5), ("go", 2.0),
            ].iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    /// Java-specific tokenization configuration  
    fn java_config() -> Self {
        Self {
            language: "java".to_string(),
            keywords: vec![
                "public".to_string(), "private".to_string(), "protected".to_string(), "static".to_string(),
                "final".to_string(), "abstract".to_string(), "class".to_string(), "interface".to_string(),
                "extends".to_string(), "implements".to_string(), "import".to_string(), "package".to_string(),
                "void".to_string(), "return".to_string(), "new".to_string(), "this".to_string(),
                "super".to_string(), "synchronized".to_string(), "volatile".to_string(), "transient".to_string(),
                "try".to_string(), "catch".to_string(), "finally".to_string(), "throw".to_string(),
                "throws".to_string(), "generic".to_string(), "annotation".to_string(),
            ],
            syntax_tokens: vec![
                "::".to_string(), "->".to_string(), "@".to_string(), "<>".to_string(),
                "==".to_string(), "!=".to_string(), "<=".to_string(), ">=".to_string(),
                "&&".to_string(), "||".to_string(), "++".to_string(), "--".to_string(),
                "+=".to_string(), "-=".to_string(), "*=".to_string(), "/=".to_string(),
            ],
            comment_patterns: vec!["//".to_string(), "/*".to_string(), "*/".to_string(), "/**".to_string()],
            string_patterns: vec!["\"".to_string()],
            identifier_patterns: vec![
                r"public\s+(?:static\s+)?class\s+(\w+)".to_string(),
                r"public\s+(?:static\s+)?interface\s+(\w+)".to_string(),
                r"public\s+(?:static\s+)?(\w+)\s+(\w+)\s*\(".to_string(),
                r"@(\w+)".to_string(),
            ],
            token_weights: [
                ("public", 1.8), ("private", 1.5), ("class", 2.0), ("interface", 2.0),
                ("static", 1.3), ("final", 1.3), ("abstract", 1.5), ("synchronized", 1.8),
                ("annotation", 1.7), ("generic", 1.5), ("extends", 1.6), ("implements", 1.6),
            ].iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    /// TypeScript-specific configuration (extends JavaScript)
    fn typescript_config() -> Self {
        let mut config = Self::javascript_config();
        config.language = "typescript".to_string();
        
        // Add TypeScript-specific keywords
        config.keywords.extend(vec![
            "interface".to_string(), "type".to_string(), "enum".to_string(), "namespace".to_string(),
            "module".to_string(), "declare".to_string(), "readonly".to_string(), "as".to_string(),
            "is".to_string(), "keyof".to_string(), "typeof".to_string(), "generic".to_string(),
            "extends".to_string(), "implements".to_string(), "abstract".to_string(),
        ]);

        // Add TypeScript-specific syntax
        config.syntax_tokens.extend(vec![
            ":".to_string(), "as".to_string(), "!".to_string(), "<>".to_string(),
            "keyof".to_string(), "typeof".to_string(),
        ]);

        // Add TypeScript-specific patterns
        config.identifier_patterns.extend(vec![
            r"interface\s+(\w+)".to_string(),
            r"type\s+(\w+)".to_string(),
            r"enum\s+(\w+)".to_string(),
            r"namespace\s+(\w+)".to_string(),
        ]);

        // Add TypeScript-specific weights
        config.token_weights.insert("interface".to_string(), 2.0);
        config.token_weights.insert("type".to_string(), 1.8);
        config.token_weights.insert("enum".to_string(), 1.5);
        config.token_weights.insert("generic".to_string(), 1.7);

        config
    }

    /// Python-specific configuration
    fn python_config() -> Self {
        Self {
            language: "python".to_string(),
            keywords: vec![
                "def".to_string(), "class".to_string(), "import".to_string(), "from".to_string(),
                "as".to_string(), "with".to_string(), "async".to_string(), "await".to_string(),
                "lambda".to_string(), "yield".to_string(), "return".to_string(), "pass".to_string(),
                "global".to_string(), "nonlocal".to_string(), "assert".to_string(), "raise".to_string(),
                "try".to_string(), "except".to_string(), "finally".to_string(), "decorator".to_string(),
            ],
            syntax_tokens: vec![
                "->".to_string(), ":=".to_string(), "**".to_string(), "//".to_string(),
                "==".to_string(), "!=".to_string(), "<=".to_string(), ">=".to_string(),
                "and".to_string(), "or".to_string(), "not".to_string(), "in".to_string(),
                "is".to_string(), "@".to_string(), "...".to_string(),
            ],
            comment_patterns: vec!["#".to_string(), "\"\"\"".to_string(), "'''".to_string()],
            string_patterns: vec!["\"".to_string(), "'".to_string(), "f\"".to_string(), "r\"".to_string()],
            identifier_patterns: vec![
                r"def\s+(\w+)".to_string(),
                r"class\s+(\w+)".to_string(),
                r"@(\w+)".to_string(),
                r"import\s+(\w+)".to_string(),
            ],
            token_weights: [
                ("def", 2.0), ("class", 2.0), ("async", 1.8), ("await", 1.8),
                ("import", 1.5), ("decorator", 1.7), ("lambda", 1.5), ("yield", 1.5),
            ].iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    /// Rust-specific configuration
    fn rust_config() -> Self {
        Self {
            language: "rust".to_string(),
            keywords: vec![
                "fn".to_string(), "struct".to_string(), "enum".to_string(), "trait".to_string(),
                "impl".to_string(), "mod".to_string(), "use".to_string(), "pub".to_string(),
                "let".to_string(), "mut".to_string(), "const".to_string(), "static".to_string(),
                "match".to_string(), "if".to_string(), "else".to_string(), "loop".to_string(),
                "while".to_string(), "for".to_string(), "async".to_string(), "await".to_string(),
                "unsafe".to_string(), "extern".to_string(), "crate".to_string(), "macro".to_string(),
            ],
            syntax_tokens: vec![
                "->".to_string(), "=>".to_string(), "::".to_string(), "..".to_string(),
                "...".to_string(), "&".to_string(), "*".to_string(), "?".to_string(),
                "|".to_string(), "||".to_string(), "&&".to_string(), "!".to_string(),
            ],
            comment_patterns: vec!["//".to_string(), "/*".to_string(), "*/".to_string(), "///".to_string()],
            string_patterns: vec!["\"".to_string(), "r\"".to_string(), "b\"".to_string()],
            identifier_patterns: vec![
                r"fn\s+(\w+)".to_string(),
                r"struct\s+(\w+)".to_string(),
                r"enum\s+(\w+)".to_string(),
                r"trait\s+(\w+)".to_string(),
                r"impl\s+(?:\w+\s+for\s+)?(\w+)".to_string(),
            ],
            token_weights: [
                ("fn", 2.0), ("struct", 2.0), ("enum", 1.8), ("trait", 2.0),
                ("impl", 1.8), ("async", 1.5), ("unsafe", 2.5), ("macro", 1.7),
            ].iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    /// Default configuration for unknown languages
    fn default_config(language: &str) -> Self {
        Self {
            language: language.to_string(),
            keywords: vec![],
            syntax_tokens: vec![],
            comment_patterns: vec!["//".to_string(), "#".to_string()],
            string_patterns: vec!["\"".to_string(), "'".to_string()],
            identifier_patterns: vec![],
            token_weights: HashMap::new(),
        }
    }
}

impl LanguageTokenizer {
    /// Create new language tokenizer
    pub async fn new(config: TokenizationConfig) -> Result<Self> {
        info!("Creating tokenizer for language: {}", config.language);
        
        let keyword_set: HashSet<String> = config.keywords.iter().cloned().collect();
        let syntax_weights = config.token_weights.clone();
        
        Ok(Self {
            config,
            keyword_set,
            syntax_weights,
            token_stats: TokenStatistics::default(),
        })
    }

    /// Tokenize text with language-specific features
    pub async fn tokenize(&mut self, text: &str) -> Result<TokenizationResult> {
        let tokens = self.extract_tokens(text)?;
        let features = self.extract_language_features(&tokens, text)?;
        let token_scores = self.calculate_token_scores(&tokens)?;
        let patterns = self.detect_patterns(text).await?;

        // Update statistics
        self.update_statistics(&tokens)?;

        debug!(
            "Tokenized {} text: {} tokens, {} patterns, keyword_density={:.3}",
            self.config.language, tokens.len(), patterns.len(), features.keyword_density
        );

        Ok(TokenizationResult {
            tokens,
            features,
            token_scores,
            patterns,
        })
    }

    /// Get tokenizer statistics
    pub fn get_statistics(&self) -> &TokenStatistics {
        &self.token_stats
    }

    /// Get language identifier
    pub fn get_language(&self) -> &str {
        &self.config.language
    }

    // Private implementation methods

    /// Extract raw tokens from text
    fn extract_tokens(&self, text: &str) -> Result<Vec<String>> {
        // Simple tokenization - split on whitespace and common delimiters
        let delimiters = [' ', '\t', '\n', '\r', '(', ')', '[', ']', '{', '}', 
                         ',', ';', '.', ':', '=', '+', '-', '*', '/', '<', '>', '!'];
        
        let tokens: Vec<String> = text
            .split(|c| delimiters.contains(&c))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        Ok(tokens)
    }

    /// Extract language-specific features
    fn extract_language_features(&self, tokens: &[String], text: &str) -> Result<LanguageFeatures> {
        let total_tokens = tokens.len();
        let keyword_count = tokens.iter()
            .filter(|token| self.keyword_set.contains(*token))
            .count();
        
        let keyword_density = if total_tokens > 0 {
            keyword_count as f32 / total_tokens as f32
        } else {
            0.0
        };

        // Count identifiers (simple heuristic: alphanumeric starting with letter)
        let identifier_count = tokens.iter()
            .filter(|token| {
                token.chars().next().map_or(false, |c| c.is_alphabetic()) &&
                token.chars().all(|c| c.is_alphanumeric() || c == '_')
            })
            .count();

        // Calculate syntax complexity (count of syntax tokens)
        let syntax_complexity = tokens.iter()
            .filter(|token| self.config.syntax_tokens.contains(token))
            .count() as f32 / total_tokens.max(1) as f32;

        // Count comments and strings (simple approximation)
        let comment_count = self.config.comment_patterns.iter()
            .map(|pattern| text.matches(pattern).count())
            .sum::<usize>();
        
        let comment_ratio = comment_count as f32 / text.len().max(1) as f32;

        let string_literal_count = self.config.string_patterns.iter()
            .map(|pattern| text.matches(pattern).count())
            .sum::<usize>() / 2; // Assume pairs of string delimiters

        let avg_token_length = if total_tokens > 0 {
            tokens.iter().map(|t| t.len()).sum::<usize>() as f32 / total_tokens as f32
        } else {
            0.0
        };

        // Pattern matching count
        let pattern_matches = self.config.identifier_patterns.len(); // Simplified

        Ok(LanguageFeatures {
            keyword_density,
            identifier_count,
            syntax_complexity,
            comment_ratio,
            string_literal_count,
            avg_token_length,
            pattern_matches,
        })
    }

    /// Calculate scores for individual tokens
    fn calculate_token_scores(&self, tokens: &[String]) -> Result<HashMap<String, f32>> {
        let mut scores = HashMap::new();
        
        for token in tokens {
            let base_score = 1.0;
            
            // Apply keyword weight
            let keyword_weight = if self.keyword_set.contains(token) {
                self.syntax_weights.get(token).copied().unwrap_or(1.5)
            } else {
                1.0
            };
            
            // Apply syntax token weight
            let syntax_weight = if self.config.syntax_tokens.contains(token) {
                1.2
            } else {
                1.0
            };
            
            // Length bonus for longer identifiers
            let length_bonus = (token.len() as f32 / 10.0).min(1.0);
            
            let final_score = base_score * keyword_weight * syntax_weight * (1.0 + length_bonus);
            scores.insert(token.clone(), final_score);
        }
        
        Ok(scores)
    }

    /// Detect language-specific patterns
    async fn detect_patterns(&self, text: &str) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Simple pattern detection based on configured patterns
        for (i, pattern_regex) in self.config.identifier_patterns.iter().enumerate() {
            // This would use a proper regex engine in production
            if text.contains(&pattern_regex.replace(r"\s+", " ").replace(r"\w+", "word")) {
                patterns.push(DetectedPattern {
                    pattern_type: match i {
                        0 => PatternType::FunctionDefinition,
                        1 => PatternType::ClassDefinition,
                        2 => PatternType::TypeDefinition,
                        3 => PatternType::ImportStatement,
                        _ => PatternType::VariableDeclaration,
                    },
                    matched_text: pattern_regex.clone(),
                    confidence: 0.8,
                    position: 0, // Would be actual position in real implementation
                });
            }
        }

        // Detect common patterns based on language
        match self.config.language.as_str() {
            "javascript" | "typescript" => {
                if text.contains("=>") {
                    patterns.push(DetectedPattern {
                        pattern_type: PatternType::Lambda,
                        matched_text: "arrow function".to_string(),
                        confidence: 0.9,
                        position: 0,
                    });
                }
            },
            "java" => {
                if text.contains("@") && text.contains("(") {
                    patterns.push(DetectedPattern {
                        pattern_type: PatternType::Lambda,
                        matched_text: "annotation".to_string(),
                        confidence: 0.85,
                        position: 0,
                    });
                }
            },
            "go" => {
                if text.contains("go ") || text.contains("goroutine") {
                    patterns.push(DetectedPattern {
                        pattern_type: PatternType::Lambda,
                        matched_text: "goroutine".to_string(),
                        confidence: 0.95,
                        position: 0,
                    });
                }
            },
            _ => {}
        }

        Ok(patterns)
    }

    /// Update internal statistics
    fn update_statistics(&mut self, tokens: &[String]) -> Result<()> {
        self.token_stats.total_tokens += tokens.len();
        
        // Update keyword frequencies
        for token in tokens {
            if self.keyword_set.contains(token) {
                *self.token_stats.keyword_frequency.entry(token.clone()).or_insert(0) += 1;
            }
            
            // Update identifier frequencies (simplified)
            if token.chars().next().map_or(false, |c| c.is_alphabetic()) {
                *self.token_stats.identifier_frequency.entry(token.clone()).or_insert(0) += 1;
            }
        }
        
        Ok(())
    }
}

impl Default for TokenStatistics {
    fn default() -> Self {
        Self {
            total_tokens: 0,
            keyword_frequency: HashMap::new(),
            identifier_frequency: HashMap::new(),
            avg_tokens_per_query: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_javascript_tokenizer() {
        let config = TokenizationConfig::for_language("javascript");
        let mut tokenizer = LanguageTokenizer::new(config).await.unwrap();
        
        let text = "function test() { const x = 42; return x; }";
        let result = tokenizer.tokenize(text).await.unwrap();
        
        assert!(!result.tokens.is_empty());
        assert!(result.tokens.contains(&"function".to_string()));
        assert!(result.tokens.contains(&"const".to_string()));
        
        // Check features
        assert!(result.features.keyword_density > 0.0);
        assert!(result.features.identifier_count > 0);
        
        // Check token scores
        assert!(result.token_scores.get("function").unwrap_or(&0.0) > &1.0);
    }

    #[tokio::test]
    async fn test_go_tokenizer() {
        let config = TokenizationConfig::for_language("go");
        let mut tokenizer = LanguageTokenizer::new(config).await.unwrap();
        
        let text = "func main() { go routine(); defer cleanup(); }";
        let result = tokenizer.tokenize(text).await.unwrap();
        
        assert!(result.tokens.contains(&"func".to_string()));
        assert!(result.tokens.contains(&"go".to_string()));
        assert!(result.tokens.contains(&"defer".to_string()));
        
        // Check Go-specific features
        assert!(result.features.keyword_density > 0.0);
        assert!(!result.patterns.is_empty());
    }

    #[tokio::test]
    async fn test_java_tokenizer() {
        let config = TokenizationConfig::for_language("java");
        let mut tokenizer = LanguageTokenizer::new(config).await.unwrap();
        
        let text = "public class Test { @Override public void method() {} }";
        let result = tokenizer.tokenize(text).await.unwrap();
        
        assert!(result.tokens.contains(&"public".to_string()));
        assert!(result.tokens.contains(&"class".to_string()));
        
        // Check Java-specific features
        assert!(result.features.keyword_density > 0.0);
        assert!(result.token_scores.get("public").unwrap_or(&0.0) > &1.0);
    }

    #[tokio::test]
    async fn test_typescript_tokenizer() {
        let config = TokenizationConfig::for_language("typescript");
        let mut tokenizer = LanguageTokenizer::new(config.clone()).await.unwrap();
        
        let text = "interface User { name: string; } type ID = number;";
        let result = tokenizer.tokenize(text).await.unwrap();
        
        assert!(result.tokens.contains(&"interface".to_string()));
        assert!(result.tokens.contains(&"type".to_string()));
        
        // TypeScript should have more keywords than JavaScript
        let js_config = TokenizationConfig::for_language("javascript");
        assert!(config.keywords.len() > js_config.keywords.len());
    }

    #[tokio::test]
    async fn test_pattern_detection() {
        let config = TokenizationConfig::for_language("javascript");
        let mut tokenizer = LanguageTokenizer::new(config).await.unwrap();
        
        let text = "const handler = (x) => x * 2;";
        let result = tokenizer.tokenize(text).await.unwrap();
        
        // Should detect arrow function pattern
        let lambda_patterns: Vec<_> = result.patterns.iter()
            .filter(|p| matches!(p.pattern_type, PatternType::Lambda))
            .collect();
        
        assert!(!lambda_patterns.is_empty());
    }

    #[tokio::test]
    async fn test_token_scoring() {
        let config = TokenizationConfig::for_language("rust");
        let mut tokenizer = LanguageTokenizer::new(config).await.unwrap();
        
        let text = "fn test() { let x = 42; unsafe { ... } }";
        let result = tokenizer.tokenize(text).await.unwrap();
        
        // Keywords should have higher scores
        let fn_score = result.token_scores.get("fn").unwrap_or(&0.0);
        let unsafe_score = result.token_scores.get("unsafe").unwrap_or(&0.0);
        let number_score = result.token_scores.get("42").unwrap_or(&0.0);
        
        assert!(fn_score > number_score);
        assert!(unsafe_score > number_score); // 'unsafe' should have high weight
    }

    #[tokio::test]
    async fn test_language_features_extraction() {
        let config = TokenizationConfig::for_language("python");
        let mut tokenizer = LanguageTokenizer::new(config).await.unwrap();
        
        let text = "def hello(name): # This is a comment\n    return f'Hello {name}'";
        let result = tokenizer.tokenize(text).await.unwrap();
        
        assert!(result.features.keyword_density > 0.0);
        assert!(result.features.comment_ratio > 0.0);
        assert!(result.features.string_literal_count > 0);
        assert!(result.features.avg_token_length > 0.0);
    }

    #[test]
    fn test_config_for_language() {
        let js_config = TokenizationConfig::for_language("javascript");
        assert_eq!(js_config.language, "javascript");
        assert!(js_config.keywords.contains(&"function".to_string()));
        
        let go_config = TokenizationConfig::for_language("go");
        assert_eq!(go_config.language, "go");
        assert!(go_config.keywords.contains(&"func".to_string()));
        
        let unknown_config = TokenizationConfig::for_language("unknown");
        assert_eq!(unknown_config.language, "unknown");
        assert!(unknown_config.keywords.is_empty());
    }

    #[tokio::test]
    async fn test_statistics_update() {
        let config = TokenizationConfig::for_language("rust");
        let mut tokenizer = LanguageTokenizer::new(config).await.unwrap();
        
        let text1 = "fn test1() {}";
        let text2 = "fn test2() { struct Data {} }";
        
        tokenizer.tokenize(text1).await.unwrap();
        tokenizer.tokenize(text2).await.unwrap();
        
        let stats = tokenizer.get_statistics();
        assert!(stats.total_tokens > 0);
        assert!(stats.keyword_frequency.get("fn").unwrap_or(&0) > &0);
        assert!(stats.keyword_frequency.get("struct").unwrap_or(&0) > &0);
    }
}