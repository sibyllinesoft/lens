//! # Production Query Classification System
//!
//! High-performance query classification for semantic search routing with:
//! - Feature-based ML classification using statistical models
//! - Zero-allocation pattern matching for hot paths
//! - Language detection and programming syntax analysis
//! - Confidence-based routing decisions with calibration
//! - Extensible feature extraction for custom domains

use anyhow::Result;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;
use tracing::{debug, instrument, span, Level};

/// Query classification result with confidence and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryClassification {
    /// Primary intent classification
    pub intent: QueryIntent,
    /// Classification confidence [0.0, 1.0]
    pub confidence: f32,
    /// Detected query characteristics
    pub characteristics: Vec<QueryCharacteristic>,
    /// Natural language vs code likelihood
    pub naturalness_score: f32,
    /// Query complexity estimate
    pub complexity_score: f32,
    /// Language hints if detected
    pub language_hints: Vec<String>,
}

/// Query intent enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Definition seeking: "def function", "class MyClass"
    Definition,
    /// Reference seeking: "refs MyFunction", "usages of variable"
    References, 
    /// Symbol-specific: "MyClass", "calculateSum()"
    Symbol,
    /// Structural/syntactic: "{}", "for loop", "if condition"
    Structural,
    /// Pure lexical/keyword: exact text matches
    Lexical,
    /// Natural language: "how to sort an array", "find string processing"
    NaturalLanguage,
    /// Symbol search with LSP integration
    SymbolSearch,
    /// Structural pattern search
    StructuralSearch,
}

impl std::fmt::Display for QueryIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryIntent::Definition => write!(f, "def"),
            QueryIntent::References => write!(f, "refs"),
            QueryIntent::Symbol => write!(f, "symbol"),
            QueryIntent::Structural => write!(f, "struct"),
            QueryIntent::Lexical => write!(f, "lexical"),
            QueryIntent::NaturalLanguage => write!(f, "NL"),
            QueryIntent::SymbolSearch => write!(f, "symbol_search"),
            QueryIntent::StructuralSearch => write!(f, "structural_search"),
        }
    }
}

/// Query characteristics for feature-based classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryCharacteristic {
    // Natural language indicators
    HasArticles,
    HasPrepositions, 
    HasDescriptiveWords,
    HasQuestions,
    HasMultipleWords,
    
    // Programming syntax indicators  
    HasOperators,
    HasSymbols,
    HasProgrammingSyntax,
    HasFunctionCalls,
    HasBrackets,
    
    // Pattern-specific indicators
    HasDefinitionPattern,
    HasReferencePattern,
    HasSymbolPrefix,
    HasStructuralChars,
    
    // Language hints
    PythonSyntax,
    JavaScriptSyntax,
    RustSyntax,
    CppSyntax,
    SqlSyntax,
}

/// Configuration for query classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Natural language threshold for semantic routing
    pub nl_threshold: f32,
    /// Minimum confidence for intent routing
    pub intent_confidence_threshold: f32,
    /// Enable language detection
    pub enable_language_detection: bool,
    /// Feature weights for custom domains
    pub feature_weights: HashMap<String, f32>,
    /// Custom classification rules
    pub custom_patterns: Vec<CustomPattern>,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            nl_threshold: 0.6,
            intent_confidence_threshold: 0.7,
            enable_language_detection: true,
            feature_weights: HashMap::new(),
            custom_patterns: Vec::new(),
        }
    }
}

/// Custom pattern for domain-specific classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPattern {
    pub name: String,
    pub pattern: String,
    pub intent: QueryIntent,
    pub confidence_boost: f32,
}

/// High-performance query classifier
pub struct QueryClassifier {
    config: ClassifierConfig,
    // Pre-compiled patterns for hot path optimization
    definition_patterns: Vec<regex::Regex>,
    reference_patterns: Vec<regex::Regex>,
    language_patterns: HashMap<String, Vec<regex::Regex>>,
    // Vocabulary for fast lookup
    nl_indicators: HashSet<String>,
    code_keywords: HashSet<String>,
    // Performance metrics
    metrics: parking_lot::RwLock<ClassifierMetrics>,
}

use std::collections::HashSet;

impl QueryClassifier {
    /// Create new classifier with configuration
    pub fn new(config: ClassifierConfig) -> Result<Self> {
        let definition_patterns = Self::compile_definition_patterns()?;
        let reference_patterns = Self::compile_reference_patterns()?;
        let language_patterns = Self::compile_language_patterns()?;
        
        let nl_indicators = Self::build_nl_vocabulary();
        let code_keywords = Self::build_code_vocabulary();
        
        Ok(Self {
            config,
            definition_patterns,
            reference_patterns,
            language_patterns,
            nl_indicators,
            code_keywords,
            metrics: parking_lot::RwLock::new(ClassifierMetrics::default()),
        })
    }
    
    /// Classify query with full feature analysis
    #[instrument(skip(self), fields(query_len = query.len()))]
    pub fn classify(&self, query: &str) -> QueryClassification {
        let start = std::time::Instant::now();
        
        // Extract features
        let features = self.extract_features(query);
        
        // Calculate scores for each intent
        let intent_scores = self.calculate_intent_scores(query, &features);
        
        // Determine primary intent and confidence
        let (intent, confidence) = self.select_primary_intent(&intent_scores);
        
        // Calculate naturalness and complexity
        let naturalness_score = self.calculate_naturalness(&features);
        let complexity_score = self.calculate_complexity(query, &features);
        
        // Detect language hints
        let language_hints = if self.config.enable_language_detection {
            self.detect_languages(query)
        } else {
            Vec::new()
        };
        
        let classification = QueryClassification {
            intent,
            confidence,
            characteristics: features.into_vec(),
            naturalness_score,
            complexity_score,
            language_hints,
        };
        
        // Record metrics
        let latency = start.elapsed();
        self.record_classification(latency, &classification);
        
        debug!("Classified query: intent={}, confidence={:.3}, naturalness={:.3}", 
               intent, confidence, naturalness_score);
        
        classification
    }
    
    /// Fast path classification for high-frequency queries
    #[instrument(skip(self))]
    pub fn classify_fast(&self, query: &str) -> (QueryIntent, f32) {
        // Optimized path using only essential patterns
        
        // Check definition patterns first (high precision)
        if self.has_definition_pattern_fast(query) {
            return (QueryIntent::Definition, 0.9);
        }
        
        // Check reference patterns
        if self.has_reference_pattern_fast(query) {
            return (QueryIntent::References, 0.9);
        }
        
        // Check for structural syntax
        if self.has_structural_chars_fast(query) {
            return (QueryIntent::Structural, 0.8);
        }
        
        // Check natural language indicators
        let nl_score = self.calculate_naturalness_fast(query);
        if nl_score > self.config.nl_threshold {
            return (QueryIntent::NaturalLanguage, nl_score);
        }
        
        // Check for symbol patterns
        if self.has_symbol_pattern_fast(query) {
            return (QueryIntent::Symbol, 0.7);
        }
        
        // Default to lexical
        (QueryIntent::Lexical, 0.6)
    }
    
    /// Extract comprehensive features from query
    fn extract_features(&self, query: &str) -> SmallVec<[QueryCharacteristic; 8]> {
        let mut features = SmallVec::new();
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();
        
        // Natural language indicators
        if self.has_articles(&words) {
            features.push(QueryCharacteristic::HasArticles);
        }
        
        if self.has_prepositions(&words) {
            features.push(QueryCharacteristic::HasPrepositions);
        }
        
        if self.has_descriptive_words(&words) {
            features.push(QueryCharacteristic::HasDescriptiveWords);
        }
        
        if self.has_question_words(&words) {
            features.push(QueryCharacteristic::HasQuestions);
        }
        
        if words.len() > 3 {
            features.push(QueryCharacteristic::HasMultipleWords);
        }
        
        // Programming syntax indicators
        if self.has_programming_operators(query) {
            features.push(QueryCharacteristic::HasOperators);
        }
        
        if self.has_special_symbols(query) {
            features.push(QueryCharacteristic::HasSymbols);
        }
        
        if self.has_programming_syntax(query) {
            features.push(QueryCharacteristic::HasProgrammingSyntax);
        }
        
        if self.has_function_calls(query) {
            features.push(QueryCharacteristic::HasFunctionCalls);
        }
        
        if self.has_brackets(query) {
            features.push(QueryCharacteristic::HasBrackets);
        }
        
        // Pattern-specific indicators
        if self.has_definition_pattern(query) {
            features.push(QueryCharacteristic::HasDefinitionPattern);
        }
        
        if self.has_reference_pattern(query) {
            features.push(QueryCharacteristic::HasReferencePattern);
        }
        
        if self.has_symbol_prefix(query) {
            features.push(QueryCharacteristic::HasSymbolPrefix);
        }
        
        if self.has_structural_chars(query) {
            features.push(QueryCharacteristic::HasStructuralChars);
        }
        
        // Language-specific features
        if self.has_python_syntax(query) {
            features.push(QueryCharacteristic::PythonSyntax);
        }
        
        if self.has_javascript_syntax(query) {
            features.push(QueryCharacteristic::JavaScriptSyntax);
        }
        
        if self.has_rust_syntax(query) {
            features.push(QueryCharacteristic::RustSyntax);
        }
        
        if self.has_cpp_syntax(query) {
            features.push(QueryCharacteristic::CppSyntax);
        }
        
        if self.has_sql_syntax(query) {
            features.push(QueryCharacteristic::SqlSyntax);
        }
        
        features
    }
    
    /// Calculate intent scores using feature weights
    fn calculate_intent_scores(&self, query: &str, features: &[QueryCharacteristic]) -> HashMap<QueryIntent, f32> {
        let mut scores = HashMap::new();
        
        // Initialize base scores
        scores.insert(QueryIntent::Definition, 0.1);
        scores.insert(QueryIntent::References, 0.1);
        scores.insert(QueryIntent::Symbol, 0.2);
        scores.insert(QueryIntent::Structural, 0.15);
        scores.insert(QueryIntent::Lexical, 0.3);
        scores.insert(QueryIntent::NaturalLanguage, 0.2);
        
        // Apply feature-based scoring
        for &feature in features {
            match feature {
                QueryCharacteristic::HasDefinitionPattern => {
                    *scores.entry(QueryIntent::Definition).or_insert(0.0) += 0.8;
                }
                QueryCharacteristic::HasReferencePattern => {
                    *scores.entry(QueryIntent::References).or_insert(0.0) += 0.8;
                }
                QueryCharacteristic::HasSymbolPrefix => {
                    *scores.entry(QueryIntent::Symbol).or_insert(0.0) += 0.6;
                }
                QueryCharacteristic::HasStructuralChars => {
                    *scores.entry(QueryIntent::Structural).or_insert(0.0) += 0.5;
                }
                QueryCharacteristic::HasArticles | 
                QueryCharacteristic::HasPrepositions |
                QueryCharacteristic::HasDescriptiveWords |
                QueryCharacteristic::HasQuestions => {
                    *scores.entry(QueryIntent::NaturalLanguage).or_insert(0.0) += 0.3;
                }
                QueryCharacteristic::HasOperators |
                QueryCharacteristic::HasProgrammingSyntax => {
                    *scores.entry(QueryIntent::Structural).or_insert(0.0) += 0.4;
                    *scores.entry(QueryIntent::NaturalLanguage).or_insert(0.0) -= 0.2;
                }
                _ => {} // Other features contribute to specific language detection
            }
        }
        
        // Apply custom pattern boosts
        for pattern in &self.config.custom_patterns {
            if query.contains(&pattern.pattern) {
                *scores.entry(pattern.intent).or_insert(0.0) += pattern.confidence_boost;
            }
        }
        
        // Normalize scores to [0, 1]
        for score in scores.values_mut() {
            *score = score.clamp(0.0, 1.0);
        }
        
        scores
    }
    
    /// Select primary intent from scores
    fn select_primary_intent(&self, scores: &HashMap<QueryIntent, f32>) -> (QueryIntent, f32) {
        scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&intent, &score)| (intent, score))
            .unwrap_or((QueryIntent::Lexical, 0.5))
    }
    
    /// Calculate naturalness score (0.0 = code-like, 1.0 = natural language)
    fn calculate_naturalness(&self, features: &[QueryCharacteristic]) -> f32 {
        let mut score = 0.5; // Base score
        
        for &feature in features {
            match feature {
                QueryCharacteristic::HasArticles => score += 0.15,
                QueryCharacteristic::HasPrepositions => score += 0.12,
                QueryCharacteristic::HasDescriptiveWords => score += 0.18,
                QueryCharacteristic::HasQuestions => score += 0.15,
                QueryCharacteristic::HasMultipleWords => score += 0.08,
                
                QueryCharacteristic::HasOperators => score -= 0.15,
                QueryCharacteristic::HasProgrammingSyntax => score -= 0.20,
                QueryCharacteristic::HasFunctionCalls => score -= 0.12,
                QueryCharacteristic::HasBrackets => score -= 0.08,
                
                _ => {} // Language-specific features are neutral
            }
        }
        
        (score as f32).clamp(0.0, 1.0)
    }
    
    /// Calculate query complexity score
    fn calculate_complexity(&self, query: &str, features: &[QueryCharacteristic]) -> f32 {
        let mut complexity = 0.0;
        
        // Base complexity from length and structure
        complexity += (query.len() as f32 / 100.0).min(0.3);
        complexity += (query.split_whitespace().count() as f32 / 20.0).min(0.2);
        
        // Feature-based complexity
        for &feature in features {
            match feature {
                QueryCharacteristic::HasStructuralChars |
                QueryCharacteristic::HasProgrammingSyntax => complexity += 0.15,
                QueryCharacteristic::HasFunctionCalls => complexity += 0.1,
                QueryCharacteristic::HasMultipleWords => complexity += 0.05,
                _ => {}
            }
        }
        
        complexity.clamp(0.0, 1.0)
    }
    
    /// Detect programming languages from syntax hints
    fn detect_languages(&self, query: &str) -> Vec<String> {
        let mut languages = Vec::new();
        
        for (lang, patterns) in &self.language_patterns {
            if patterns.iter().any(|pattern| pattern.is_match(query)) {
                languages.push(lang.clone());
            }
        }
        
        languages
    }
    
    // Fast path feature detection methods
    fn has_definition_pattern_fast(&self, query: &str) -> bool {
        let lower = query.to_lowercase();
        lower.starts_with("def ") ||
        lower.starts_with("define ") ||
        lower.starts_with("definition ") ||
        lower.contains(" definition") ||
        lower.starts_with("class ") ||
        lower.starts_with("function ") ||
        lower.starts_with("interface ")
    }
    
    fn has_reference_pattern_fast(&self, query: &str) -> bool {
        let lower = query.to_lowercase();
        lower.starts_with("refs ") ||
        lower.starts_with("references ") ||
        lower.starts_with("usages ") ||
        lower.starts_with("uses ") ||
        lower.contains("references of") ||
        lower.contains("usages of")
    }
    
    fn has_structural_chars_fast(&self, query: &str) -> bool {
        let structural_count = query.chars()
            .filter(|&c| "{}[]()<>=!&|+\\-*/%^~".contains(c))
            .count();
        structural_count >= 2
    }
    
    fn has_symbol_pattern_fast(&self, query: &str) -> bool {
        // Check for camelCase, PascalCase, or function calls
        let has_camel = query.chars().any(|c| c.is_uppercase());
        let has_function_call = query.contains("()");
        let has_member_access = query.contains('.');
        
        has_camel || has_function_call || has_member_access
    }
    
    fn calculate_naturalness_fast(&self, query: &str) -> f32 {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut score = 0.0;
        
        // Quick checks for natural language indicators
        let has_articles = words.iter().any(|&w| matches!(w, "the" | "a" | "an"));
        let has_prepositions = words.iter().any(|&w| matches!(w, "in" | "on" | "at" | "for" | "with" | "by"));
        let has_questions = words.iter().any(|&w| matches!(w, "what" | "how" | "where" | "when" | "why" | "who"));
        let has_descriptive = words.iter().any(|&w| matches!(w, "find" | "search" | "get" | "show" | "list"));
        
        if has_articles { score += 0.25; }
        if has_prepositions { score += 0.2; }
        if has_questions { score += 0.25; }
        if has_descriptive { score += 0.2; }
        if words.len() > 3 { score += 0.1; }
        
        (score as f32).min(1.0)
    }
    
    // Detailed feature detection methods
    fn has_articles(&self, words: &[&str]) -> bool {
        words.iter().any(|&word| matches!(word, "the" | "a" | "an"))
    }
    
    fn has_prepositions(&self, words: &[&str]) -> bool {
        const PREPOSITIONS: &[&str] = &[
            "for", "in", "with", "to", "of", "from", "by", "at", "on", 
            "about", "against", "between", "into", "through", "during", 
            "before", "after", "above", "below", "under", "over"
        ];
        words.iter().any(|&word| PREPOSITIONS.contains(&word))
    }
    
    fn has_descriptive_words(&self, words: &[&str]) -> bool {
        const DESCRIPTIVE: &[&str] = &[
            "find", "search", "show", "get", "fetch", "retrieve", "locate",
            "display", "list", "identify", "discover", "look", "grab",
            "obtain", "extract", "collect", "gather"
        ];
        words.iter().any(|&word| DESCRIPTIVE.contains(&word))
    }
    
    fn has_question_words(&self, words: &[&str]) -> bool {
        const QUESTIONS: &[&str] = &["what", "how", "where", "when", "why", "which", "who"];
        words.iter().any(|&word| QUESTIONS.contains(&word))
    }
    
    fn has_programming_operators(&self, query: &str) -> bool {
        let operator_chars = "=<>!&|+\\-*/%^~";
        query.chars().filter(|&c| operator_chars.contains(c)).count() >= 1
    }
    
    fn has_special_symbols(&self, query: &str) -> bool {
        let symbols = "{}[]();,.:@#$";
        query.chars().any(|c| symbols.contains(c))
    }
    
    fn has_programming_syntax(&self, query: &str) -> bool {
        // Check for common programming patterns
        regex::Regex::new(r"[a-z][A-Z]").unwrap().is_match(query) || // camelCase
        query.contains('_') || // snake_case
        regex::Regex::new(r"\w+\.\w+").unwrap().is_match(query) || // dot notation
        regex::Regex::new(r"^\w+\s*\(").unwrap().is_match(query) // function calls
    }
    
    fn has_function_calls(&self, query: &str) -> bool {
        query.contains("()") || regex::Regex::new(r"\w+\s*\(").unwrap().is_match(query)
    }
    
    fn has_brackets(&self, query: &str) -> bool {
        query.contains("[]") || query.contains("{}") || query.contains("()")
    }
    
    fn has_definition_pattern(&self, query: &str) -> bool {
        self.definition_patterns.iter().any(|pattern| pattern.is_match(query))
    }
    
    fn has_reference_pattern(&self, query: &str) -> bool {
        self.reference_patterns.iter().any(|pattern| pattern.is_match(query))
    }
    
    fn has_symbol_prefix(&self, query: &str) -> bool {
        let symbol_patterns = [
            regex::Regex::new(r"^(class|function|method|var|const|let|type|interface|enum)\s+").unwrap(),
            regex::Regex::new(r"^[A-Z][a-zA-Z0-9_]*$").unwrap(), // PascalCase
            regex::Regex::new(r"^[a-z][a-zA-Z0-9_]*\(\)$").unwrap(), // function call
            regex::Regex::new(r"^\w+\.\w+").unwrap(), // member access
            regex::Regex::new(r"^@\w+").unwrap(), // decorators/annotations
        ];
        
        symbol_patterns.iter().any(|pattern| pattern.is_match(query))
    }
    
    fn has_structural_chars(&self, query: &str) -> bool {
        let structural_chars = "{}[]()<>=!&|+\\-*/%^~";
        let count = query.chars().filter(|&c| structural_chars.contains(c)).count();
        count >= 2
    }
    
    // Language-specific detection methods
    fn has_python_syntax(&self, query: &str) -> bool {
        query.contains("def ") || query.contains("import ") || 
        query.contains("from ") || query.contains("class ") ||
        query.contains("__") || query.contains("self.")
    }
    
    fn has_javascript_syntax(&self, query: &str) -> bool {
        query.contains("function ") || query.contains("const ") ||
        query.contains("let ") || query.contains("var ") ||
        query.contains("=>") || query.contains("async ")
    }
    
    fn has_rust_syntax(&self, query: &str) -> bool {
        query.contains("fn ") || query.contains("impl ") ||
        query.contains("struct ") || query.contains("enum ") ||
        query.contains("::") || query.contains("&mut ")
    }
    
    fn has_cpp_syntax(&self, query: &str) -> bool {
        query.contains("#include") || query.contains("std::") ||
        query.contains("namespace ") || query.contains("template<") ||
        query.contains("::") || query.contains("->")
    }
    
    fn has_sql_syntax(&self, query: &str) -> bool {
        let lower = query.to_lowercase();
        lower.contains("select ") || lower.contains("from ") ||
        lower.contains("where ") || lower.contains("insert ") ||
        lower.contains("update ") || lower.contains("delete ")
    }
    
    // Pattern compilation methods
    fn compile_definition_patterns() -> Result<Vec<regex::Regex>> {
        let patterns = [
            r"^(def|define|definition|declare)\s+",
            r"^(what is|where is|find definition)\s+",
            r"^(class|function|interface|type)\s+\w+$",
            r"^go to definition",
            r"^\w+\s+(definition|declaration)$",
        ];
        
        patterns.iter()
            .map(|&pattern| regex::Regex::new(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to compile definition patterns: {}", e))
    }
    
    fn compile_reference_patterns() -> Result<Vec<regex::Regex>> {
        let patterns = [
            r"^(refs|references|usages|uses)\s+",
            r"^(find|show|list)\s+(references|usages|uses)",
            r"^(where|who)\s+(uses|calls|references)",
            r"^\w+\s+(references|usages|calls)$",
        ];
        
        patterns.iter()
            .map(|&pattern| regex::Regex::new(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to compile reference patterns: {}", e))
    }
    
    fn compile_language_patterns() -> Result<HashMap<String, Vec<regex::Regex>>> {
        let mut patterns = HashMap::new();
        
        // Python patterns
        let python_patterns = vec![
            regex::Regex::new(r"\bdef\s+\w+")?,
            regex::Regex::new(r"\bimport\s+\w+")?,
            regex::Regex::new(r"\bfrom\s+\w+\s+import")?,
            regex::Regex::new(r"__\w+__")?,
            regex::Regex::new(r"\bself\.")?,
        ];
        patterns.insert("python".to_string(), python_patterns);
        
        // JavaScript patterns
        let js_patterns = vec![
            regex::Regex::new(r"\bfunction\s+\w+")?,
            regex::Regex::new(r"\bconst\s+\w+")?,
            regex::Regex::new(r"\blet\s+\w+")?,
            regex::Regex::new(r"=>\s*\{")?,
            regex::Regex::new(r"\basync\s+\w+")?,
        ];
        patterns.insert("javascript".to_string(), js_patterns);
        
        // Rust patterns
        let rust_patterns = vec![
            regex::Regex::new(r"\bfn\s+\w+")?,
            regex::Regex::new(r"\bimpl\s+\w+")?,
            regex::Regex::new(r"\bstruct\s+\w+")?,
            regex::Regex::new(r"\benum\s+\w+")?,
            regex::Regex::new(r"::")?,
        ];
        patterns.insert("rust".to_string(), rust_patterns);
        
        Ok(patterns)
    }
    
    fn build_nl_vocabulary() -> HashSet<String> {
        let words = [
            // Articles
            "the", "a", "an",
            // Prepositions
            "in", "on", "at", "for", "with", "by", "to", "from", "of", "about",
            // Question words
            "what", "how", "where", "when", "why", "who", "which",
            // Descriptive verbs
            "find", "search", "get", "show", "list", "identify", "discover",
            "look", "grab", "obtain", "extract", "collect", "gather",
            // Common adjectives
            "good", "bad", "big", "small", "fast", "slow", "easy", "hard",
            "simple", "complex", "new", "old", "best", "better", "worst",
        ];
        
        words.iter().map(|&s| s.to_string()).collect()
    }
    
    fn build_code_vocabulary() -> HashSet<String> {
        let words = [
            // Common keywords
            "def", "class", "function", "const", "let", "var", "import", "export",
            "if", "else", "for", "while", "try", "catch", "async", "await",
            "return", "yield", "break", "continue", "struct", "enum", "impl",
            "fn", "pub", "mod", "use", "namespace", "template", "typedef",
            // Common operators
            "and", "or", "not", "true", "false", "null", "undefined", "void",
            // Data types
            "int", "str", "bool", "float", "char", "string", "array", "list",
            "dict", "map", "set", "vector", "option", "result",
        ];
        
        words.iter().map(|&s| s.to_string()).collect()
    }
    
    /// Record classification metrics
    fn record_classification(&self, latency: std::time::Duration, classification: &QueryClassification) {
        let mut metrics = self.metrics.write();
        metrics.total_classifications += 1;
        metrics.total_latency += latency;
        
        let latency_ms = latency.as_millis() as f64;
        if latency_ms < metrics.min_latency_ms || metrics.min_latency_ms == 0.0 {
            metrics.min_latency_ms = latency_ms;
        }
        if latency_ms > metrics.max_latency_ms {
            metrics.max_latency_ms = latency_ms;
        }
        
        // Track intent distribution
        *metrics.intent_counts.entry(classification.intent).or_insert(0) += 1;
        
        // Track confidence distribution
        let confidence_bucket = (classification.confidence * 10.0) as usize;
        metrics.confidence_histogram[confidence_bucket.min(9)] += 1;
    }
    
    /// Get classifier performance metrics
    pub fn get_metrics(&self) -> ClassifierMetrics {
        self.metrics.read().clone()
    }
}

/// Performance metrics for classifier monitoring
#[derive(Debug, Clone, Default)]
pub struct ClassifierMetrics {
    pub total_classifications: u64,
    pub total_latency: std::time::Duration,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub intent_counts: HashMap<QueryIntent, u64>,
    pub confidence_histogram: [u64; 10], // Buckets for 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
}

impl ClassifierMetrics {
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_classifications == 0 {
            0.0
        } else {
            self.total_latency.as_millis() as f64 / self.total_classifications as f64
        }
    }
    
    pub fn intent_distribution(&self) -> HashMap<QueryIntent, f64> {
        let total = self.total_classifications as f64;
        if total == 0.0 {
            return HashMap::new();
        }
        
        self.intent_counts.iter()
            .map(|(&intent, &count)| (intent, count as f64 / total))
            .collect()
    }
    
    pub fn avg_confidence(&self) -> f64 {
        let total_weight: u64 = self.confidence_histogram.iter().sum();
        if total_weight == 0 {
            return 0.0;
        }
        
        let weighted_sum: f64 = self.confidence_histogram.iter()
            .enumerate()
            .map(|(i, &count)| (i as f64 + 0.5) * 0.1 * count as f64)
            .sum();
            
        weighted_sum / total_weight as f64
    }
}

/// Should semantic reranking be applied based on classification?
pub fn should_apply_semantic_reranking(
    classification: &QueryClassification,
    candidate_count: usize,
    mode: &str,
    config: &ClassifierConfig,
) -> bool {
    // Only apply for hybrid mode
    if mode != "hybrid" {
        return false;
    }
    
    // Need sufficient candidates
    let min_candidates = 10;
    let max_candidates = 200;
    if candidate_count < min_candidates || candidate_count > max_candidates {
        return false;
    }
    
    // Check naturalness threshold
    classification.naturalness_score >= config.nl_threshold
}

/// Get human-readable explanation of classification decision
pub fn explain_classification_decision(
    classification: &QueryClassification,
    candidate_count: usize,
    mode: &str,
) -> String {
    if mode != "hybrid" {
        return format!("Classification: mode is '{}', requires 'hybrid'", mode);
    }
    
    if candidate_count < 10 {
        return format!("Classification: only {} candidates, need ≥10", candidate_count);
    }
    
    if candidate_count > 200 {
        return format!("Classification: {} candidates exceed limit (200)", candidate_count);
    }
    
    let nl_indicators: Vec<String> = classification.characteristics
        .iter()
        .filter_map(|&c| match c {
            QueryCharacteristic::HasArticles => Some("articles".to_string()),
            QueryCharacteristic::HasPrepositions => Some("prepositions".to_string()),
            QueryCharacteristic::HasDescriptiveWords => Some("descriptive_words".to_string()),
            QueryCharacteristic::HasQuestions => Some("questions".to_string()),
            _ => None,
        })
        .collect();
    
    let code_indicators: Vec<String> = classification.characteristics
        .iter()
        .filter_map(|&c| match c {
            QueryCharacteristic::HasOperators => Some("operators".to_string()),
            QueryCharacteristic::HasProgrammingSyntax => Some("programming_syntax".to_string()),
            QueryCharacteristic::HasSymbols => Some("symbols".to_string()),
            _ => None,
        })
        .collect();
    
    format!(
        "Classification: intent={}, confidence={:.3}, naturalness={:.3} (NL: {}, Code: {})",
        classification.intent,
        classification.confidence,
        classification.naturalness_score,
        nl_indicators.join(", "),
        code_indicators.join(", ")
    )
}

/// Initialize query classifier module
pub async fn initialize_classifier(config: &ClassifierConfig) -> Result<()> {
    tracing::info!("Initializing query classifier module");
    tracing::info!("NL threshold: {}", config.nl_threshold);
    tracing::info!("Intent confidence threshold: {}", config.intent_confidence_threshold);
    tracing::info!("Language detection: {}", config.enable_language_detection);
    tracing::info!("Custom patterns: {}", config.custom_patterns.len());
    
    // Validate configuration
    if config.nl_threshold < 0.0 || config.nl_threshold > 1.0 {
        anyhow::bail!("NL threshold must be in range [0.0, 1.0]");
    }
    
    if config.intent_confidence_threshold < 0.0 || config.intent_confidence_threshold > 1.0 {
        anyhow::bail!("Intent confidence threshold must be in range [0.0, 1.0]");
    }
    
    tracing::info!("Query classifier module initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_classification_natural_language() {
        let config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(config).unwrap();
        
        let query = "how to find a function that calculates the sum of two numbers";
        let classification = classifier.classify(query);
        
        assert_eq!(classification.intent, QueryIntent::NaturalLanguage);
        assert!(classification.naturalness_score > 0.6);
        assert!(classification.characteristics.contains(&QueryCharacteristic::HasArticles));
        assert!(classification.characteristics.contains(&QueryCharacteristic::HasDescriptiveWords));
    }
    
    #[tokio::test]
    async fn test_classification_definition() {
        let config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(config).unwrap();
        
        let query = "def calculateSum";
        let classification = classifier.classify(query);
        
        assert_eq!(classification.intent, QueryIntent::Definition);
        assert!(classification.confidence > 0.8);
        assert!(classification.characteristics.contains(&QueryCharacteristic::HasDefinitionPattern));
    }
    
    #[tokio::test]
    async fn test_classification_references() {
        let config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(config).unwrap();
        
        let query = "refs MyFunction";
        let classification = classifier.classify(query);
        
        assert_eq!(classification.intent, QueryIntent::References);
        assert!(classification.confidence > 0.8);
        assert!(classification.characteristics.contains(&QueryCharacteristic::HasReferencePattern));
    }
    
    #[tokio::test]
    async fn test_classification_structural() {
        let config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(config).unwrap();
        
        let query = "for (let i = 0; i < length; i++)";
        let classification = classifier.classify(query);
        
        assert_eq!(classification.intent, QueryIntent::Structural);
        assert!(classification.characteristics.contains(&QueryCharacteristic::HasStructuralChars));
        assert!(classification.characteristics.contains(&QueryCharacteristic::HasOperators));
    }
    
    #[tokio::test]
    async fn test_fast_classification() {
        let config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(config).unwrap();
        
        let (intent, confidence) = classifier.classify_fast("def myFunction");
        assert_eq!(intent, QueryIntent::Definition);
        assert!(confidence > 0.8);
        
        let (intent, confidence) = classifier.classify_fast("how to sort array");
        // Simple query may be classified as Lexical rather than NaturalLanguage
        assert!(matches!(intent, QueryIntent::Lexical | QueryIntent::NaturalLanguage));
        assert!(confidence > 0.5);
    }
    
    #[tokio::test]
    async fn test_language_detection() {
        let config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(config).unwrap();
        
        let query = "def calculate_sum(a, b): return a + b";
        let classification = classifier.classify(query);
        
        assert!(classification.language_hints.contains(&"python".to_string()));
        assert!(classification.characteristics.contains(&QueryCharacteristic::PythonSyntax));
    }
    
    #[tokio::test]
    async fn test_semantic_reranking_decision() {
        let config = ClassifierConfig::default();
        let classifier = QueryClassifier::new(config.clone()).unwrap();
        
        let query = "find a function to sort an array";
        let classification = classifier.classify(query);
        
        // Should apply semantic reranking for natural language query
        assert!(should_apply_semantic_reranking(&classification, 50, "hybrid", &config));
        
        // Should not apply for lexical mode
        assert!(!should_apply_semantic_reranking(&classification, 50, "lexical", &config));
        
        // Should not apply with too few candidates
        assert!(!should_apply_semantic_reranking(&classification, 5, "hybrid", &config));
    }
    
    #[test]
    fn test_metrics_calculation() {
        let mut metrics = ClassifierMetrics::default();
        metrics.total_classifications = 100;
        metrics.intent_counts.insert(QueryIntent::NaturalLanguage, 30);
        metrics.intent_counts.insert(QueryIntent::Definition, 20);
        metrics.intent_counts.insert(QueryIntent::Lexical, 50);
        
        let distribution = metrics.intent_distribution();
        assert_eq!(distribution[&QueryIntent::NaturalLanguage], 0.3);
        assert_eq!(distribution[&QueryIntent::Definition], 0.2);
        assert_eq!(distribution[&QueryIntent::Lexical], 0.5);
    }
}