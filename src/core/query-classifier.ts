/**
 * Query Classification for Stage-C Semantic Reranking
 * Determines whether to apply semantic reranking based on query type
 */

export interface QueryClassification {
  isNaturalLanguage: boolean;
  confidence: number;
  characteristics: QueryCharacteristic[];
}

export type QueryCharacteristic = 
  | 'has_articles'           // "the", "a", "an"
  | 'has_prepositions'       // "for", "in", "with", "to"
  | 'has_descriptive_words'  // "find", "show", "get", "search"
  | 'has_multiple_words'     // More than 2-3 words
  | 'has_operators'          // "def", "class", regex patterns
  | 'has_symbols'            // Punctuation, operators
  | 'has_programming_syntax' // camelCase, snake_case, brackets
  | 'has_questions'          // "what", "how", "where", "why"

/**
 * Classify query to determine if semantic reranking should be applied
 */
export function classifyQuery(query: string): QueryClassification {
  const normalizedQuery = query.toLowerCase().trim();
  const words = normalizedQuery.split(/\s+/).filter(word => word.length > 0);
  
  const characteristics: QueryCharacteristic[] = [];
  let naturalLanguageScore = 0;
  
  // Check for articles (strong NL indicator)
  const articles = ['the', 'a', 'an'];
  if (words.some(word => articles.includes(word))) {
    characteristics.push('has_articles');
    naturalLanguageScore += 0.3;
  }
  
  // Check for prepositions (strong NL indicator)
  const prepositions = ['for', 'in', 'with', 'to', 'of', 'from', 'by', 'at', 'on'];
  if (words.some(word => prepositions.includes(word))) {
    characteristics.push('has_prepositions');
    naturalLanguageScore += 0.25;
  }
  
  // Check for descriptive action words (strong NL indicator)
  const descriptiveWords = [
    'find', 'search', 'show', 'get', 'fetch', 'retrieve', 'locate',
    'display', 'list', 'identify', 'discover', 'look', 'grab'
  ];
  if (words.some(word => descriptiveWords.includes(word))) {
    characteristics.push('has_descriptive_words');
    naturalLanguageScore += 0.35; // Increased weight
  }
  
  // Check for question words (strong NL indicator)
  const questionWords = ['what', 'how', 'where', 'when', 'why', 'which', 'who'];
  if (words.some(word => questionWords.includes(word))) {
    characteristics.push('has_questions');
    naturalLanguageScore += 0.3;
  }
  
  // Check length (moderate NL indicator for longer queries)  
  if (words.length > 3) {
    characteristics.push('has_multiple_words');
    naturalLanguageScore += Math.min(0.25, words.length * 0.06);
  }
  
  // Check for programming syntax (reduces NL score)
  const programmingPatterns = [
    /^(def|class|function|const|let|var|if|for|while)\s/i,
    /[(){}\[\]]/,
    /[a-z][A-Z]/, // camelCase
    /_[a-z]/, // snake_case
    /\w+\.\w+/, // dot notation
    /[=<>!]+/, // operators
    /^\w+\(/  // function calls
  ];
  
  if (programmingPatterns.some(pattern => pattern.test(query))) {
    characteristics.push('has_programming_syntax');
    naturalLanguageScore -= 0.4;
  }
  
  // Check for programming keywords (reduces NL score)
  const programmingKeywords = [
    'def', 'class', 'function', 'const', 'let', 'var', 'import', 'export',
    'if', 'else', 'for', 'while', 'try', 'catch', 'async', 'await',
    'return', 'yield', 'break', 'continue'
  ];
  if (words.some(word => programmingKeywords.includes(word))) {
    characteristics.push('has_operators');
    naturalLanguageScore -= 0.3;
  }
  
  // Check for symbols (reduces NL score)
  if (/[^a-zA-Z0-9\s]/.test(query)) {
    characteristics.push('has_symbols');
    naturalLanguageScore -= 0.2;
  }
  
  // Clamp score to [0, 1]
  const confidence = Math.max(0, Math.min(1, naturalLanguageScore));
  
  return {
    isNaturalLanguage: confidence > 0.5,
    confidence,
    characteristics
  };
}

/**
 * Should Stage-C semantic reranking be applied for this query and candidates?
 */
export function shouldApplySemanticReranking(
  query: string, 
  candidateCount: number,
  mode: string = 'hybrid',
  config?: { nlThreshold?: number; minCandidates?: number; maxCandidates?: number; confidenceCutoff?: number; forceSemanticForBenchmark?: boolean }
): boolean {
  // Only apply for hybrid mode
  if (mode !== 'hybrid') {
    return false;
  }
  
  // BENCHMARK OVERRIDE: Force semantic reranking for benchmark testing
  if (config?.forceSemanticForBenchmark) {
    // Still respect candidate count limits for performance
    const minCandidates = config?.minCandidates ?? 10;
    const maxCandidates = config?.maxCandidates ?? 200;
    
    if (candidateCount < minCandidates || candidateCount > maxCandidates) {
      return false;
    }
    
    return true; // Force semantic reranking for all benchmark queries
  }
  
  // Use configurable thresholds if provided, otherwise defaults
  const minCandidates = config?.minCandidates ?? 10;
  const maxCandidates = config?.maxCandidates ?? 200;
  const nlThreshold = config?.nlThreshold ?? 0.5;
  
  // Need sufficient candidates to rerank
  if (candidateCount < minCandidates) {
    return false;
  }
  
  // Don't apply for very large candidate sets (performance constraint)
  if (candidateCount > maxCandidates) {
    return false;
  }
  
  // Check if query is natural language with configurable threshold
  const classification = classifyQuery(query);
  
  // Apply confidence cutoff if specified
  if (config?.confidenceCutoff !== undefined && classification.confidence < config.confidenceCutoff) {
    return false;
  }
  
  // Apply semantic reranking for natural language queries using configurable threshold
  return classification.confidence > nlThreshold;
}

/**
 * Get human-readable explanation of why semantic reranking was/wasn't applied
 */
export function explainSemanticDecision(
  query: string,
  candidateCount: number,
  mode: string = 'hybrid'
): string {
  const classification = classifyQuery(query);
  
  if (mode !== 'hybrid') {
    return `Semantic reranking skipped: mode is '${mode}', requires 'hybrid'`;
  }
  
  if (candidateCount < 10) {
    return `Semantic reranking skipped: only ${candidateCount} candidates, need ≥10`;
  }
  
  if (candidateCount > 200) {
    return `Semantic reranking skipped: ${candidateCount} candidates exceed performance limit (200)`;
  }
  
  if (!classification.isNaturalLanguage) {
    const reasons = classification.characteristics
      .filter(c => ['has_operators', 'has_symbols', 'has_programming_syntax'].includes(c))
      .join(', ');
    return `Semantic reranking skipped: keyword query detected (${reasons || 'programming syntax'})`;
  }
  
  const nlReasons = classification.characteristics
    .filter(c => !['has_operators', 'has_symbols', 'has_programming_syntax'].includes(c))
    .join(', ');
  return `Semantic reranking applied: natural language query (${nlReasons})`;
}