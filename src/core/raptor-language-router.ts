/**
 * Per-Language RAPTOR Card Routing System
 * 
 * Enables language-specific RAPTOR feature cards for enhanced code understanding:
 * - TypeScript/JavaScript: React patterns, async/await, type narrowing
 * - Python: List comprehensions, decorators, asyncio patterns  
 * - Rust: Ownership patterns, trait implementations, error handling
 * - Go: Goroutines, channels, interface satisfaction
 * - Java: Streams, annotations, generics patterns
 */

export interface LanguageRaptorConfig {
  language: string;
  enabled: boolean;
  feature_cards: RaptorFeatureCard[];
  routing_priority: number;
  language_confidence_threshold: number;
}

export interface RaptorFeatureCard {
  card_id: string;
  card_name: string;
  description: string;
  patterns: LanguagePattern[];
  weight: number;
  context_window: number;
}

export interface LanguagePattern {
  pattern_type: 'syntax' | 'semantic' | 'idiom' | 'anti_pattern';
  pattern_regex: string;
  ast_node_types?: string[];
  context_keywords: string[];
  examples: string[];
  importance_weight: number;
}

export interface LanguageDetectionResult {
  detected_language: string;
  confidence: number;
  file_extension: string;
  syntax_indicators: string[];
  fallback_language?: string;
}

export interface RaptorRoutingResult {
  original_query: string;
  detected_language: string;
  selected_cards: RaptorFeatureCard[];
  routing_confidence: number;
  enhanced_features: EnhancedFeature[];
  fallback_applied: boolean;
}

export interface EnhancedFeature {
  feature_name: string;
  feature_value: number;
  source_card: string;
  pattern_matches: PatternMatch[];
}

export interface PatternMatch {
  pattern_id: string;
  match_text: string;
  match_confidence: number;
  context: string;
}

export class RaptorLanguageRouter {
  private languageConfigs: Map<string, LanguageRaptorConfig> = new Map();
  
  constructor() {
    this.initializeLanguageConfigs();
  }
  
  /**
   * Route query through language-specific RAPTOR cards
   */
  public async routeQuery(
    query: string,
    filePath: string,
    codeContext?: string
  ): Promise<RaptorRoutingResult> {
    console.log(`🎯 Routing RAPTOR query for: ${filePath}`);
    
    try {
      // Step 1: Detect language from file and context
      const languageDetection = await this.detectLanguage(filePath, codeContext);
      
      // Step 2: Get language-specific RAPTOR configuration
      const languageConfig = this.languageConfigs.get(languageDetection.detected_language);
      
      if (!languageConfig || !languageConfig.enabled) {
        console.log(`⚠️  No RAPTOR config for language: ${languageDetection.detected_language}`);
        return this.createFallbackResult(query, languageDetection.detected_language);
      }
      
      // Step 3: Select relevant feature cards based on query
      const selectedCards = await this.selectFeatureCards(query, languageConfig, codeContext);
      
      // Step 4: Extract enhanced features using selected cards
      const enhancedFeatures = await this.extractEnhancedFeatures(query, selectedCards, codeContext);
      
      // Step 5: Calculate routing confidence
      const routingConfidence = this.calculateRoutingConfidence(languageDetection, selectedCards);
      
      console.log(`✨ RAPTOR routing: ${languageDetection.detected_language} (${selectedCards.length} cards, confidence: ${routingConfidence.toFixed(2)})`);
      
      return {
        original_query: query,
        detected_language: languageDetection.detected_language,
        selected_cards: selectedCards,
        routing_confidence: routingConfidence,
        enhanced_features: enhancedFeatures,
        fallback_applied: false
      };
      
    } catch (error) {
      console.error('❌ RAPTOR routing failed:', error);
      return this.createFallbackResult(query, 'unknown');
    }
  }
  
  /**
   * Detect language from file path and code context
   */
  private async detectLanguage(filePath: string, codeContext?: string): Promise<LanguageDetectionResult> {
    const fileExtension = filePath.split('.').pop()?.toLowerCase() || '';
    const syntaxIndicators: string[] = [];
    
    // Primary detection via file extension
    let detectedLanguage = 'unknown';
    let confidence = 0.6; // Base confidence for extension-based detection
    
    const extensionMap: Record<string, string> = {
      'ts': 'typescript',
      'tsx': 'typescript',
      'js': 'javascript', 
      'jsx': 'javascript',
      'py': 'python',
      'rs': 'rust',
      'go': 'go',
      'java': 'java',
      'kt': 'kotlin',
      'swift': 'swift',
      'cpp': 'cpp',
      'c': 'c',
      'cs': 'csharp',
      'rb': 'ruby',
      'php': 'php'
    };
    
    if (extensionMap[fileExtension]) {
      detectedLanguage = extensionMap[fileExtension];
    }
    
    // Enhanced detection using code context
    if (codeContext) {
      const contextIndicators = this.analyzeCodeContext(codeContext);
      syntaxIndicators.push(...contextIndicators.indicators);
      
      // Boost confidence if context matches extension
      if (contextIndicators.language === detectedLanguage) {
        confidence = Math.min(confidence + 0.3, 0.95);
      }
      // Override if context is very confident and different
      else if (contextIndicators.confidence > 0.8) {
        detectedLanguage = contextIndicators.language;
        confidence = contextIndicators.confidence;
      }
    }
    
    return {
      detected_language: detectedLanguage,
      confidence: confidence,
      file_extension: fileExtension,
      syntax_indicators: syntaxIndicators,
      fallback_language: detectedLanguage === 'unknown' ? 'typescript' : undefined
    };
  }
  
  /**
   * Analyze code context for language indicators
   */
  private analyzeCodeContext(codeContext: string): { language: string; confidence: number; indicators: string[] } {
    const indicators: string[] = [];
    let language = 'unknown';
    let confidence = 0.5;
    
    // TypeScript/JavaScript patterns
    if (/\b(interface|type\s+\w+\s*=|as\s+\w+|\?\.|Promise<|async\s+function|\bconst\b)/i.test(codeContext)) {
      indicators.push('typescript_syntax');
      if (/\binterface\s+\w+|type\s+\w+\s*=/.test(codeContext)) {
        language = 'typescript';
        confidence = 0.85;
      } else {
        language = 'javascript';  
        confidence = 0.75;
      }
    }
    
    // Python patterns
    else if (/\b(def\s+\w+|import\s+\w+|from\s+\w+\s+import|\bself\b|:\s*\n\s+|\[.*for.*in.*\])/i.test(codeContext)) {
      indicators.push('python_syntax');
      language = 'python';
      confidence = 0.8;
    }
    
    // Rust patterns
    else if (/\b(fn\s+\w+|let\s+mut|impl\s+\w+|match\s+\w+|\bSome\(|\bNone\b|&str|&mut)/i.test(codeContext)) {
      indicators.push('rust_syntax');
      language = 'rust';
      confidence = 0.9;
    }
    
    // Go patterns
    else if (/\b(func\s+\w+|package\s+\w+|go\s+func|chan\s+\w+|\bgofmt\b|:=)/i.test(codeContext)) {
      indicators.push('go_syntax');
      language = 'go';
      confidence = 0.85;
    }
    
    // Java patterns
    else if (/\b(public\s+class|private\s+\w+|\bextends\b|\bimplements\b|@\w+|System\.out)/i.test(codeContext)) {
      indicators.push('java_syntax');
      language = 'java';
      confidence = 0.8;
    }
    
    return { language, confidence, indicators };
  }
  
  /**
   * Select relevant feature cards for the query and language
   */
  private async selectFeatureCards(
    query: string,
    languageConfig: LanguageRaptorConfig,
    codeContext?: string
  ): Promise<RaptorFeatureCard[]> {
    const queryLower = query.toLowerCase();
    const contextLower = codeContext?.toLowerCase() || '';
    const selectedCards: Array<{ card: RaptorFeatureCard; relevanceScore: number }> = [];
    
    for (const card of languageConfig.feature_cards) {
      let relevanceScore = 0;
      
      // Check if query mentions card-specific patterns
      for (const pattern of card.patterns) {
        // Keyword matching
        const keywordMatches = pattern.context_keywords.filter(keyword => 
          queryLower.includes(keyword) || contextLower.includes(keyword)
        );
        relevanceScore += keywordMatches.length * pattern.importance_weight;
        
        // Pattern matching in context
        if (codeContext && pattern.pattern_regex) {
          try {
            const regex = new RegExp(pattern.pattern_regex, 'i');
            if (regex.test(codeContext)) {
              relevanceScore += pattern.importance_weight * 2; // Higher weight for actual pattern matches
            }
          } catch (e) {
            // Invalid regex, skip
          }
        }
      }
      
      // Apply card base weight
      relevanceScore *= card.weight;
      
      if (relevanceScore > 0) {
        selectedCards.push({ card, relevanceScore });
      }
    }
    
    // Sort by relevance and take top cards
    selectedCards.sort((a, b) => b.relevanceScore - a.relevanceScore);
    const maxCards = Math.min(selectedCards.length, 5); // Limit to 5 cards per query
    
    return selectedCards.slice(0, maxCards).map(sc => sc.card);
  }
  
  /**
   * Extract enhanced features using selected RAPTOR cards
   */
  private async extractEnhancedFeatures(
    query: string,
    selectedCards: RaptorFeatureCard[],
    codeContext?: string
  ): Promise<EnhancedFeature[]> {
    const features: EnhancedFeature[] = [];
    
    for (const card of selectedCards) {
      const cardFeatures = await this.extractCardFeatures(query, card, codeContext);
      features.push(...cardFeatures);
    }
    
    return features;
  }
  
  /**
   * Extract features from a specific RAPTOR card
   */
  private async extractCardFeatures(
    query: string,
    card: RaptorFeatureCard,
    codeContext?: string
  ): Promise<EnhancedFeature[]> {
    const features: EnhancedFeature[] = [];
    
    for (const pattern of card.patterns) {
      const patternMatches = await this.findPatternMatches(pattern, query, codeContext);
      
      if (patternMatches.length > 0) {
        // Calculate feature value based on matches
        const featureValue = patternMatches.reduce((sum, match) => sum + match.match_confidence, 0) / patternMatches.length;
        
        features.push({
          feature_name: `${card.card_id}_${pattern.pattern_type}`,
          feature_value: featureValue,
          source_card: card.card_id,
          pattern_matches: patternMatches
        });
      }
    }
    
    return features;
  }
  
  /**
   * Find pattern matches in query and code context
   */
  private async findPatternMatches(
    pattern: LanguagePattern,
    query: string,
    codeContext?: string
  ): Promise<PatternMatch[]> {
    const matches: PatternMatch[] = [];
    
    // Check query for keyword matches
    const queryLower = query.toLowerCase();
    for (const keyword of pattern.context_keywords) {
      if (queryLower.includes(keyword)) {
        matches.push({
          pattern_id: `${pattern.pattern_type}_${keyword}`,
          match_text: keyword,
          match_confidence: 0.7,
          context: 'query'
        });
      }
    }
    
    // Check code context for regex pattern matches
    if (codeContext && pattern.pattern_regex) {
      try {
        const regex = new RegExp(pattern.pattern_regex, 'gi');
        const regexMatches = [...codeContext.matchAll(regex)];
        
        for (const match of regexMatches) {
          matches.push({
            pattern_id: `${pattern.pattern_type}_regex`,
            match_text: match[0],
            match_confidence: 0.9,
            context: this.extractContext(codeContext, match.index || 0, 50)
          });
        }
      } catch (e) {
        // Invalid regex, skip
      }
    }
    
    return matches;
  }
  
  /**
   * Calculate routing confidence based on detection and card selection
   */
  private calculateRoutingConfidence(
    detection: LanguageDetectionResult,
    selectedCards: RaptorFeatureCard[]
  ): number {
    let confidence = detection.confidence;
    
    // Boost confidence based on card selection success
    if (selectedCards.length > 0) {
      const cardBoost = Math.min(selectedCards.length * 0.1, 0.2);
      confidence = Math.min(confidence + cardBoost, 0.95);
    }
    
    // Penalty for fallback language use
    if (detection.fallback_language) {
      confidence *= 0.8;
    }
    
    return confidence;
  }
  
  /**
   * Create fallback result when routing fails
   */
  private createFallbackResult(query: string, language: string): RaptorRoutingResult {
    return {
      original_query: query,
      detected_language: language,
      selected_cards: [],
      routing_confidence: 0.1,
      enhanced_features: [],
      fallback_applied: true
    };
  }
  
  /**
   * Extract context around a match position
   */
  private extractContext(text: string, position: number, windowSize: number): string {
    const start = Math.max(0, position - windowSize);
    const end = Math.min(text.length, position + windowSize);
    return text.substring(start, end).trim();
  }
  
  /**
   * Initialize language-specific RAPTOR configurations
   */
  private initializeLanguageConfigs(): void {
    // TypeScript/JavaScript configuration
    this.languageConfigs.set('typescript', {
      language: 'typescript',
      enabled: true,
      routing_priority: 1,
      language_confidence_threshold: 0.7,
      feature_cards: [
        {
          card_id: 'typescript_types',
          card_name: 'TypeScript Type System',
          description: 'Advanced TypeScript type patterns and narrowing',
          weight: 1.2,
          context_window: 100,
          patterns: [
            {
              pattern_type: 'syntax',
              pattern_regex: '\\binterface\\s+\\w+|type\\s+\\w+\\s*=|as\\s+\\w+',
              context_keywords: ['interface', 'type', 'generic', 'union', 'intersection', 'narrowing'],
              examples: ['interface User', 'type Status = "pending" | "complete"'],
              importance_weight: 1.0
            },
            {
              pattern_type: 'idiom',
              pattern_regex: '\\?\\.|\\!\\.|\\?\\?',
              context_keywords: ['optional', 'chaining', 'nullish', 'coalescing'],
              examples: ['user?.name', 'value ?? default'],
              importance_weight: 0.8
            }
          ]
        },
        {
          card_id: 'react_patterns',
          card_name: 'React Component Patterns',
          description: 'React hooks, components, and state management',
          weight: 1.1,
          context_window: 150,
          patterns: [
            {
              pattern_type: 'idiom',
              pattern_regex: 'use\\w+\\(|useState|useEffect|useContext',
              context_keywords: ['hook', 'component', 'state', 'effect', 'context', 'props'],
              examples: ['useState()', 'useEffect(() => {})'],
              importance_weight: 1.0
            }
          ]
        }
      ]
    });
    
    // Python configuration
    this.languageConfigs.set('python', {
      language: 'python',
      enabled: true,
      routing_priority: 1,
      language_confidence_threshold: 0.7,
      feature_cards: [
        {
          card_id: 'python_comprehensions',
          card_name: 'Python Comprehensions and Iterators',
          description: 'List/dict comprehensions, generators, and iteration patterns',
          weight: 1.1,
          context_window: 80,
          patterns: [
            {
              pattern_type: 'syntax',
              pattern_regex: '\\[.+for.+in.+\\]|\\{.+for.+in.+\\}|\\(.+for.+in.+\\)',
              context_keywords: ['comprehension', 'generator', 'iterator', 'enumerate', 'zip'],
              examples: ['[x for x in items]', '{k: v for k, v in dict.items()}'],
              importance_weight: 1.0
            }
          ]
        },
        {
          card_id: 'python_async',
          card_name: 'Python Async/Await Patterns',
          description: 'Asyncio, coroutines, and concurrent programming',
          weight: 1.2,
          context_window: 120,
          patterns: [
            {
              pattern_type: 'syntax',
              pattern_regex: '\\basync\\s+def|await\\s+\\w+|asyncio\\.',
              context_keywords: ['async', 'await', 'asyncio', 'coroutine', 'task', 'gather'],
              examples: ['async def func():', 'await some_async_call()'],
              importance_weight: 1.1
            }
          ]
        }
      ]
    });
    
    // Rust configuration
    this.languageConfigs.set('rust', {
      language: 'rust',
      enabled: true,
      routing_priority: 1,
      language_confidence_threshold: 0.8,
      feature_cards: [
        {
          card_id: 'rust_ownership',
          card_name: 'Rust Ownership and Borrowing',
          description: 'Ownership, borrowing, and lifetime patterns',
          weight: 1.3,
          context_window: 100,
          patterns: [
            {
              pattern_type: 'syntax',
              pattern_regex: '&mut\\s+\\w+|&\\w+|\'\\w+|Box<|Rc<|Arc<',
              context_keywords: ['borrow', 'ownership', 'lifetime', 'reference', 'move', 'clone'],
              examples: ['&mut self', '&str', "'a"],
              importance_weight: 1.2
            }
          ]
        },
        {
          card_id: 'rust_error_handling',
          card_name: 'Rust Error Handling',
          description: 'Result, Option, and error propagation patterns',
          weight: 1.2,
          context_window: 90,
          patterns: [
            {
              pattern_type: 'idiom',
              pattern_regex: 'Result<.+>|Option<.+>|\\?|unwrap\\(\\)|expect\\(',
              context_keywords: ['result', 'option', 'error', 'unwrap', 'expect', 'match'],
              examples: ['Result<T, E>', 'Some(value)', 'value?'],
              importance_weight: 1.1
            }
          ]
        }
      ]
    });
    
    // Go configuration  
    this.languageConfigs.set('go', {
      language: 'go',
      enabled: true,
      routing_priority: 1,
      language_confidence_threshold: 0.75,
      feature_cards: [
        {
          card_id: 'go_concurrency',
          card_name: 'Go Concurrency Patterns',
          description: 'Goroutines, channels, and concurrent programming',
          weight: 1.3,
          context_window: 100,
          patterns: [
            {
              pattern_type: 'syntax',
              pattern_regex: '\\bgo\\s+\\w+|\\bgoroutine\\b|\\bchan\\b|<-\\s*\\w+|\\w+\\s*<-',
              context_keywords: ['goroutine', 'channel', 'select', 'concurrent', 'mutex', 'waitgroup'],
              examples: ['go func()', 'chan int', 'ch <- value'],
              importance_weight: 1.2
            }
          ]
        }
      ]
    });
    
    // Java configuration
    this.languageConfigs.set('java', {
      language: 'java',
      enabled: true,
      routing_priority: 1,
      language_confidence_threshold: 0.7,
      feature_cards: [
        {
          card_id: 'java_streams',
          card_name: 'Java Streams and Functional Programming',
          description: 'Stream API, lambdas, and functional patterns',
          weight: 1.1,
          context_window: 120,
          patterns: [
            {
              pattern_type: 'idiom',
              pattern_regex: '\\.stream\\(\\)|\\.filter\\(|\\.map\\(|\\.collect\\(|->',
              context_keywords: ['stream', 'lambda', 'filter', 'map', 'collect', 'optional'],
              examples: ['list.stream().filter(x -> x > 0)', 'Optional.of(value)'],
              importance_weight: 1.0
            }
          ]
        }
      ]
    });
    
    console.log(`✅ Initialized RAPTOR language routing for ${this.languageConfigs.size} languages`);
  }
  
  /**
   * Get available languages with RAPTOR support
   */
  public getSupportedLanguages(): string[] {
    return Array.from(this.languageConfigs.keys());
  }
  
  /**
   * Update language configuration
   */
  public updateLanguageConfig(language: string, config: Partial<LanguageRaptorConfig>): void {
    const existingConfig = this.languageConfigs.get(language);
    if (existingConfig) {
      this.languageConfigs.set(language, { ...existingConfig, ...config });
      console.log(`🔧 Updated RAPTOR config for ${language}`);
    }
  }
  
  /**
   * Enable/disable language routing
   */
  public setLanguageEnabled(language: string, enabled: boolean): void {
    const config = this.languageConfigs.get(language);
    if (config) {
      config.enabled = enabled;
      console.log(`${enabled ? '✅' : '❌'} RAPTOR routing ${enabled ? 'enabled' : 'disabled'} for ${language}`);
    }
  }
}

export const raptorLanguageRouter = new RaptorLanguageRouter();