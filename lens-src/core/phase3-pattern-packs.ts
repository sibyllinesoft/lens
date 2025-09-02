/**
 * Phase 3 Pattern Packs for Stage B Symbol/AST Coverage Expansion
 * Implements ctor_impl, test_func_names, and config_keys pattern recognition
 */

import { LensTracer } from '../telemetry/tracer.js';

export interface PatternPack {
  name: string;
  description: string;
  patterns: PatternDefinition[];
  languages: string[];
  priority: number;
}

export interface PatternDefinition {
  name: string;
  regex: RegExp;
  astNodeTypes: string[];
  symbolKinds: string[];
  contextRequired: boolean;
  examples: string[];
}

export interface PatternMatch {
  pattern_name: string;
  file_path: string;
  line: number;
  col: number;
  match_text: string;
  symbol_kind: string;
  ast_context?: string;
  confidence: number;
}

/**
 * Constructor Implementation Pattern Pack
 * Detects constructor implementations across different languages
 */
export const CtorImplPatternPack: PatternPack = {
  name: "ctor_impl",
  description: "Constructor implementations and initialization patterns",
  languages: ["typescript", "javascript", "python", "java", "rust", "go"],
  priority: 8,
  patterns: [
    {
      name: "typescript_constructor",
      regex: /constructor\s*\([^)]*\)\s*{/g,
      astNodeTypes: ["MethodDefinition", "Constructor"],
      symbolKinds: ["constructor", "method"],
      contextRequired: true,
      examples: [
        "constructor(name: string) {",
        "constructor() {",
        "constructor(private config: Config) {"
      ]
    },
    {
      name: "python_init",
      regex: /def\s+__init__\s*\([^)]*\):/g,
      astNodeTypes: ["FunctionDef", "AsyncFunctionDef"],
      symbolKinds: ["constructor", "method"],
      contextRequired: true,
      examples: [
        "def __init__(self):",
        "def __init__(self, name: str):",
        "async def __init__(self, config):"
      ]
    },
    {
      name: "rust_new_impl",
      regex: /impl.*{\s*(?:pub\s+)?fn\s+new\s*\([^)]*\)/g,
      astNodeTypes: ["ImplItem", "Function"],
      symbolKinds: ["function", "constructor"],
      contextRequired: true,
      examples: [
        "impl MyStruct {\n    fn new() -> Self {",
        "impl<T> MyStruct<T> {\n    pub fn new(value: T) -> Self {"
      ]
    },
    {
      name: "go_constructor_func",
      regex: /func\s+New[A-Z]\w*\s*\([^)]*\)\s*[*]?\w+/g,
      astNodeTypes: ["FuncDecl"],
      symbolKinds: ["function", "constructor"],
      contextRequired: false,
      examples: [
        "func NewUser(name string) *User {",
        "func NewConfig() Config {"
      ]
    },
    {
      name: "java_constructor",
      regex: /(?:public|private|protected)?\s*[A-Z]\w*\s*\([^)]*\)\s*{/g,
      astNodeTypes: ["ConstructorDeclaration"],
      symbolKinds: ["constructor"],
      contextRequired: true,
      examples: [
        "public User(String name) {",
        "private Config() {",
        "User(String name, int age) {"
      ]
    }
  ]
};

/**
 * Test Function Names Pattern Pack
 * Detects test functions and testing patterns
 */
export const TestFuncNamesPatternPack: PatternPack = {
  name: "test_func_names",
  description: "Test function patterns and testing frameworks",
  languages: ["typescript", "javascript", "python", "rust", "go", "java"],
  priority: 7,
  patterns: [
    {
      name: "jest_test_functions",
      regex: /(?:test|it|describe)\s*\(\s*['"`]([^'"`]+)['"`]/g,
      astNodeTypes: ["CallExpression"],
      symbolKinds: ["function"],
      contextRequired: false,
      examples: [
        'test("should create user", () => {',
        'it("validates input", async () => {',
        'describe("UserService", () => {'
      ]
    },
    {
      name: "python_unittest",
      regex: /def\s+test_\w+\s*\([^)]*\):/g,
      astNodeTypes: ["FunctionDef"],
      symbolKinds: ["method", "function"],
      contextRequired: false,
      examples: [
        "def test_user_creation(self):",
        "def test_config_validation():",
        "async def test_api_endpoint(self):"
      ]
    },
    {
      name: "python_pytest",
      regex: /def\s+test_\w+\s*\([^)]*\):|@pytest\.(mark\.)?\w+/g,
      astNodeTypes: ["FunctionDef", "Decorator"],
      symbolKinds: ["function"],
      contextRequired: false,
      examples: [
        "def test_user_creation():",
        "@pytest.fixture",
        "@pytest.mark.asyncio"
      ]
    },
    {
      name: "rust_test_functions",
      regex: /#\[test\]\s*(?:async\s+)?fn\s+\w+/g,
      astNodeTypes: ["Function", "Attribute"],
      symbolKinds: ["function"],
      contextRequired: false,
      examples: [
        "#[test]\nfn test_user_creation() {",
        "#[tokio::test]\nasync fn test_async_operation() {"
      ]
    },
    {
      name: "go_test_functions",
      regex: /func\s+Test\w+\s*\(\s*t\s+\*testing\.T\s*\)/g,
      astNodeTypes: ["FuncDecl"],
      symbolKinds: ["function"],
      contextRequired: false,
      examples: [
        "func TestUserCreation(t *testing.T) {",
        "func TestConfigValidation(t *testing.T) {"
      ]
    },
    {
      name: "junit_test_methods",
      regex: /@Test\s+(?:public\s+)?(?:void\s+)?\w+\s*\([^)]*\)/g,
      astNodeTypes: ["MethodDeclaration", "Annotation"],
      symbolKinds: ["method"],
      contextRequired: false,
      examples: [
        "@Test\npublic void testUserCreation() {",
        "@Test\nvoid shouldValidateInput() {"
      ]
    }
  ]
};

/**
 * Configuration Keys Pattern Pack  
 * Detects configuration keys, environment variables, and settings
 */
export const ConfigKeysPatternPack: PatternPack = {
  name: "config_keys",
  description: "Configuration keys, environment variables, and settings patterns",
  languages: ["typescript", "javascript", "python", "rust", "go", "java", "yaml", "json"],
  priority: 6,
  patterns: [
    {
      name: "env_var_access",
      regex: /process\.env\.([A-Z_][A-Z0-9_]*)|os\.environ\.get\(['"']([A-Z_][A-Z0-9_]*)['"']/g,
      astNodeTypes: ["MemberExpression", "CallExpression"],
      symbolKinds: ["variable", "property"],
      contextRequired: false,
      examples: [
        "process.env.DATABASE_URL",
        "process.env.API_KEY",
        'os.environ.get("DATABASE_URL")'
      ]
    },
    {
      name: "config_object_keys",
      regex: /(?:config|settings|options)\.([a-zA-Z_][a-zA-Z0-9_]*)/g,
      astNodeTypes: ["MemberExpression"],
      symbolKinds: ["property"],
      contextRequired: false,
      examples: [
        "config.database_url",
        "settings.api_timeout",
        "options.retry_count"
      ]
    },
    {
      name: "yaml_config_keys",
      regex: /^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:/gm,
      astNodeTypes: ["Property"],
      symbolKinds: ["property"],
      contextRequired: false,
      examples: [
        "database_url:",
        "api_timeout: 5000",
        "  retry_count: 3"
      ]
    },
    {
      name: "json_config_keys",
      regex: /"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:/g,
      astNodeTypes: ["Property"],
      symbolKinds: ["property"],
      contextRequired: false,
      examples: [
        '"database_url": "...",',
        '"api_timeout": 5000,',
        '"retry_count": 3'
      ]
    },
    {
      name: "rust_config_struct",
      regex: /(?:pub\s+)?(\w+):\s*(?:String|&str|i32|u32|f64|bool)/g,
      astNodeTypes: ["Field"],
      symbolKinds: ["field", "property"],
      contextRequired: true,
      examples: [
        "pub database_url: String,",
        "api_timeout: u32,",
        "retry_count: i32,"
      ]
    },
    {
      name: "go_config_struct",
      regex: /(\w+)\s+(?:string|int|int64|float64|bool)\s+`[^`]*`/g,
      astNodeTypes: ["Field"],
      symbolKinds: ["field"],
      contextRequired: true,
      examples: [
        'DatabaseURL string `env:"DATABASE_URL"`',
        'APITimeout int `yaml:"api_timeout"`',
        'RetryCount int `json:"retry_count"`'
      ]
    }
  ]
};

/**
 * Phase 3 Pattern Pack Engine
 * Orchestrates pattern matching for enhanced symbol coverage
 */
export class Phase3PatternPackEngine {
  private patterns: Map<string, PatternPack> = new Map();
  
  constructor() {
    // Register built-in pattern packs
    this.registerPatternPack(CtorImplPatternPack);
    this.registerPatternPack(TestFuncNamesPatternPack);
    this.registerPatternPack(ConfigKeysPatternPack);
  }

  /**
   * Register a new pattern pack
   */
  registerPatternPack(pack: PatternPack): void {
    this.patterns.set(pack.name, pack);
    console.log(`📦 Registered pattern pack: ${pack.name} (${pack.patterns.length} patterns)`);
  }

  /**
   * Get all registered pattern packs
   */
  getPatternPacks(): PatternPack[] {
    return Array.from(this.patterns.values());
  }

  /**
   * Get pattern pack by name
   */
  getPatternPack(name: string): PatternPack | undefined {
    return this.patterns.get(name);
  }

  /**
   * Find patterns in source code
   */
  async findPatterns(
    sourceCode: string, 
    filePath: string, 
    language: string,
    patternNames?: string[]
  ): Promise<PatternMatch[]> {
    const span = LensTracer.createChildSpan('find_patterns');
    
    try {
      const matches: PatternMatch[] = [];
      const lines = sourceCode.split('\n');
      
      // Determine which patterns to use
      const patternsToUse = patternNames 
        ? patternNames.map(name => this.patterns.get(name)).filter(p => p) as PatternPack[]
        : this.getPatternPacks().filter(pack => pack.languages.includes(language));
      
      for (const pack of patternsToUse) {
        for (const pattern of pack.patterns) {
          // Reset regex state
          pattern.regex.lastIndex = 0;
          
          let match;
          while ((match = pattern.regex.exec(sourceCode)) !== null) {
            // Calculate line and column
            const beforeMatch = sourceCode.substring(0, match.index);
            const lineNumber = beforeMatch.split('\n').length;
            const lastNewlineIndex = beforeMatch.lastIndexOf('\n');
            const columnNumber = match.index - lastNewlineIndex;
            
            // Extract context if required
            let astContext: string | undefined;
            if (pattern.contextRequired && lines[lineNumber - 1]) {
              const contextStart = Math.max(0, lineNumber - 3);
              const contextEnd = Math.min(lines.length, lineNumber + 2);
              astContext = lines.slice(contextStart, contextEnd).join('\n');
            }
            
            // Determine symbol kind
            const symbolKind = pattern.symbolKinds[0] || 'unknown';
            
            // Calculate confidence based on context and pattern specificity
            let confidence = 0.8; // Base confidence
            if (pattern.contextRequired && astContext) confidence += 0.1;
            if (pattern.astNodeTypes.length > 0) confidence += 0.1;
            
            matches.push({
              pattern_name: pattern.name,
              file_path: filePath,
              line: lineNumber,
              col: columnNumber,
              match_text: match[0],
              symbol_kind: symbolKind,
              ast_context: astContext,
              confidence: Math.min(1.0, confidence),
            });
          }
        }
      }
      
      // Sort by confidence and line number
      matches.sort((a, b) => {
        if (a.confidence !== b.confidence) {
          return b.confidence - a.confidence;
        }
        return a.line - b.line;
      });
      
      span.setAttributes({
        success: true,
        file_path: filePath,
        language,
        patterns_used: patternsToUse.length,
        matches_found: matches.length,
      });
      
      return matches;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ 
        success: false, 
        error: errorMsg,
        file_path: filePath,
        language 
      });
      throw new Error(`Pattern matching failed: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Get pattern statistics
   */
  getStatistics(): {
    total_packs: number;
    total_patterns: number;
    languages_supported: string[];
    patterns_by_language: Record<string, number>;
  } {
    const packs = this.getPatternPacks();
    const allLanguages = new Set<string>();
    const patternsByLanguage: Record<string, number> = {};
    let totalPatterns = 0;
    
    for (const pack of packs) {
      totalPatterns += pack.patterns.length;
      
      for (const language of pack.languages) {
        allLanguages.add(language);
        patternsByLanguage[language] = (patternsByLanguage[language] || 0) + pack.patterns.length;
      }
    }
    
    return {
      total_packs: packs.length,
      total_patterns: totalPatterns,
      languages_supported: Array.from(allLanguages).sort(),
      patterns_by_language: patternsByLanguage,
    };
  }

  /**
   * Validate pattern pack configuration
   */
  validatePatternPack(pack: PatternPack): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    if (!pack.name || typeof pack.name !== 'string') {
      errors.push('Pattern pack name is required and must be a string');
    }
    
    if (!pack.patterns || !Array.isArray(pack.patterns) || pack.patterns.length === 0) {
      errors.push('Pattern pack must have at least one pattern');
    }
    
    if (!pack.languages || !Array.isArray(pack.languages) || pack.languages.length === 0) {
      errors.push('Pattern pack must support at least one language');
    }
    
    if (typeof pack.priority !== 'number' || pack.priority < 1 || pack.priority > 10) {
      errors.push('Pattern pack priority must be a number between 1 and 10');
    }
    
    // Validate individual patterns
    for (let i = 0; i < (pack.patterns?.length || 0); i++) {
      const pattern = pack.patterns[i];
      
      if (!pattern.name || typeof pattern.name !== 'string') {
        errors.push(`Pattern ${i}: name is required and must be a string`);
      }
      
      if (!pattern.regex || !(pattern.regex instanceof RegExp)) {
        errors.push(`Pattern ${i}: regex is required and must be a RegExp`);
      }
      
      if (!pattern.symbolKinds || !Array.isArray(pattern.symbolKinds) || pattern.symbolKinds.length === 0) {
        errors.push(`Pattern ${i}: must have at least one symbol kind`);
      }
      
      if (!pattern.examples || !Array.isArray(pattern.examples) || pattern.examples.length === 0) {
        errors.push(`Pattern ${i}: must have at least one example`);
      }
    }
    
    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Export pattern pack definitions for external use
   */
  exportPatternPacks(): Record<string, PatternPack> {
    const result: Record<string, PatternPack> = {};
    
    for (const [name, pack] of this.patterns) {
      // Clone the pattern pack to avoid external modifications
      result[name] = {
        ...pack,
        patterns: pack.patterns.map(p => ({ ...p })),
      };
    }
    
    return result;
  }
}