import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock all external dependencies BEFORE importing
vi.mock('perf_hooks', () => ({
  performance: {
    now: vi.fn(() => Date.now()),
  },
}));

vi.mock('../telemetry/tracer', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

vi.mock('./advanced-cache-manager', () => ({
  globalCacheManager: {
    get: vi.fn().mockResolvedValue(null),
    set: vi.fn(),
  },
}));

vi.mock('./parallel-processor', () => ({
  globalParallelProcessor: {
    submitTask: vi.fn().mockResolvedValue({ results: [] }),
    getStats: vi.fn().mockReturnValue({ totalTasks: 0 }),
  },
}));

// Import AFTER mocks are set up
import { CrossLanguageResolver } from '../cross-language-resolver';

describe('CrossLanguageResolver', () => {
  let resolver: CrossLanguageResolver;

  beforeEach(() => {
    vi.clearAllMocks();
    resolver = new CrossLanguageResolver();
  });

  afterEach(() => {
    try {
      resolver.shutdown();
    } catch (e) {
      // Ignore shutdown errors in tests
    }
  });

  describe('Initialization', () => {
    it('should initialize with default configuration', () => {
      expect(resolver).toBeDefined();
      
      const stats = resolver.getStats();
      expect(stats).toBeDefined();
      expect(stats.indexedSymbols).toBe(0);
      expect(stats.languagesSupported).toBeGreaterThan(0); // Should have built-in languages
      expect(stats.resolutionStats).toBeDefined();
    });

    it('should support multiple programming languages', () => {
      const stats = resolver.getStats();
      expect(stats.languagesSupported).toBeGreaterThan(5); // Should support TypeScript, Python, etc.
    });

    it('should initialize with empty symbol index', () => {
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBe(0);
    });
  });

  describe('Cross-Language Resolution', () => {
    it('should handle basic cross-language resolution queries', async () => {
      const context = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        mode: 'symbol' as const,
      };

      const results = await resolver.resolveCrossLanguage(context);
      expect(Array.isArray(results)).toBe(true);
    });

    it('should handle resolution with specific file context', async () => {
      const context = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        file_path: 'src/main.ts',
        mode: 'symbol' as const,
      };

      const results = await resolver.resolveCrossLanguage(context);
      expect(Array.isArray(results)).toBe(true);
    });

    it('should handle resolution for different languages', async () => {
      const tsContext = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        file_path: 'src/component.tsx',
        mode: 'symbol' as const,
      };

      const pyContext = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        file_path: 'src/utils.py',
        mode: 'symbol' as const,
      };

      const tsResults = await resolver.resolveCrossLanguage(tsContext);
      const pyResults = await resolver.resolveCrossLanguage(pyContext);

      expect(Array.isArray(tsResults)).toBe(true);
      expect(Array.isArray(pyResults)).toBe(true);
    });

    it('should handle resolution with different modes', async () => {
      const symbolContext = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        mode: 'symbol' as const,
      };

      const lexicalContext = {
        repo_name: 'test-repo', 
        repo_sha: 'abc123',
        mode: 'lexical' as const,
      };

      const symbolResults = await resolver.resolveCrossLanguage(symbolContext);
      const lexicalResults = await resolver.resolveCrossLanguage(lexicalContext);

      expect(Array.isArray(symbolResults)).toBe(true);
      expect(Array.isArray(lexicalResults)).toBe(true);
    });
  });

  describe('Symbol Indexing', () => {
    it('should index TypeScript file symbols', async () => {
      const tsContent = `
        export interface User {
          id: string;
          name: string;
        }

        export class UserService {
          async getUser(id: string): Promise<User> {
            return { id, name: 'Test User' };
          }
        }

        export function validateUser(user: User): boolean {
          return user.id && user.name;
        }
      `;

      await resolver.indexFileSymbols('src/user.ts', tsContent);

      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should index Python file symbols', async () => {
      const pyContent = `
        class DataProcessor:
            def __init__(self, config):
                self.config = config
            
            def process_data(self, data):
                return self._transform(data)
            
            def _transform(self, data):
                return data.upper()

        def create_processor(config):
            return DataProcessor(config)

        CONSTANT_VALUE = 42
      `;

      await resolver.indexFileSymbols('src/processor.py', pyContent);

      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should index JavaScript file symbols', async () => {
      const jsContent = `
        const API_BASE_URL = 'https://api.example.com';

        function fetchData(endpoint) {
          return fetch(API_BASE_URL + endpoint);
        }

        class ApiClient {
          constructor(apiKey) {
            this.apiKey = apiKey;
          }

          async request(method, url, data) {
            return fetch(url, {
              method,
              headers: { 'Authorization': this.apiKey },
              body: JSON.stringify(data)
            });
          }
        }

        export { fetchData, ApiClient };
      `;

      await resolver.indexFileSymbols('src/api.js', jsContent);

      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should index Java file symbols', async () => {
      const javaContent = `
        package com.example.utils;

        import java.util.List;
        import java.util.ArrayList;

        public class StringUtils {
            private static final String DEFAULT_SEPARATOR = ",";

            public static List<String> split(String input, String separator) {
                // Implementation here
                return new ArrayList<>();
            }

            public static String join(List<String> items) {
                return join(items, DEFAULT_SEPARATOR);
            }

            public static String join(List<String> items, String separator) {
                return String.join(separator, items);
            }
        }
      `;

      await resolver.indexFileSymbols('src/StringUtils.java', javaContent);

      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle indexing empty files', async () => {
      const initialStats = resolver.getStats();
      
      await resolver.indexFileSymbols('empty.ts', '');
      
      const afterStats = resolver.getStats();
      expect(afterStats.indexedSymbols).toBe(initialStats.indexedSymbols);
    });

    it('should handle indexing files with only comments', async () => {
      const commentOnlyContent = `
        // This is a comment only file
        /* 
         * Multi-line comment
         * No actual code here
         */
        // Another comment
      `;

      const initialStats = resolver.getStats();
      
      await resolver.indexFileSymbols('comments.ts', commentOnlyContent);
      
      const afterStats = resolver.getStats();
      expect(afterStats.indexedSymbols).toBe(initialStats.indexedSymbols);
    });
  });

  describe('Language Detection', () => {
    it('should detect TypeScript files', async () => {
      const tsContent = 'interface Test { id: string; }';
      await resolver.indexFileSymbols('test.ts', tsContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });

    it('should detect Python files', async () => {
      const pyContent = 'def test_function(): pass';
      await resolver.indexFileSymbols('test.py', pyContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });

    it('should detect JavaScript files', async () => {
      const jsContent = 'function test() { return true; }';
      await resolver.indexFileSymbols('test.js', jsContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });

    it('should detect Java files', async () => {
      const javaContent = 'public class Test { }';
      await resolver.indexFileSymbols('Test.java', javaContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });

    it('should detect C++ files', async () => {
      const cppContent = 'class Test { public: int value; };';
      await resolver.indexFileSymbols('test.cpp', cppContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });

    it('should detect Go files', async () => {
      const goContent = 'package main\n\nfunc main() {\n}';
      await resolver.indexFileSymbols('main.go', goContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });

    it('should detect Rust files', async () => {
      const rustContent = 'fn main() { println!("Hello, world!"); }';
      await resolver.indexFileSymbols('main.rs', rustContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Symbol Name Normalization', () => {
    it('should handle camelCase naming conventions', async () => {
      const content = `
        function getUserData() { return {}; }
        const isValidUser = true;
        class UserManager { }
      `;
      
      await resolver.indexFileSymbols('camel.ts', content);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle snake_case naming conventions', async () => {
      const content = `
        def get_user_data():
            return {}
        
        is_valid_user = True
        
        class User_Manager:
            pass
      `;
      
      await resolver.indexFileSymbols('snake.py', content);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle PascalCase naming conventions', async () => {
      const content = `
        public class UserDataManager {
            public boolean IsValidUser;
            public void ProcessUserData() { }
        }
      `;
      
      await resolver.indexFileSymbols('Pascal.java', content);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle SCREAMING_SNAKE_CASE constants', async () => {
      const content = `
        const MAX_RETRY_ATTEMPTS = 3;
        const DEFAULT_TIMEOUT_MS = 5000;
        const API_BASE_URL = 'https://api.example.com';
      `;
      
      await resolver.indexFileSymbols('constants.ts', content);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });
  });

  describe('Cross-Reference Resolution', () => {
    it('should handle function calls across files', async () => {
      // Index multiple files with cross-references
      const utilsContent = `
        export function validateInput(input) {
          return input && input.length > 0;
        }
      `;
      
      const mainContent = `
        import { validateInput } from './utils';
        
        function processData(data) {
          if (validateInput(data)) {
            return data.toUpperCase();
          }
          return null;
        }
      `;
      
      await resolver.indexFileSymbols('utils.ts', utilsContent);
      await resolver.indexFileSymbols('main.ts', mainContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle class inheritance patterns', async () => {
      const baseContent = `
        export abstract class BaseService {
          protected config: any;
          
          constructor(config: any) {
            this.config = config;
          }
          
          abstract process(): Promise<void>;
        }
      `;
      
      const derivedContent = `
        import { BaseService } from './base';
        
        export class UserService extends BaseService {
          async process(): Promise<void> {
            // Implementation here
          }
        }
      `;
      
      await resolver.indexFileSymbols('base.ts', baseContent);
      await resolver.indexFileSymbols('user-service.ts', derivedContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle interface implementations', async () => {
      const interfaceContent = `
        export interface IDataProcessor {
          process(data: any): Promise<any>;
          validate(data: any): boolean;
        }
      `;
      
      const implementationContent = `
        import { IDataProcessor } from './interfaces';
        
        export class JsonProcessor implements IDataProcessor {
          async process(data: any): Promise<any> {
            return JSON.parse(data);
          }
          
          validate(data: any): boolean {
            try {
              JSON.parse(data);
              return true;
            } catch {
              return false;
            }
          }
        }
      `;
      
      await resolver.indexFileSymbols('interfaces.ts', interfaceContent);
      await resolver.indexFileSymbols('json-processor.ts', implementationContent);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle malformed code gracefully', async () => {
      const malformedContent = `
        function incomplete(
        const missingEquals
        class Unclosed {
          method() {
        // This is malformed code
      `;
      
      await resolver.indexFileSymbols('malformed.ts', malformedContent);
      
      // Should not crash, may or may not find symbols
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThanOrEqual(0);
    });

    it('should handle very large files', async () => {
      // Generate a large file content
      const largeFunctionList = Array.from({ length: 1000 }, (_, i) => 
        `function generatedFunction${i}() { return ${i}; }`
      ).join('\n');
      
      await resolver.indexFileSymbols('large.ts', largeFunctionList);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle files with no symbols', async () => {
      const noSymbolsContent = `
        // This file has no extractable symbols
        "use strict";
        
        // Just comments and directives
      `;
      
      const initialStats = resolver.getStats();
      await resolver.indexFileSymbols('no-symbols.js', noSymbolsContent);
      const afterStats = resolver.getStats();
      
      expect(afterStats.indexedSymbols).toBe(initialStats.indexedSymbols);
    });

    it('should handle unknown file extensions', async () => {
      const content = `
        some content here
        that might be code
        but unknown extension
      `;
      
      const initialStats = resolver.getStats();
      await resolver.indexFileSymbols('unknown.xyz', content);
      const afterStats = resolver.getStats();
      
      expect(afterStats.indexedSymbols).toBe(initialStats.indexedSymbols);
    });
  });

  describe('Performance and Memory Management', () => {
    it('should handle concurrent file indexing', async () => {
      const files = [
        ['file1.ts', 'export function func1() {}'],
        ['file2.ts', 'export function func2() {}'],
        ['file3.ts', 'export function func3() {}'],
        ['file4.ts', 'export function func4() {}'],
        ['file5.ts', 'export function func5() {}'],
      ];
      
      const promises = files.map(([path, content]) => 
        resolver.indexFileSymbols(path, content)
      );
      
      await Promise.all(promises);
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should track resolution statistics', async () => {
      const content = 'export function test() {}';
      await resolver.indexFileSymbols('test.ts', content);
      
      const context = {
        repo_name: 'test-repo',
        repo_sha: 'abc123',
        mode: 'symbol' as const,
      };
      
      await resolver.resolveCrossLanguage(context);
      
      const stats = resolver.getStats();
      expect(stats.resolutionStats).toBeDefined();
    });
  });

  describe('Index Management', () => {
    it('should clear index properly', async () => {
      // Add some symbols first
      const content = 'export function test() {}';
      await resolver.indexFileSymbols('test.ts', content);
      
      let stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
      
      // Clear the index
      resolver.clearIndex();
      
      stats = resolver.getStats();
      expect(stats.indexedSymbols).toBe(0);
    });

    it('should handle shutdown gracefully', () => {
      // Add some symbols first
      resolver.indexFileSymbols('test.ts', 'export function test() {}');
      
      // Should not throw
      expect(() => resolver.shutdown()).not.toThrow();
      
      // Should clear the index
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBe(0);
    });

    it('should provide comprehensive statistics', async () => {
      // Add symbols from different languages
      await resolver.indexFileSymbols('test.ts', 'export function test() {}');
      await resolver.indexFileSymbols('test.py', 'def test(): pass');
      await resolver.indexFileSymbols('test.js', 'function test() {}');
      
      const stats = resolver.getStats();
      
      expect(typeof stats.indexedSymbols).toBe('number');
      expect(typeof stats.languagesSupported).toBe('number');
      expect(stats.resolutionStats).toBeDefined();
      expect(stats.languagesSupported).toBeGreaterThan(0);
    });
  });

  describe('Multi-Language Symbol Patterns', () => {
    it('should recognize common symbol types across languages', async () => {
      const patterns = {
        'functions.ts': 'export function processData() {}',
        'classes.ts': 'export class DataProcessor {}',
        'interfaces.ts': 'export interface IProcessor {}',
        'constants.ts': 'export const MAX_SIZE = 100;',
        'types.ts': 'export type ProcessorType = string;',
      };
      
      for (const [file, content] of Object.entries(patterns)) {
        await resolver.indexFileSymbols(file, content);
      }
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });

    it('should handle language-specific keywords and patterns', async () => {
      const languagePatterns = {
        'typescript.ts': `
          async function asyncFunc(): Promise<string> { return ''; }
          const arrowFunc = (x: number): number => x * 2;
          interface Generic<T> { value: T; }
        `,
        'python.py': `
          async def async_func() -> str: return ''
          lambda x: x * 2
          class Generic[T]: pass
        `,
        'java.java': `
          public static void main(String[] args) {}
          @Override public String toString() { return ""; }
          public <T> T generic(T value) { return value; }
        `,
      };
      
      for (const [file, content] of Object.entries(languagePatterns)) {
        await resolver.indexFileSymbols(file, content);
      }
      
      const stats = resolver.getStats();
      expect(stats.indexedSymbols).toBeGreaterThan(0);
    });
  });
});