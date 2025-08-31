/**
 * Unit tests for Symbol Search Engine
 * Tests symbol extraction, indexing, and AST-based search
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SymbolSearchEngine } from '../../src/indexer/symbols.js';
import { SegmentStorage } from '../../src/storage/segments.js';
import type { SearchContext } from '../../src/types/core.js';

describe('SymbolSearchEngine', () => {
  let engine: SymbolSearchEngine;
  let segmentStorage: SegmentStorage;

  beforeEach(async () => {
    segmentStorage = new SegmentStorage('./test-segments');
    engine = new SymbolSearchEngine(segmentStorage);
    await engine.initialize();
  });

  afterEach(async () => {
    await engine.shutdown();
    await segmentStorage.shutdown();
  });

  describe('Symbol Indexing', () => {
    it('should index TypeScript functions', async () => {
      const content = `
function calculateSum(a: number, b: number): number {
  return a + b;
}

const arrowFunc = (x: number) => x * 2;

export function publicFunction() {
  return 'public';
}
      `;

      await engine.indexFile('/test/functions.ts', content, 'typescript');
      
      const stats = engine.getStats();
      expect(stats.symbols).toBeGreaterThan(0);
    });

    it('should index TypeScript classes and interfaces', async () => {
      const content = `
interface UserInterface {
  id: number;
  name: string;
}

class UserService implements UserInterface {
  id: number;
  name: string;
  
  constructor(id: number, name: string) {
    this.id = id;
    this.name = name;
  }
  
  getName(): string {
    return this.name;
  }
}

type UserType = {
  id: number;
  email: string;
};
      `;

      await engine.indexFile('/test/classes.ts', content, 'typescript');
      
      const stats = engine.getStats();
      expect(stats.symbols).toBeGreaterThan(0);
      expect(stats.ast_nodes).toBeGreaterThan(0);
    });

    it('should index Python code', async () => {
      const content = `
def calculate_sum(a, b):
    return a + b

class UserService:
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name

my_variable = 42
CONSTANT_VALUE = "test"
      `;

      await engine.indexFile('/test/python.py', content, 'python');
      
      const stats = engine.getStats();
      expect(stats.symbols).toBeGreaterThan(0);
    });

    it('should index Rust code', async () => {
      const content = `
fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

struct User {
    id: u32,
    name: String,
}

impl User {
    fn new(id: u32, name: String) -> Self {
        User { id, name }
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}

let my_var = 42;
const CONSTANT: i32 = 100;
      `;

      await engine.indexFile('/test/rust.rs', content, 'rust');
      
      const stats = engine.getStats();
      expect(stats.symbols).toBeGreaterThan(0);
    });
  });

  describe('Symbol Search', () => {
    beforeEach(async () => {
      const content = `
function calculateSum(a: number, b: number): number {
  return a + b;
}

function calculateProduct(x: number, y: number): number {
  return x * y;
}

class MathUtils {
  static add(a: number, b: number): number {
    return calculateSum(a, b);
  }
  
  multiply(x: number, y: number): number {
    return calculateProduct(x, y);
  }
}

const myVariable = 42;
const helper = new MathUtils();
      `;

      await engine.indexFile('/test/math.ts', content, 'typescript');
    });

    it('should find function symbols', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-1',
        query: 'calculate',
        mode: 'struct',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.searchSymbols('calculate', ctx, 10);
      
      expect(results.length).toBeGreaterThan(0);
      results.forEach(result => {
        expect(result.match_reasons).toContain('symbol');
        expect(result.symbol_kind).toBe('function');
      });
    });

    it('should find class symbols', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-2',
        query: 'MathUtils',
        mode: 'struct',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.searchSymbols('MathUtils', ctx, 10);
      
      expect(results.length).toBeGreaterThan(0);
      results.forEach(result => {
        expect(result.match_reasons).toContain('symbol');
        expect(result.symbol_kind).toBe('class');
      });
    });

    it('should find variable symbols', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-3',
        query: 'myVariable',
        mode: 'struct',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.searchSymbols('myVariable', ctx, 10);
      
      expect(results.length).toBeGreaterThan(0);
      results.forEach(result => {
        expect(result.match_reasons).toContain('symbol');
        expect(result.symbol_kind).toBe('variable');
      });
    });

    it('should score exact matches higher', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-4',
        query: 'calculateSum',
        mode: 'struct',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.searchSymbols('calculateSum', ctx, 10);
      
      expect(results.length).toBeGreaterThan(0);
      
      // Find exact match
      const exactMatch = results.find(r => r.context?.includes('calculateSum'));
      expect(exactMatch).toBeDefined();
      expect(exactMatch!.score).toBe(1.0);
    });

    it('should handle partial matches', async () => {
      const ctx: SearchContext = {
        trace_id: 'test-trace-5',
        query: 'calc',
        mode: 'struct',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const results = await engine.searchSymbols('calc', ctx, 10);
      
      expect(results.length).toBeGreaterThan(0);
      results.forEach(result => {
        expect(result.score).toBeGreaterThan(0);
        expect(result.score).toBeLessThan(1.0);
      });
    });
  });

  describe('Symbols Near Location', () => {
    beforeEach(async () => {
      const content = `
function topLevelFunction() {
  const localVar = 42;
  
  function nestedFunction() {
    return localVar;
  }
  
  return nestedFunction();
}

class TestClass {
  private value: number;
  
  constructor(value: number) {
    this.value = value;
  }
  
  getValue(): number {
    return this.value;
  }
  
  setValue(newValue: number): void {
    this.value = newValue;
  }
}

const globalVar = new TestClass(10);
      `;

      await engine.indexFile('/test/location.ts', content, 'typescript');
    });

    it('should find symbols near a location', async () => {
      const results = await engine.findSymbolsNear('/test/location.ts', 10, 5);
      
      expect(results.length).toBeGreaterThan(0);
      results.forEach(result => {
        expect(result.match_reasons).toContain('struct');
        expect(result.file_path).toBe('/test/location.ts');
      });
    });

    it('should respect radius parameter', async () => {
      const smallRadius = await engine.findSymbolsNear('/test/location.ts', 10, 2);
      const largeRadius = await engine.findSymbolsNear('/test/location.ts', 10, 10);
      
      expect(largeRadius.length).toBeGreaterThanOrEqual(smallRadius.length);
    });

    it('should score by proximity to target line', async () => {
      const results = await engine.findSymbolsNear('/test/location.ts', 15, 10);
      
      if (results.length > 1) {
        // Results should be sorted by score (proximity)
        for (let i = 1; i < results.length; i++) {
          expect(results[i].score).toBeLessThanOrEqual(results[i - 1].score);
        }
      }
    });
  });

  describe('AST Parsing', () => {
    it('should parse basic AST structure', async () => {
      const content = `
{
  const obj = {
    prop: 'value',
    nested: {
      deep: true
    }
  };
  
  if (obj.nested.deep) {
    console.log('nested');
  }
}
      `;

      await engine.indexFile('/test/ast.ts', content, 'typescript');
      
      const stats = engine.getStats();
      expect(stats.ast_nodes).toBeGreaterThan(0);
    });

    it('should handle nested blocks', async () => {
      const content = `
function outer() {
  {
    const level1 = 'first';
    {
      const level2 = 'second';
      {
        const level3 = 'third';
      }
    }
  }
}
      `;

      await engine.indexFile('/test/nested.ts', content, 'typescript');
      
      const stats = engine.getStats();
      expect(stats.ast_nodes).toBeGreaterThan(3);
    });
  });

  describe('Performance', () => {
    it('should index symbols within time limits', async () => {
      // Generate a large file with many symbols
      const lines = [];
      for (let i = 0; i < 50; i++) {
        lines.push(`function func${i}(param${i}: number): number {`);
        lines.push(`  const local${i} = param${i} * 2;`);
        lines.push(`  return local${i};`);
        lines.push(`}`);
        lines.push('');
        lines.push(`class Class${i} {`);
        lines.push(`  private field${i}: string;`);
        lines.push(`  method${i}(): void {}`);
        lines.push(`}`);
        lines.push('');
      }
      const content = lines.join('\n');

      const startTime = Date.now();
      await engine.indexFile('/test/large.ts', content, 'typescript');
      const indexTime = Date.now() - startTime;

      expect(indexTime).toBeLessThan(100); // Should be fast
      
      const stats = engine.getStats();
      expect(stats.symbols).toBeGreaterThan(100);
    });

    it('should search symbols within time limits', async () => {
      const content = `
function quickSearch() { return 'quick'; }
function fastSearch() { return 'fast'; }
function speedySearch() { return 'speedy'; }
class SearchClass { searchMethod() {} }
const searchVar = 'search';
      `;

      await engine.indexFile('/test/search.ts', content, 'typescript');

      const ctx: SearchContext = {
        trace_id: 'perf-test',
        query: 'search',
        mode: 'struct',
        k: 50,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: [],
      };

      const startTime = Date.now();
      const results = await engine.searchSymbols('search', ctx, 50);
      const searchTime = Date.now() - startTime;

      expect(searchTime).toBeLessThan(20); // Should be under Stage-B target
      expect(results.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle empty files', async () => {
      await engine.indexFile('/test/empty.ts', '', 'typescript');
      
      const stats = engine.getStats();
      expect(stats.symbols).toBe(0);
    });

    it('should handle malformed code gracefully', async () => {
      const content = `
function broken(
  // Missing closing paren and brace
class Incomplete {
  method() {
    // Missing closing brace
      `;

      await engine.indexFile('/test/malformed.ts', content, 'typescript');
      
      // Should not throw error
      const stats = engine.getStats();
      expect(stats.symbols).toBeGreaterThanOrEqual(0);
    });

    it('should handle non-existent files for symbols near', async () => {
      const results = await engine.findSymbolsNear('/nonexistent/file.ts', 10, 5);
      
      expect(results).toEqual([]);
    });

    it('should handle invalid line numbers', async () => {
      await engine.indexFile('/test/simple.ts', 'const x = 1;', 'typescript');
      
      const results = await engine.findSymbolsNear('/test/simple.ts', -1, 5);
      expect(results).toEqual([]);
      
      const results2 = await engine.findSymbolsNear('/test/simple.ts', 10000, 5);
      expect(results2).toEqual([]);
    });
  });

  describe('Language-Specific Features', () => {
    it('should extract different symbol types per language', async () => {
      const languages: Array<{ lang: 'typescript' | 'python' | 'rust' | 'go' | 'java' | 'bash', content: string }> = [
        {
          lang: 'typescript',
          content: 'interface ITest { prop: string; } class Test implements ITest { prop = "test"; }'
        },
        {
          lang: 'python',
          content: 'class Test:\n    def method(self):\n        pass\n\ndef function():\n    pass'
        },
        {
          lang: 'rust',
          content: 'struct Test { field: i32 } impl Test { fn method(&self) {} } fn function() {}'
        },
        {
          lang: 'go',
          content: 'type Test struct { Field int } func (t *Test) Method() {} func Function() {}'
        },
        {
          lang: 'java',
          content: 'public class Test { private int field; public void method() {} } public static void main() {}'
        },
        {
          lang: 'bash',
          content: 'function test_func() { echo "test"; } test_var="value" source other_file.sh'
        }
      ];

      for (const { lang, content } of languages) {
        await engine.indexFile(`/test/file.${lang}`, content, lang);
      }

      const stats = engine.getStats();
      expect(stats.symbols).toBeGreaterThan(languages.length);
    });
  });
});