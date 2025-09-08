/**
 * Unit Tests for Core Domain Types
 * Tests all interfaces, types, and type compatibility for the Lens search system core types
 */

import { describe, it, expect } from 'vitest';
import type {
  // Re-export
  SearchHit,
  
  // Shard and segment types
  Shard,
  Segment,
  SegmentType,
  ShardStatus,
  
  // Index structures for Layer 1 (Lexical+Fuzzy)
  TrigramIndex,
  FST,
  FSTState,
  FSTTransition,
  DocumentPosition,
  
  // Symbol and AST types for Layer 2
  SymbolIndex,
  SymbolDefinition,
  SymbolReference,
  ASTNode,
  SymbolKind,
  
  // Semantic types for Layer 3 (Rerank)
  SemanticIndex,
  HNSWIndex,
  HNSWLayer,
  HNSWNode,
  
  // Search processing types
  SearchContext,
  StageResult,
  Candidate,
  SearchMode,
  MatchReason,
  
  // Work units for NATS/JetStream
  WorkUnit,
  WorkType,
  
  // File-based segment interface
  MMapSegment,
  
  // Telemetry and monitoring
  SearchMetrics,
  SystemHealth,
  HealthStatus,
  
  // LSP-assist system types
  LSPHint,
  LSPSidecarConfig,
  LSPCapabilities,
  WorkspaceConfig,
  LSPFeatures,
  QueryIntent,
  IntentClassification,
  LSPBenchmarkResult,
  LossTaxonomy,
  
  // Search result interface
  SearchResult,
  
  // Additional interfaces
  TestFailure,
  ChangeEvent,
  CodeOwner,
  SupportedLanguage,
} from '../core.js';

describe('Core Types - Basic Type Definitions', () => {
  describe('SegmentType', () => {
    it('should accept valid segment types', () => {
      const lexical: SegmentType = 'lexical';
      const symbols: SegmentType = 'symbols';
      const ast: SegmentType = 'ast';
      const semantic: SegmentType = 'semantic';
      
      expect(lexical).toBe('lexical');
      expect(symbols).toBe('symbols');
      expect(ast).toBe('ast');
      expect(semantic).toBe('semantic');
    });
  });

  describe('ShardStatus', () => {
    it('should accept valid shard statuses', () => {
      const active: ShardStatus = 'active';
      const compacting: ShardStatus = 'compacting';
      const inactive: ShardStatus = 'inactive';
      const error: ShardStatus = 'error';
      
      expect(active).toBe('active');
      expect(compacting).toBe('compacting');
      expect(inactive).toBe('inactive');
      expect(error).toBe('error');
    });
  });

  describe('SymbolKind', () => {
    it('should accept all valid symbol kinds', () => {
      const validKinds: SymbolKind[] = [
        'function',
        'class',
        'variable',
        'type',
        'interface',
        'constant',
        'enum',
        'method',
        'property'
      ];
      
      validKinds.forEach(kind => {
        const symbolKind: SymbolKind = kind;
        expect(symbolKind).toBe(kind);
      });
    });
  });

  describe('SearchMode', () => {
    it('should accept valid search modes', () => {
      const lex: SearchMode = 'lex';
      const lexical: SearchMode = 'lexical';
      const struct: SearchMode = 'struct';
      const hybrid: SearchMode = 'hybrid';
      
      expect(lex).toBe('lex');
      expect(lexical).toBe('lexical');
      expect(struct).toBe('struct');
      expect(hybrid).toBe('hybrid');
    });
  });

  describe('MatchReason', () => {
    it('should accept all valid match reasons', () => {
      const validReasons: MatchReason[] = [
        'exact',
        'fuzzy',
        'symbol',
        'struct',
        'semantic',
        'lsp_hint',
        'unicode_normalized',
        'raptor_diversity',
        'structural',
        'exact_name',
        'semantic_type',
        'subtoken'
      ];
      
      validReasons.forEach(reason => {
        const matchReason: MatchReason = reason;
        expect(matchReason).toBe(reason);
      });
    });
  });

  describe('WorkType', () => {
    it('should accept all valid work types', () => {
      const validWorkTypes: WorkType[] = [
        'index_shard',
        'compact_shard',
        'build_symbols',
        'build_ast',
        'build_semantic',
        'health_check',
        'lsp_harvest',
        'lsp_sync'
      ];
      
      validWorkTypes.forEach(workType => {
        const type: WorkType = workType;
        expect(type).toBe(workType);
      });
    });
  });

  describe('HealthStatus', () => {
    it('should accept valid health statuses', () => {
      const ok: HealthStatus = 'ok';
      const degraded: HealthStatus = 'degraded';
      const down: HealthStatus = 'down';
      
      expect(ok).toBe('ok');
      expect(degraded).toBe('degraded');
      expect(down).toBe('down');
    });
  });

  describe('QueryIntent', () => {
    it('should accept all valid query intents', () => {
      const validIntents: QueryIntent[] = [
        'def',
        'refs',
        'symbol',
        'struct',
        'lexical',
        'NL'
      ];
      
      validIntents.forEach(intent => {
        const queryIntent: QueryIntent = intent;
        expect(queryIntent).toBe(intent);
      });
    });
  });

  describe('SupportedLanguage', () => {
    it('should accept all supported programming languages', () => {
      const validLanguages: SupportedLanguage[] = [
        'typescript',
        'javascript',
        'python',
        'rust',
        'go',
        'java',
        'bash',
        'cpp',
        'c',
        'csharp',
        'php',
        'ruby',
        'scala',
        'kotlin',
        'swift',
        'dart',
        'lua',
        'r',
        'shell',
        'yaml',
        'json',
        'markdown',
        'html',
        'css',
        'sql'
      ];
      
      validLanguages.forEach(lang => {
        const language: SupportedLanguage = lang;
        expect(language).toBe(lang);
      });
    });
  });
});

describe('Core Interfaces - Shard and Segment Types', () => {
  describe('Segment', () => {
    it('should create a valid segment object', () => {
      const segment: Segment = {
        id: 'segment-123',
        type: 'lexical',
        file_path: '/path/to/segment.dat',
        size_bytes: 1024 * 1024,
        memory_mapped: true,
        last_accessed: new Date('2024-01-01T00:00:00Z'),
      };
      
      expect(segment.id).toBe('segment-123');
      expect(segment.type).toBe('lexical');
      expect(segment.file_path).toBe('/path/to/segment.dat');
      expect(segment.size_bytes).toBe(1024 * 1024);
      expect(segment.memory_mapped).toBe(true);
      expect(segment.last_accessed).toBeInstanceOf(Date);
    });
  });

  describe('Shard', () => {
    it('should create a valid shard object', () => {
      const shard: Shard = {
        id: 'shard-456',
        path_hash: 'abc123def',
        segments: [],
        status: 'active',
        last_compacted: new Date('2024-01-01T00:00:00Z'),
        size_mb: 128,
      };
      
      expect(shard.id).toBe('shard-456');
      expect(shard.path_hash).toBe('abc123def');
      expect(Array.isArray(shard.segments)).toBe(true);
      expect(shard.status).toBe('active');
      expect(shard.last_compacted).toBeInstanceOf(Date);
      expect(shard.size_mb).toBe(128);
    });

    it('should support shards with segments', () => {
      const segment: Segment = {
        id: 'seg-1',
        type: 'semantic',
        file_path: '/segment1.dat',
        size_bytes: 2048,
        memory_mapped: false,
        last_accessed: new Date(),
      };
      
      const shard: Shard = {
        id: 'shard-with-segments',
        path_hash: 'hash789',
        segments: [segment],
        status: 'compacting',
        last_compacted: new Date(),
        size_mb: 256,
      };
      
      expect(shard.segments).toHaveLength(1);
      expect(shard.segments[0].type).toBe('semantic');
    });
  });
});

describe('Core Interfaces - Layer 1 (Lexical+Fuzzy)', () => {
  describe('DocumentPosition', () => {
    it('should create a valid document position', () => {
      const position: DocumentPosition = {
        doc_id: 'doc-123',
        file_path: '/src/example.ts',
        line: 42,
        col: 15,
        length: 10,
      };
      
      expect(position.doc_id).toBe('doc-123');
      expect(position.file_path).toBe('/src/example.ts');
      expect(position.line).toBe(42);
      expect(position.col).toBe(15);
      expect(position.length).toBe(10);
    });
  });

  describe('FSTState', () => {
    it('should create a valid FST state', () => {
      const state: FSTState = {
        id: 1,
        is_final: true,
        edit_distance: 2,
      };
      
      expect(state.id).toBe(1);
      expect(state.is_final).toBe(true);
      expect(state.edit_distance).toBe(2);
    });
  });

  describe('FSTTransition', () => {
    it('should create a valid FST transition', () => {
      const transition: FSTTransition = {
        from_state: 0,
        to_state: 1,
        input_char: 'a',
        output_char: 'a',
        cost: 0.5,
      };
      
      expect(transition.from_state).toBe(0);
      expect(transition.to_state).toBe(1);
      expect(transition.input_char).toBe('a');
      expect(transition.output_char).toBe('a');
      expect(transition.cost).toBe(0.5);
    });

    it('should support transitions without input/output chars', () => {
      const epsilonTransition: FSTTransition = {
        from_state: 0,
        to_state: 1,
        cost: 1.0,
      };
      
      expect(epsilonTransition.input_char).toBeUndefined();
      expect(epsilonTransition.output_char).toBeUndefined();
      expect(epsilonTransition.cost).toBe(1.0);
    });
  });

  describe('FST and TrigramIndex', () => {
    it('should create a valid FST structure', () => {
      const states: FSTState[] = [
        { id: 0, is_final: false, edit_distance: 0 },
        { id: 1, is_final: true, edit_distance: 1 },
      ];
      
      const transitions = new Map<string, FSTTransition[]>();
      transitions.set('a', [
        { from_state: 0, to_state: 1, input_char: 'a', cost: 1.0 }
      ]);
      
      const fst: FST = {
        states,
        transitions,
      };
      
      expect(fst.states).toHaveLength(2);
      expect(fst.transitions.size).toBe(1);
      expect(fst.transitions.get('a')).toHaveLength(1);
    });

    it('should create a valid trigram index', () => {
      const position: DocumentPosition = {
        doc_id: 'doc1',
        file_path: '/test.ts',
        line: 1,
        col: 0,
        length: 3,
      };
      
      const trigrams = new Map<string, Set<DocumentPosition>>();
      trigrams.set('fun', new Set([position]));
      
      const fst: FST = {
        states: [],
        transitions: new Map(),
      };
      
      const trigramIndex: TrigramIndex = {
        trigrams,
        fst,
      };
      
      expect(trigramIndex.trigrams.size).toBe(1);
      expect(trigramIndex.trigrams.get('fun')?.size).toBe(1);
      expect(trigramIndex.fst).toBeDefined();
    });
  });
});

describe('Core Interfaces - Layer 2 (Symbol/AST)', () => {
  describe('SymbolDefinition', () => {
    it('should create a valid symbol definition', () => {
      const symbol: SymbolDefinition = {
        name: 'myFunction',
        kind: 'function',
        file_path: '/src/utils.ts',
        line: 10,
        col: 0,
        scope: 'global',
        signature: 'function myFunction(arg: string): number',
      };
      
      expect(symbol.name).toBe('myFunction');
      expect(symbol.kind).toBe('function');
      expect(symbol.file_path).toBe('/src/utils.ts');
      expect(symbol.line).toBe(10);
      expect(symbol.col).toBe(0);
      expect(symbol.scope).toBe('global');
      expect(symbol.signature).toBeDefined();
    });

    it('should support alternative field names', () => {
      const symbol: SymbolDefinition = {
        name: 'MyClass',
        kind: 'class',
        file_path: '/src/class.ts',
        file: '/src/class.ts', // Alternative field name
        line: 1,
        col: 0,
        scope: 'module',
      };
      
      expect(symbol.file).toBe('/src/class.ts');
      expect(symbol.file_path).toBe('/src/class.ts');
    });
  });

  describe('SymbolReference', () => {
    it('should create a valid symbol reference', () => {
      const reference: SymbolReference = {
        symbol_name: 'myFunction',
        file_path: '/src/main.ts',
        line: 20,
        col: 5,
        context: 'const result = myFunction("test");',
      };
      
      expect(reference.symbol_name).toBe('myFunction');
      expect(reference.file_path).toBe('/src/main.ts');
      expect(reference.line).toBe(20);
      expect(reference.col).toBe(5);
      expect(reference.context).toBe('const result = myFunction("test");');
    });
  });

  describe('ASTNode', () => {
    it('should create a valid AST node', () => {
      const node: ASTNode = {
        id: 'node-123',
        type: 'FunctionDeclaration',
        file_path: '/src/func.ts',
        start_line: 5,
        start_col: 0,
        end_line: 10,
        end_col: 1,
        children_ids: ['child-1', 'child-2'],
        text: 'function example() { return 42; }',
      };
      
      expect(node.id).toBe('node-123');
      expect(node.type).toBe('FunctionDeclaration');
      expect(node.file_path).toBe('/src/func.ts');
      expect(node.start_line).toBe(5);
      expect(node.start_col).toBe(0);
      expect(node.end_line).toBe(10);
      expect(node.end_col).toBe(1);
      expect(node.children_ids).toHaveLength(2);
      expect(node.text).toBeDefined();
    });

    it('should support nodes with parent relationships', () => {
      const parentNode: ASTNode = {
        id: 'parent-1',
        type: 'ClassDeclaration',
        file_path: '/src/class.ts',
        start_line: 1,
        start_col: 0,
        end_line: 20,
        end_col: 1,
        children_ids: ['child-1'],
        text: 'class MyClass { ... }',
      };
      
      const childNode: ASTNode = {
        id: 'child-1',
        type: 'MethodDefinition',
        file_path: '/src/class.ts',
        start_line: 5,
        start_col: 2,
        end_line: 10,
        end_col: 3,
        parent_id: 'parent-1',
        children_ids: [],
        text: 'method() { ... }',
      };
      
      expect(childNode.parent_id).toBe('parent-1');
      expect(parentNode.children_ids).toContain('child-1');
    });
  });

  describe('SymbolIndex', () => {
    it('should create a valid symbol index', () => {
      const definition: SymbolDefinition = {
        name: 'testFunc',
        kind: 'function',
        file_path: '/test.ts',
        line: 1,
        col: 0,
        scope: 'global',
      };
      
      const reference: SymbolReference = {
        symbol_name: 'testFunc',
        file_path: '/main.ts',
        line: 10,
        col: 5,
        context: 'testFunc()',
      };
      
      const astNode: ASTNode = {
        id: 'ast-1',
        type: 'Identifier',
        file_path: '/test.ts',
        start_line: 1,
        start_col: 9,
        end_line: 1,
        end_col: 17,
        children_ids: [],
        text: 'testFunc',
      };
      
      const symbolIndex: SymbolIndex = {
        definitions: new Map([['testFunc', [definition]]]),
        references: new Map([['testFunc', [reference]]]),
        ast_nodes: new Map([['testFunc', [astNode]]]),
      };
      
      expect(symbolIndex.definitions.size).toBe(1);
      expect(symbolIndex.references.size).toBe(1);
      expect(symbolIndex.ast_nodes.size).toBe(1);
      expect(symbolIndex.definitions.get('testFunc')).toHaveLength(1);
    });
  });
});

describe('Core Interfaces - Layer 3 (Semantic)', () => {
  describe('HNSWNode', () => {
    it('should create a valid HNSW node', () => {
      const vector = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const connections = new Set<number>([1, 2, 3]);
      
      const node: HNSWNode = {
        id: 0,
        vector,
        connections,
      };
      
      expect(node.id).toBe(0);
      expect(node.vector).toBeInstanceOf(Float32Array);
      expect(node.vector).toHaveLength(4);
      expect(node.connections.size).toBe(3);
      expect(node.connections.has(1)).toBe(true);
    });
  });

  describe('HNSWLayer', () => {
    it('should create a valid HNSW layer', () => {
      const node: HNSWNode = {
        id: 0,
        vector: new Float32Array([1, 2, 3]),
        connections: new Set([1]),
      };
      
      const nodes = new Map<number, HNSWNode>();
      nodes.set(0, node);
      
      const layer: HNSWLayer = {
        level: 0,
        nodes,
      };
      
      expect(layer.level).toBe(0);
      expect(layer.nodes.size).toBe(1);
      expect(layer.nodes.get(0)).toBe(node);
    });
  });

  describe('HNSWIndex', () => {
    it('should create a valid HNSW index', () => {
      const layer: HNSWLayer = {
        level: 0,
        nodes: new Map(),
      };
      
      const index: HNSWIndex = {
        layers: [layer],
        entry_point: 0,
        max_connections: 16,
        level_multiplier: 1.0 / Math.log(2),
      };
      
      expect(index.layers).toHaveLength(1);
      expect(index.entry_point).toBe(0);
      expect(index.max_connections).toBe(16);
      expect(index.level_multiplier).toBeGreaterThan(0);
    });
  });

  describe('SemanticIndex', () => {
    it('should create a valid semantic index', () => {
      const vectors = new Map<string, Float32Array>();
      vectors.set('doc1', new Float32Array([0.1, 0.2, 0.3]));
      
      const semanticIndex: SemanticIndex = {
        vectors,
      };
      
      expect(semanticIndex.vectors.size).toBe(1);
      expect(semanticIndex.vectors.get('doc1')).toBeInstanceOf(Float32Array);
    });

    it('should support semantic index with HNSW', () => {
      const vectors = new Map<string, Float32Array>();
      const hnswIndex: HNSWIndex = {
        layers: [],
        entry_point: 0,
        max_connections: 16,
        level_multiplier: 1.44,
      };
      
      const semanticIndex: SemanticIndex = {
        vectors,
        hnsw_index: hnswIndex,
      };
      
      expect(semanticIndex.hnsw_index).toBeDefined();
      expect(semanticIndex.hnsw_index?.max_connections).toBe(16);
    });
  });
});

describe('Core Interfaces - Search Processing', () => {
  describe('SearchContext', () => {
    it('should create a valid search context with required fields', () => {
      const context: SearchContext = {
        trace_id: 'trace-123',
        repo_sha: 'abc123',
        query: 'function search',
        mode: 'hybrid',
        k: 50,
        fuzzy_distance: 2,
        started_at: new Date(),
        stages: [],
      };
      
      expect(context.trace_id).toBe('trace-123');
      expect(context.repo_sha).toBe('abc123');
      expect(context.query).toBe('function search');
      expect(context.mode).toBe('hybrid');
      expect(context.k).toBe(50);
      expect(context.fuzzy_distance).toBe(2);
      expect(context.started_at).toBeInstanceOf(Date);
      expect(Array.isArray(context.stages)).toBe(true);
    });

    it('should support optional fields', () => {
      const context: SearchContext = {
        trace_id: 'trace-456',
        repo_sha: 'def456',
        query: 'class Component',
        mode: 'struct',
        k: 100,
        fuzzy_distance: 1,
        started_at: new Date(),
        stages: [],
        fuzzy: true,
        searchType: 'semantic',
        filters: {
          language: ['typescript'],
          symbolType: ['class'],
          repositories: ['repo1'],
          customFilter: 'value',
        },
        stageTimings: {
          stage_a: 5,
          stage_b: 7,
          stage_c: 12,
        },
        rankingStrategy: 'semantic_boost',
        userId: 'user123',
        repositories: ['repo1', 'repo2'],
        language: 'typescript',
      };
      
      expect(context.fuzzy).toBe(true);
      expect(context.searchType).toBe('semantic');
      expect(context.filters?.language).toContain('typescript');
      expect(context.stageTimings?.stage_a).toBe(5);
      expect(context.rankingStrategy).toBe('semantic_boost');
      expect(context.userId).toBe('user123');
      expect(context.repositories).toContain('repo1');
      expect(context.language).toBe('typescript');
    });
  });

  describe('StageResult', () => {
    it('should create valid stage results', () => {
      const stageA: StageResult = {
        stage: 'stage_a',
        latency_ms: 5,
        candidates_in: 1000,
        candidates_out: 100,
        method: 'trigram+fst',
      };
      
      const stageB: StageResult = {
        stage: 'stage_b',
        latency_ms: 7,
        candidates_in: 100,
        candidates_out: 50,
        method: 'symbol+ast',
        error: 'partial_failure',
      };
      
      expect(stageA.stage).toBe('stage_a');
      expect(stageA.latency_ms).toBe(5);
      expect(stageA.candidates_in).toBe(1000);
      expect(stageA.candidates_out).toBe(100);
      expect(stageA.method).toBe('trigram+fst');
      expect(stageA.error).toBeUndefined();
      
      expect(stageB.error).toBe('partial_failure');
    });
  });

  describe('Candidate', () => {
    it('should create a valid candidate with required fields', () => {
      const candidate: Candidate = {
        doc_id: 'doc-123',
        file_path: '/src/example.ts',
        line: 42,
        col: 10,
        score: 0.95,
        match_reasons: ['exact', 'symbol'],
      };
      
      expect(candidate.doc_id).toBe('doc-123');
      expect(candidate.file_path).toBe('/src/example.ts');
      expect(candidate.line).toBe(42);
      expect(candidate.col).toBe(10);
      expect(candidate.score).toBe(0.95);
      expect(candidate.match_reasons).toHaveLength(2);
    });

    it('should support all optional fields', () => {
      const candidate: Candidate = {
        doc_id: 'doc-456',
        file_path: '/src/class.ts',
        file: '/src/class.ts', // Alternative field name
        line: 15,
        col: 5,
        score: 0.87,
        match_reasons: ['fuzzy', 'semantic'],
        why: ['Fuzzy match with edit distance 1', 'High semantic similarity'],
        lang: 'typescript',
        ast_path: 'ClassDeclaration.MethodDefinition',
        symbol_kind: 'method',
        snippet: 'public method() {',
        byte_offset: 1024,
        span_len: 20,
        context_before: 'class MyClass {\n  ',
        context_after: '\n    return true;\n  }',
        context: 'Full context around the match',
      };
      
      expect(candidate.file).toBe('/src/class.ts');
      expect(candidate.why).toHaveLength(2);
      expect(candidate.lang).toBe('typescript');
      expect(candidate.ast_path).toBe('ClassDeclaration.MethodDefinition');
      expect(candidate.symbol_kind).toBe('method');
      expect(candidate.snippet).toBe('public method() {');
      expect(candidate.byte_offset).toBe(1024);
      expect(candidate.span_len).toBe(20);
      expect(candidate.context_before).toBeDefined();
      expect(candidate.context_after).toBeDefined();
      expect(candidate.context).toBeDefined();
    });
  });
});

describe('Core Interfaces - Work Units and System Management', () => {
  describe('WorkUnit', () => {
    it('should create a valid work unit', () => {
      const workUnit: WorkUnit = {
        id: 'work-123',
        type: 'index_shard',
        repo_sha: 'abc123def',
        shard_id: 'shard-456',
        payload: { files: ['file1.ts', 'file2.ts'] },
        priority: 5,
        created_at: new Date(),
        assigned_to: 'worker-1',
      };
      
      expect(workUnit.id).toBe('work-123');
      expect(workUnit.type).toBe('index_shard');
      expect(workUnit.repo_sha).toBe('abc123def');
      expect(workUnit.shard_id).toBe('shard-456');
      expect(workUnit.payload).toBeDefined();
      expect(workUnit.priority).toBe(5);
      expect(workUnit.created_at).toBeInstanceOf(Date);
      expect(workUnit.assigned_to).toBe('worker-1');
    });

    it('should support work units without assignment', () => {
      const workUnit: WorkUnit = {
        id: 'work-456',
        type: 'health_check',
        repo_sha: 'def456ghi',
        shard_id: 'shard-789',
        payload: null,
        priority: 1,
        created_at: new Date(),
      };
      
      expect(workUnit.assigned_to).toBeUndefined();
      expect(workUnit.payload).toBeNull();
    });
  });

  describe('MMapSegment', () => {
    it('should create a valid memory-mapped segment', () => {
      const buffer = Buffer.alloc(1024);
      const segment: MMapSegment = {
        file_path: '/data/segment.dat',
        fd: 3,
        size: 1024,
        buffer,
        readonly: true,
      };
      
      expect(segment.file_path).toBe('/data/segment.dat');
      expect(segment.fd).toBe(3);
      expect(segment.size).toBe(1024);
      expect(segment.buffer).toBeInstanceOf(Buffer);
      expect(segment.readonly).toBe(true);
    });
  });
});

describe('Core Interfaces - Telemetry and Monitoring', () => {
  describe('SearchMetrics', () => {
    it('should create valid search metrics', () => {
      const queriesByMode = new Map<SearchMode, number>();
      queriesByMode.set('hybrid', 100);
      queriesByMode.set('lexical', 50);
      
      const errorRates = new Map<string, number>();
      errorRates.set('timeout', 0.02);
      errorRates.set('index_error', 0.01);
      
      const metrics: SearchMetrics = {
        total_queries: 150,
        queries_by_mode: queriesByMode,
        stage_latencies: {
          stage_a_p50: 3,
          stage_a_p95: 8,
          stage_b_p50: 5,
          stage_b_p95: 12,
          stage_c_p50: 8,
          stage_c_p95: 20,
          total_p50: 16,
          total_p95: 40,
        },
        cache_hit_rates: {
          trigram_cache: 0.85,
          symbol_cache: 0.72,
          semantic_cache: 0.65,
        },
        error_rates: errorRates,
      };
      
      expect(metrics.total_queries).toBe(150);
      expect(metrics.queries_by_mode.get('hybrid')).toBe(100);
      expect(metrics.stage_latencies.stage_a_p95).toBe(8);
      expect(metrics.cache_hit_rates.trigram_cache).toBe(0.85);
      expect(metrics.error_rates.get('timeout')).toBe(0.02);
    });
  });

  describe('SystemHealth', () => {
    it('should create valid system health status', () => {
      const health: SystemHealth = {
        status: 'ok',
        shards_healthy: 8,
        shards_total: 10,
        memory_usage_gb: 12.5,
        active_queries: 25,
        worker_pool_status: {
          ingest_active: 3,
          query_active: 8,
          maintenance_active: 1,
        },
        last_compaction: new Date('2024-01-01T12:00:00Z'),
      };
      
      expect(health.status).toBe('ok');
      expect(health.shards_healthy).toBe(8);
      expect(health.shards_total).toBe(10);
      expect(health.memory_usage_gb).toBe(12.5);
      expect(health.active_queries).toBe(25);
      expect(health.worker_pool_status.query_active).toBe(8);
      expect(health.last_compaction).toBeInstanceOf(Date);
    });
  });
});

describe('Core Interfaces - LSP System Types', () => {
  describe('LSPHint', () => {
    it('should create a valid LSP hint', () => {
      const hint: LSPHint = {
        symbol_id: 'sym-123',
        name: 'myFunction',
        kind: 'function',
        file_path: '/src/utils.ts',
        line: 10,
        col: 0,
        definition_uri: 'file:///src/utils.ts#10:0',
        signature: 'function myFunction(param: string): number',
        type_info: '(param: string) => number',
        aliases: ['myFunc', 'utilFunction'],
        resolved_imports: ['./types', '../constants'],
        references_count: 15,
      };
      
      expect(hint.symbol_id).toBe('sym-123');
      expect(hint.name).toBe('myFunction');
      expect(hint.kind).toBe('function');
      expect(hint.aliases).toHaveLength(2);
      expect(hint.resolved_imports).toHaveLength(2);
      expect(hint.references_count).toBe(15);
    });
  });

  describe('LSPCapabilities', () => {
    it('should create valid LSP capabilities', () => {
      const capabilities: LSPCapabilities = {
        definition: true,
        references: true,
        hover: true,
        completion: false,
        rename: false,
        workspace_symbols: true,
      };
      
      expect(capabilities.definition).toBe(true);
      expect(capabilities.references).toBe(true);
      expect(capabilities.hover).toBe(true);
      expect(capabilities.completion).toBe(false);
      expect(capabilities.rename).toBe(false);
      expect(capabilities.workspace_symbols).toBe(true);
    });
  });

  describe('WorkspaceConfig', () => {
    it('should create valid workspace config', () => {
      const pathMappings = new Map<string, string>();
      pathMappings.set('@/*', './src/*');
      
      const config: WorkspaceConfig = {
        root_path: '/project/root',
        include_patterns: ['**/*.ts', '**/*.js'],
        exclude_patterns: ['**/node_modules/**', '**/*.test.ts'],
        path_mappings: pathMappings,
        config_files: ['tsconfig.json', 'package.json'],
      };
      
      expect(config.root_path).toBe('/project/root');
      expect(config.include_patterns).toHaveLength(2);
      expect(config.exclude_patterns).toHaveLength(2);
      expect(config.path_mappings?.get('@/*')).toBe('./src/*');
      expect(config.config_files).toContain('tsconfig.json');
    });

    it('should support minimal workspace config', () => {
      const config: WorkspaceConfig = {
        include_patterns: ['**/*.ts'],
        exclude_patterns: ['**/dist/**'],
      };
      
      expect(config.root_path).toBeUndefined();
      expect(config.path_mappings).toBeUndefined();
      expect(config.config_files).toBeUndefined();
      expect(config.include_patterns).toHaveLength(1);
    });
  });

  describe('LSPSidecarConfig', () => {
    it('should create valid LSP sidecar config', () => {
      const config: LSPSidecarConfig = {
        language: 'typescript',
        lsp_server: 'typescript-language-server',
        capabilities: {
          definition: true,
          references: true,
          hover: true,
          completion: true,
          rename: false,
          workspace_symbols: true,
        },
        workspace_config: {
          include_patterns: ['**/*.ts'],
          exclude_patterns: ['**/node_modules/**'],
        },
        harvest_ttl_hours: 24,
        pressure_threshold: 0.8,
      };
      
      expect(config.language).toBe('typescript');
      expect(config.lsp_server).toBe('typescript-language-server');
      expect(config.capabilities.definition).toBe(true);
      expect(config.harvest_ttl_hours).toBe(24);
      expect(config.pressure_threshold).toBe(0.8);
    });
  });

  describe('IntentClassification', () => {
    it('should create valid intent classification', () => {
      const classification: IntentClassification = {
        intent: 'def',
        confidence: 0.95,
        features: {
          has_definition_pattern: true,
          has_reference_pattern: false,
          has_symbol_prefix: true,
          has_structural_chars: false,
          is_natural_language: false,
        },
      };
      
      expect(classification.intent).toBe('def');
      expect(classification.confidence).toBe(0.95);
      expect(classification.features.has_definition_pattern).toBe(true);
      expect(classification.features.is_natural_language).toBe(false);
    });
  });

  describe('LSPBenchmarkResult', () => {
    it('should create valid LSP benchmark result', () => {
      const lossTaxonomy: LossTaxonomy = {
        NO_SYM_COVERAGE: 0.1,
        WRONG_ALIAS: 0.05,
        PATH_MAP: 0.02,
        USABILITY_INTENT: 0.03,
        RANKING_ONLY: 0.8,
      };
      
      const result: LSPBenchmarkResult = {
        mode: 'lsp_assist',
        task_type: 'def',
        success_at_1: 0.85,
        success_at_5: 0.95,
        ndcg_at_10: 0.92,
        recall_at_50: 0.98,
        zero_result_rate: 0.02,
        timeout_rate: 0.01,
        p95_latency_ms: 45,
        loss_taxonomy: lossTaxonomy,
      };
      
      expect(result.mode).toBe('lsp_assist');
      expect(result.task_type).toBe('def');
      expect(result.success_at_1).toBe(0.85);
      expect(result.loss_taxonomy.RANKING_ONLY).toBe(0.8);
    });
  });
});

describe('Core Interfaces - Additional Types', () => {
  describe('SearchResult', () => {
    it('should create a valid search result', () => {
      const hits: SearchHit[] = []; // Empty array for testing
      
      const result: SearchResult = {
        hits,
        stage_a_latency: 5,
        stage_b_latency: 7,
        stage_c_latency: 12,
        stage_a_skipped: false,
        stage_b_skipped: false,
        stage_c_skipped: true,
      };
      
      expect(Array.isArray(result.hits)).toBe(true);
      expect(result.stage_a_latency).toBe(5);
      expect(result.stage_b_latency).toBe(7);
      expect(result.stage_c_latency).toBe(12);
      expect(result.stage_c_skipped).toBe(true);
    });
  });

  describe('TestFailure', () => {
    it('should create a valid test failure', () => {
      const failure: TestFailure = {
        test_name: 'should validate input',
        error_message: 'Expected true but received false',
        file_path: '/test/unit.test.ts',
        line_number: 42,
        timestamp: new Date(),
        stack_trace: 'Error: Expected true...\n  at test (/test/unit.test.ts:42:5)',
      };
      
      expect(failure.test_name).toBe('should validate input');
      expect(failure.error_message).toBe('Expected true but received false');
      expect(failure.file_path).toBe('/test/unit.test.ts');
      expect(failure.line_number).toBe(42);
      expect(failure.timestamp).toBeInstanceOf(Date);
      expect(failure.stack_trace).toBeDefined();
    });
  });

  describe('ChangeEvent', () => {
    it('should create valid change events', () => {
      const addEvent: ChangeEvent = {
        event_type: 'file_added',
        file_path: '/src/new-file.ts',
        timestamp: new Date(),
        change_id: 'change-123',
        metadata: { size_bytes: 1024 },
      };
      
      const renameEvent: ChangeEvent = {
        event_type: 'file_renamed',
        file_path: '/src/renamed-file.ts',
        old_file_path: '/src/old-file.ts',
        timestamp: new Date(),
        change_id: 'change-456',
      };
      
      expect(addEvent.event_type).toBe('file_added');
      expect(addEvent.file_path).toBe('/src/new-file.ts');
      expect(addEvent.old_file_path).toBeUndefined();
      expect(addEvent.metadata).toBeDefined();
      
      expect(renameEvent.event_type).toBe('file_renamed');
      expect(renameEvent.old_file_path).toBe('/src/old-file.ts');
    });
  });

  describe('CodeOwner', () => {
    it('should create a valid code owner', () => {
      const owner: CodeOwner = {
        email: 'developer@example.com',
        username: 'devuser',
        file_patterns: ['src/**/*.ts', 'tests/**/*.test.ts'],
        team: 'backend',
        role: 'senior-engineer',
        last_updated: new Date(),
      };
      
      expect(owner.email).toBe('developer@example.com');
      expect(owner.username).toBe('devuser');
      expect(owner.file_patterns).toHaveLength(2);
      expect(owner.team).toBe('backend');
      expect(owner.role).toBe('senior-engineer');
      expect(owner.last_updated).toBeInstanceOf(Date);
    });
  });
});

describe('Type Compatibility and Edge Cases', () => {
  describe('Optional Fields and Undefined Values', () => {
    it('should handle optional fields correctly', () => {
      const minimalContext: SearchContext = {
        trace_id: 'trace',
        repo_sha: 'sha',
        query: 'test',
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 1,
        started_at: new Date(),
        stages: [],
      };
      
      expect(minimalContext.fuzzy).toBeUndefined();
      expect(minimalContext.filters).toBeUndefined();
      expect(minimalContext.userId).toBeUndefined();
    });

    it('should handle union types correctly', () => {
      const candidate: Candidate = {
        doc_id: 'doc',
        file_path: '/test.ts',
        line: 1,
        col: 0,
        score: 1.0,
        match_reasons: ['exact'],
        lang: undefined, // Explicitly undefined is allowed
      };
      
      expect(candidate.lang).toBeUndefined();
    });
  });

  describe('Type Constraints', () => {
    it('should enforce numeric constraints', () => {
      const lspFeatures: LSPFeatures = {
        lsp_def_hit: 1, // Must be 0 or 1
        lsp_ref_count: 5,
        type_match: 0.8,
        alias_resolved: 0, // Must be 0 or 1
      };
      
      expect(lspFeatures.lsp_def_hit).toBe(1);
      expect(lspFeatures.alias_resolved).toBe(0);
      expect(typeof lspFeatures.lsp_ref_count).toBe('number');
    });
  });

  describe('Map and Set Types', () => {
    it('should work with Map types', () => {
      const vectors = new Map<string, Float32Array>();
      vectors.set('doc1', new Float32Array([1, 2, 3]));
      vectors.set('doc2', new Float32Array([4, 5, 6]));
      
      const semanticIndex: SemanticIndex = { vectors };
      
      expect(semanticIndex.vectors.size).toBe(2);
      expect(semanticIndex.vectors.get('doc1')?.[0]).toBe(1);
    });

    it('should work with Set types', () => {
      const connections = new Set<number>([1, 2, 3, 4]);
      
      const node: HNSWNode = {
        id: 0,
        vector: new Float32Array([1, 2]),
        connections,
      };
      
      expect(node.connections.has(2)).toBe(true);
      expect(node.connections.has(5)).toBe(false);
    });
  });

  describe('Date Type Handling', () => {
    it('should handle Date objects correctly', () => {
      const now = new Date();
      
      const shard: Shard = {
        id: 'shard',
        path_hash: 'hash',
        segments: [],
        status: 'active',
        last_compacted: now,
        size_mb: 100,
      };
      
      expect(shard.last_compacted).toBeInstanceOf(Date);
      expect(shard.last_compacted.getTime()).toBe(now.getTime());
    });
  });
});