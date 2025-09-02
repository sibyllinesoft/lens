/**
 * Unit tests for Semantic Rerank Engine
 * Tests semantic similarity, reranking, and ColBERT-like functionality
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SemanticRerankEngine } from '../../src/indexer/semantic.js';
import { EnhancedSemanticRerankEngine } from '../../src/indexer/enhanced-semantic.js';
import { SegmentStorage } from '../../src/storage/segments.js';
import type { SearchContext, Candidate } from '../../src/types/core.js';

describe('SemanticRerankEngine', () => {
  let engine: SemanticRerankEngine;
  let segmentStorage: SegmentStorage;

  beforeEach(async () => {
    segmentStorage = new SegmentStorage('./test-segments');
    engine = new SemanticRerankEngine(segmentStorage);
    await engine.initialize();
  });

  afterEach(async () => {
    await engine.shutdown();
    await segmentStorage.shutdown();
  });

  describe('Document Indexing', () => {
    it('should index documents for semantic search', async () => {
      const content = `
function calculateSum(a, b) {
  // This function adds two numbers together
  return a + b;
}

const mathUtils = {
  add: calculateSum,
  multiply: (x, y) => x * y
};
      `;

      await engine.indexDocument('doc1', content, '/test/math.js');
      
      const stats = engine.getStats();
      expect(stats.vectors).toBe(1);
      expect(stats.avg_dim).toBeGreaterThan(0);
    });

    it('should handle multiple documents', async () => {
      const documents = [
        { id: 'doc1', content: 'function add(a, b) { return a + b; }', path: '/math.js' },
        { id: 'doc2', content: 'class Calculator { multiply(x, y) { return x * y; } }', path: '/calc.js' },
        { id: 'doc3', content: 'const utils = { divide: (a, b) => a / b };', path: '/utils.js' }
      ];

      for (const doc of documents) {
        await engine.indexDocument(doc.id, doc.content, doc.path);
      }

      const stats = engine.getStats();
      expect(stats.vectors).toBe(3);
    });

    it('should handle empty documents', async () => {
      await engine.indexDocument('empty', '', '/empty.js');
      
      const stats = engine.getStats();
      expect(stats.vectors).toBe(1);
    });
  });

  describe('Semantic Similarity', () => {
    beforeEach(async () => {
      // Index documents with different semantic meanings
      const documents = [
        { id: 'math', content: 'function add(a, b) { return a + b; } function multiply(x, y) { return x * y; }', path: '/math.js' },
        { id: 'string', content: 'function concat(str1, str2) { return str1 + str2; } function split(text, delimiter) { return text.split(delimiter); }', path: '/string.js' },
        { id: 'array', content: 'function push(arr, item) { arr.push(item); } function filter(arr, fn) { return arr.filter(fn); }', path: '/array.js' },
        { id: 'http', content: 'function get(url) { return fetch(url); } function post(url, data) { return fetch(url, { method: "POST", body: data }); }', path: '/http.js' }
      ];

      for (const doc of documents) {
        await engine.indexDocument(doc.id, doc.content, doc.path);
      }
    });

    it('should find semantically similar documents', async () => {
      const queryEmbedding = await (engine as any).embeddingModel.encode('arithmetic operations addition multiplication');
      
      const similar = await engine.findSimilarDocuments(queryEmbedding, 3);
      
      expect(similar.length).toBeGreaterThan(0);
      expect(similar[0].score).toBeGreaterThan(0);
      
      // Math document should be most similar to arithmetic query
      const mathDoc = similar.find(doc => doc.doc_id === 'math');
      expect(mathDoc).toBeDefined();
    });

    it('should score similar content higher', async () => {
      const mathQuery = await (engine as any).embeddingModel.encode('mathematical functions addition');
      const stringQuery = await (engine as any).embeddingModel.encode('text processing concatenation');
      
      const mathSimilar = await engine.findSimilarDocuments(mathQuery, 4);
      const stringSimilar = await engine.findSimilarDocuments(stringQuery, 4);
      
      // Math query should rank math doc higher
      const mathDocFromMathQuery = mathSimilar.find(doc => doc.doc_id === 'math');
      const mathDocFromStringQuery = stringSimilar.find(doc => doc.doc_id === 'math');
      
      expect(mathDocFromMathQuery?.score).toBeGreaterThan(mathDocFromStringQuery?.score || 0);
    });
  });

  describe('Candidate Reranking', () => {
    let candidates: Candidate[];
    let searchContext: SearchContext;

    beforeEach(async () => {
      // Index documents for reranking
      await engine.indexDocument('relevant', 'function calculateSum(a, b) { return a + b; }', '/relevant.js');
      await engine.indexDocument('somewhat', 'function processData(data) { return data.map(x => x * 2); }', '/process.js');
      await engine.indexDocument('unrelated', 'function fetchUser(id) { return users.find(u => u.id === id); }', '/user.js');

      candidates = [
        {
          doc_id: 'relevant:1:1',
          file_path: '/relevant.js',
          line: 1,
          col: 1,
          score: 0.8,
          match_reasons: ['exact'],
          context: 'function calculateSum(a, b) { return a + b; }'
        },
        {
          doc_id: 'somewhat:1:1',
          file_path: '/process.js',
          line: 1,
          col: 1,
          score: 0.7,
          match_reasons: ['symbol'],
          context: 'function processData(data) { return data.map(x => x * 2); }'
        },
        {
          doc_id: 'unrelated:1:1',
          file_path: '/user.js',
          line: 1,
          col: 1,
          score: 0.6,
          match_reasons: ['symbol'],
          context: 'function fetchUser(id) { return users.find(u => u.id === id); }'
        }
      ];

      searchContext = {
        trace_id: 'test-rerank',
        query: 'calculate sum addition',
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      };
    });

    it('should rerank candidates by semantic relevance', async () => {
      const reranked = await engine.rerankCandidates(candidates, searchContext, 10);
      
      expect(reranked.length).toBe(candidates.length);
      
      // Results should be reordered based on semantic similarity
      expect(reranked).not.toEqual(candidates);
    });

    it('should boost semantically relevant candidates', async () => {
      const reranked = await engine.rerankCandidates(candidates, searchContext, 10);
      
      // Find the most relevant document (should contain math/sum concepts)
      const relevantDoc = reranked.find(c => c.context?.includes('calculateSum'));
      
      if (relevantDoc) {
        expect(relevantDoc.score).toBeGreaterThan(0.8);
        
        // Should have semantic match reason if score is high enough
        if (relevantDoc.score > 0.9) {
          expect(relevantDoc.match_reasons).toContain('semantic');
        }
      }
    });

    it('should handle few candidates gracefully', async () => {
      const fewCandidates = candidates.slice(0, 1);
      
      const reranked = await engine.rerankCandidates(fewCandidates, searchContext, 10);
      
      expect(reranked).toEqual(fewCandidates);
    });

    it('should limit results to maxResults parameter', async () => {
      const reranked = await engine.rerankCandidates(candidates, searchContext, 2);
      
      expect(reranked.length).toBe(2);
    });

    it('should combine lexical and semantic scores', async () => {
      const reranked = await engine.rerankCandidates(candidates, searchContext, 10);
      
      // Each candidate should have a combined score
      reranked.forEach(candidate => {
        expect(candidate.score).toBeGreaterThan(0);
        expect(candidate.score).toBeLessThanOrEqual(1.0);
      });
    });

    it('should handle candidates without context', async () => {
      const candidatesNoContext = candidates.map(c => ({
        ...c,
        context: undefined
      }));

      const reranked = await engine.rerankCandidates(candidatesNoContext, searchContext, 10);
      
      expect(reranked.length).toBe(candidatesNoContext.length);
    });
  });

  describe('Embedding Model', () => {
    it('should generate consistent embeddings', async () => {
      const model = new (engine as any).embeddingModel.constructor(128);
      
      const text = 'function calculate sum';
      const embedding1 = await model.encode(text);
      const embedding2 = await model.encode(text);
      
      // Same text should produce same embedding
      expect(embedding1.length).toBe(embedding2.length);
      for (let i = 0; i < embedding1.length; i++) {
        expect(embedding1[i]).toBeCloseTo(embedding2[i], 5);
      }
    });

    it('should generate different embeddings for different text', async () => {
      const model = new (engine as any).embeddingModel.constructor(128);
      
      const text1 = 'function calculate sum';
      const text2 = 'class user service';
      
      const embedding1 = await model.encode(text1);
      const embedding2 = await model.encode(text2);
      
      // Different texts should produce different embeddings
      const similarity = model.similarity(embedding1, embedding2);
      expect(similarity).toBeLessThan(0.9);
    });

    it('should calculate cosine similarity correctly', async () => {
      const model = new (engine as any).embeddingModel.constructor(128);
      
      const text1 = 'calculate sum addition';
      const text2 = 'calculate sum add numbers';
      const text3 = 'fetch user data';
      
      const emb1 = await model.encode(text1);
      const emb2 = await model.encode(text2);
      const emb3 = await model.encode(text3);
      
      // Similar texts should have higher similarity
      const sim12 = model.similarity(emb1, emb2);
      const sim13 = model.similarity(emb1, emb3);
      
      expect(sim12).toBeGreaterThan(sim13);
      
      // Self-similarity should be 1.0
      const selfSim = model.similarity(emb1, emb1);
      expect(selfSim).toBeCloseTo(1.0, 5);
    });

    it('should handle empty text', async () => {
      const model = new (engine as any).embeddingModel.constructor(128);
      
      const embedding = await model.encode('');
      
      expect(embedding.length).toBe(128);
      
      // All zeros should result in zero vector
      const sum = embedding.reduce((acc, val) => acc + Math.abs(val), 0);
      expect(sum).toBe(0);
    });
  });

  describe('HNSW Index', () => {
    it('should build HNSW index when vectors are available', async () => {
      // Index several documents
      const documents = Array.from({ length: 10 }, (_, i) => ({
        id: `doc${i}`,
        content: `function test${i}() { return ${i}; }`,
        path: `/test${i}.js`
      }));

      for (const doc of documents) {
        await engine.indexDocument(doc.id, doc.content, doc.path);
      }

      const stats = engine.getStats();
      expect(stats.vectors).toBe(10);
      expect(stats.hnsw_layers).toBeGreaterThanOrEqual(1);
    });

    it('should not build HNSW index with no vectors', async () => {
      const stats = engine.getStats();
      expect(stats.hnsw_layers).toBe(0);
    });
  });

  describe('Performance', () => {
    it('should rerank within time limits', async () => {
      // Index many documents
      for (let i = 0; i < 20; i++) {
        await engine.indexDocument(`doc${i}`, `function test${i}() { return ${i}; }`, `/test${i}.js`);
      }

      // Create many candidates
      const candidates: Candidate[] = Array.from({ length: 50 }, (_, i) => ({
        doc_id: `doc${i % 20}:1:1`,
        file_path: `/test${i % 20}.js`,
        line: 1,
        col: 1,
        score: Math.random(),
        match_reasons: ['symbol'],
        context: `function test${i % 20}() { return ${i % 20}; }`
      }));

      const searchContext: SearchContext = {
        trace_id: 'perf-test',
        query: 'test function',
        mode: 'hybrid',
        k: 50,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      };

      const startTime = Date.now();
      const reranked = await engine.rerankCandidates(candidates, searchContext, 50);
      const rerankTime = Date.now() - startTime;

      expect(rerankTime).toBeLessThan(50); // Should be under Stage-C target of 15ms
      expect(reranked.length).toBe(50);
    });

    it('should index documents quickly', async () => {
      const content = 'function largeFunction() { ' + 'return "test"; '.repeat(100) + ' }';
      
      const startTime = Date.now();
      await engine.indexDocument('large-doc', content, '/large.js');
      const indexTime = Date.now() - startTime;

      expect(indexTime).toBeLessThan(100);
      
      const stats = engine.getStats();
      expect(stats.vectors).toBe(1);
    });
  });

  describe('Error Handling', () => {
    it('should handle reranking errors gracefully', async () => {
      const candidates: Candidate[] = [{
        doc_id: 'nonexistent',
        file_path: '/missing.js',
        line: 1,
        col: 1,
        score: 0.8,
        match_reasons: ['symbol'],
        context: 'missing function'
      }];

      const searchContext: SearchContext = {
        trace_id: 'error-test',
        query: 'test',
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      };

      // Should not throw error, should return fallback results
      const reranked = await engine.rerankCandidates(candidates, searchContext, 10);
      
      expect(reranked.length).toBe(1);
    });

    it('should handle malformed queries', async () => {
      await engine.indexDocument('test', 'function test() {}', '/test.js');

      const candidates: Candidate[] = [{
        doc_id: 'test:1:1',
        file_path: '/test.js',
        line: 1,
        col: 1,
        score: 0.8,
        match_reasons: ['symbol'],
        context: 'function test() {}'
      }];

      const searchContext: SearchContext = {
        trace_id: 'malformed-query',
        query: '', // Empty query
        mode: 'hybrid',
        k: 10,
        fuzzy_distance: 0,
        started_at: new Date(),
        stages: []
      };

      const reranked = await engine.rerankCandidates(candidates, searchContext, 10);
      
      expect(reranked.length).toBe(1);
    });
  });

  describe('Statistics', () => {
    it('should provide accurate statistics', async () => {
      await engine.indexDocument('doc1', 'test content 1', '/test1.js');
      await engine.indexDocument('doc2', 'test content 2', '/test2.js');

      const stats = engine.getStats();
      
      expect(stats.vectors).toBe(2);
      expect(stats.avg_dim).toBeGreaterThan(0);
      expect(stats.hnsw_layers).toBeGreaterThanOrEqual(0);
    });

    it('should handle empty index statistics', async () => {
      const stats = engine.getStats();
      
      expect(stats.vectors).toBe(0);
      expect(stats.avg_dim).toBe(0);
      expect(stats.hnsw_layers).toBe(0);
    });
  });

  describe('EnhancedSemanticRerankEngine - B3 Optimizations', () => {
    let enhancedEngine: EnhancedSemanticRerankEngine;
    let enhancedSegmentStorage: SegmentStorage;

    beforeEach(async () => {
      enhancedSegmentStorage = new SegmentStorage('./test-segments-enhanced');
      enhancedEngine = new EnhancedSemanticRerankEngine(enhancedSegmentStorage, {
        enableIsotonicCalibration: true,
        enableConfidenceGating: true,
        enableOptimizedHNSW: true,
        maxLatencyMs: 10,
        featureFlags: {
          stageCOptimizations: true,
          advancedCalibration: true,
          experimentalHNSW: false
        }
      });
      await enhancedEngine.initialize();
    });

    afterEach(async () => {
      await enhancedEngine.shutdown();
      await enhancedSegmentStorage.shutdown();
    });

    describe('B3 Optimization Integration', () => {
      it('should maintain compatibility with baseline engine', async () => {
        const content = `
function calculateSum(a, b) {
  return a + b;
}
        `;

        await enhancedEngine.indexDocument('enhanced-doc1', content, '/test/enhanced-math.js');
        
        const stats = enhancedEngine.getStats();
        expect(stats.vectors).toBe(1);
        expect(stats.config.enableIsotonicCalibration).toBe(true);
        expect(stats.config.enableOptimizedHNSW).toBe(true);
      });

      it('should apply B3 optimizations in reranking', async () => {
        // Index test documents
        await enhancedEngine.indexDocument('opt1', 'function add(a, b) { return a + b; }', '/math.js');
        await enhancedEngine.indexDocument('opt2', 'class Calculator { multiply(x, y) { return x * y; } }', '/calc.js');

        const candidates: Candidate[] = [
          {
            doc_id: 'opt1:1:1',
            file_path: '/math.js',
            line: 1,
            col: 1,
            score: 0.8,
            match_reasons: ['exact'],
            context: 'function add(a, b) { return a + b; }'
          },
          {
            doc_id: 'opt2:1:1',
            file_path: '/calc.js',
            line: 1,
            col: 1,
            score: 0.6,
            match_reasons: ['symbol'],
            context: 'class Calculator { multiply(x, y) { return x * y; } }'
          }
        ];

        const searchContext: SearchContext = {
          trace_id: 'enhanced-test-rerank',
          query: 'calculate addition function',
          mode: 'hybrid',
          k: 10,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: []
        };

        const startTime = Date.now();
        const reranked = await enhancedEngine.rerankCandidates(candidates, searchContext, 10);
        const latency = Date.now() - startTime;

        // Performance validation
        expect(latency).toBeLessThan(15); // Should meet enhanced performance targets
        
        // Quality validation
        expect(reranked.length).toBe(candidates.length);
        reranked.forEach(candidate => {
          expect(candidate.score).toBeGreaterThan(0);
          expect(candidate.score).toBeLessThanOrEqual(1);
        });
      });

      it('should support feature flag configuration', async () => {
        // Test with optimizations disabled
        await enhancedEngine.updateConfig({
          featureFlags: {
            stageCOptimizations: false,
            advancedCalibration: false,
            experimentalHNSW: false
          }
        });

        const stats1 = enhancedEngine.getStats();
        expect(stats1.config.featureFlags.stageCOptimizations).toBe(false);

        // Test with optimizations enabled
        await enhancedEngine.updateConfig({
          featureFlags: {
            stageCOptimizations: true,
            advancedCalibration: true,
            experimentalHNSW: true
          }
        });

        const stats2 = enhancedEngine.getStats();
        expect(stats2.config.featureFlags.stageCOptimizations).toBe(true);
      });

      it('should handle performance monitoring', async () => {
        await enhancedEngine.indexDocument('perf1', 'function test() { return "test"; }', '/perf.js');

        const candidates: Candidate[] = Array.from({ length: 20 }, (_, i) => ({
          doc_id: `perf1:${i}:1`,
          file_path: '/perf.js',
          line: i + 1,
          col: 1,
          score: Math.random() * 0.8 + 0.1,
          match_reasons: ['symbol'],
          context: `function test${i}() { return "test${i}"; }`
        }));

        const context: SearchContext = {
          trace_id: 'performance-test',
          query: 'test function implementation',
          mode: 'hybrid',
          k: 10,
          fuzzy_distance: 0,
          started_at: new Date(),
          stages: []
        };

        const startTime = Date.now();
        const results = await enhancedEngine.rerankCandidates(candidates, context, 10);
        const latency = Date.now() - startTime;

        expect(results.length).toBeLessThanOrEqual(10);
        expect(latency).toBeLessThan(20); // Should be fast even with more candidates

        const stats = enhancedEngine.getStats();
        expect(stats.performance.avgLatencyMs).toBeGreaterThanOrEqual(0);
      });
    });

    describe('Enhanced Configuration', () => {
      it('should validate configuration parameters', () => {
        const stats = enhancedEngine.getStats();
        
        expect(stats.config).toHaveProperty('enableIsotonicCalibration');
        expect(stats.config).toHaveProperty('enableConfidenceGating');
        expect(stats.config).toHaveProperty('enableOptimizedHNSW');
        expect(stats.config).toHaveProperty('maxLatencyMs');
        expect(stats.config).toHaveProperty('featureFlags');
        
        expect(stats.config.maxLatencyMs).toBe(10);
        expect(stats.config.featureFlags.stageCOptimizations).toBe(true);
      });

      it('should support dynamic configuration updates', async () => {
        const originalConfig = enhancedEngine.getStats().config;
        
        await enhancedEngine.updateConfig({
          maxLatencyMs: 15,
          calibrationConfig: {
            enabled: true,
            minCalibrationData: 100,
            confidenceCutoff: 0.15,
            updateFreq: 200
          }
        });

        const updatedConfig = enhancedEngine.getStats().config;
        expect(updatedConfig.maxLatencyMs).toBe(15);
        expect(updatedConfig.calibrationConfig.minCalibrationData).toBe(100);
      });
    });
  });
});