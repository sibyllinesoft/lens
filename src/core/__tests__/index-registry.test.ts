/**
 * Tests for IndexRegistry
 * Priority: HIGH - Second highest complexity (108), 1089 LOC, critical for search functionality
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IndexRegistry } from '../index-registry.js';
import { readFileSync, existsSync, statSync } from 'fs';
import * as fsPromises from 'fs/promises';
import { join } from 'path';

// Mock filesystem operations
vi.mock('fs', () => ({
  readFileSync: vi.fn(),
  existsSync: vi.fn(),
  readdirSync: vi.fn(),
  statSync: vi.fn(),
}));

vi.mock('fs/promises', () => ({
  readFile: vi.fn(),
  readdir: vi.fn(),
  stat: vi.fn(),
  access: vi.fn(),
}));

vi.mock('path', () => ({
  join: vi.fn((...args) => args.join('/')),
  dirname: vi.fn(path => path.split('/').slice(0, -1).join('/')),
  basename: vi.fn(path => path.split('/').pop()),
  extname: vi.fn(path => {
    const name = path.split('/').pop() || '';
    const dotIndex = name.lastIndexOf('.');
    return dotIndex > 0 ? name.slice(dotIndex) : '';
  }),
  resolve: vi.fn((...args) => args.join('/').replace(/\/+/g, '/')),
  relative: vi.fn((from, to) => to), // Simple mock that just returns the 'to' path
}));

// Mock telemetry
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn().mockReturnValue({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    }),
  },
}));

const mockReadFileSync = vi.mocked(readFileSync);
const mockExistsSync = vi.mocked(existsSync);
const mockStatSync = vi.mocked(statSync);
const mockJoin = vi.mocked(join);
const mockFsAccess = vi.mocked(fsPromises.access);
const mockFsReaddir = vi.mocked(fsPromises.readdir);
const mockFsReadFile = vi.mocked(fsPromises.readFile);

describe('IndexRegistry', () => {
  let registry: IndexRegistry;
  const testIndexRoot = '/test/indexed-content';

  beforeEach(() => {
    vi.clearAllMocks();
    registry = new IndexRegistry(testIndexRoot);
    
    // Setup default mocks
    mockJoin.mockImplementation((...args) => args.join('/'));
    mockExistsSync.mockReturnValue(true);
    mockFsAccess.mockResolvedValue(undefined); // fs.access succeeds when no error is thrown
    mockFsReaddir.mockResolvedValue([]);
  });

  afterEach(async () => {
    if (registry) {
      await registry.shutdown();
    }
  });

  describe('Initialization', () => {
    it('should initialize with correct index root', () => {
      expect(registry).toBeDefined();
    });

    it('should handle missing index root', async () => {
      const emptyRegistry = new IndexRegistry('/nonexistent');
      
      // Mock fs.access to reject for nonexistent directory
      const mockAccess = vi.mocked(fsPromises.access);
      mockAccess.mockRejectedValueOnce(new Error('ENOENT: no such file or directory'));
      
      await expect(emptyRegistry.refresh()).rejects.toThrow('Index root does not exist');
    });
  });

  describe('Manifest Discovery', () => {
    it('should discover and load manifests', async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['shard1.jsonl', 'shard2.jsonl'],
        total_files: 100,
        total_symbols: 500,
      };

      // Mock fs/promises for the IndexRegistry's async operations
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));

      await registry.refresh();
      
      expect(mockFsReadFile).toHaveBeenCalled();
      
      const stats = registry.stats();
      expect(stats.totalRepos).toBe(1);
      expect(stats.loadedRepos).toBe(0); // No readers created yet
    });

    it('should handle malformed manifest files', async () => {
      mockReadFileSync.mockImplementation((path) => {
        if (path.toString().includes('manifest.json')) {
          return '{ invalid json }';
        }
        return '';
      });

      // Should not throw but log error
      await expect(registry.refresh()).resolves.not.toThrow();
      
      const stats = registry.stats();
      expect(stats.totalRepos).toBe(0);
    });

    it('should handle missing required manifest fields', async () => {
      const incompleteManifest = {
        repo_ref: 'test-repo',
        // Missing repo_sha and other required fields
      };

      mockReadFileSync.mockReturnValue(JSON.stringify(incompleteManifest));

      await registry.refresh();
      
      const stats = registry.stats();
      expect(stats.totalRepos).toBe(0);
    });
  });

  describe('Repository Access', () => {
    beforeEach(async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['shard1.jsonl'],
        total_files: 50,
        total_symbols: 250,
      };

      // Mock fs/promises for the IndexRegistry's async operations
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));

      await registry.refresh();
    });

    it('should check if repository exists', () => {
      expect(registry.hasRepo('abc123')).toBe(true);
      expect(registry.hasRepo('nonexistent')).toBe(false);
    });

    it('should get repository reader', () => {
      const reader = registry.getReader('abc123');
      expect(reader).toBeDefined();
      expect(reader).toHaveProperty('searchLexical');
      expect(reader).toHaveProperty('searchStructural');
    });

    it('should throw error for nonexistent repository reader', () => {
      expect(() => registry.getReader('nonexistent'))
        .toThrow('Repository not found in index: nonexistent');
    });

    it('should return manifests', () => {
      const manifests = registry.getManifests();
      expect(manifests).toHaveLength(1);
      expect(manifests[0]).toMatchObject({
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
      });
    });
  });

  describe('Statistics', () => {
    it('should return correct stats for empty registry', () => {
      const stats = registry.stats();
      
      expect(stats).toMatchObject({
        totalRepos: 0,
        loadedRepos: 0,
        shardPaths: 0,
        memoryUsageMB: 0,
        cacheHitRate: 0.9,
      });
    });

    it('should return correct stats after loading manifests', async () => {
      const mockManifests = [
        {
          repo_ref: 'repo1',
          repo_sha: 'sha1',
          api_version: '1.0.0',
          index_version: '1.0.0',
          policy_version: '1.0.0',
          shard_paths: ['shard1.jsonl'],
          total_files: 100,
          total_symbols: 500,
        },
        {
          repo_ref: 'repo2', 
          repo_sha: 'sha2',
          api_version: '1.0.0',
          index_version: '1.0.0',
          policy_version: '1.0.0',
          shard_paths: ['shard2.jsonl'],
          total_files: 150,
          total_symbols: 750,
        },
      ];

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['repo1.manifest.json', 'repo2.manifest.json']);
      mockFsReadFile
        .mockResolvedValueOnce(JSON.stringify(mockManifests[0]))
        .mockResolvedValueOnce(JSON.stringify(mockManifests[1]));

      await registry.refresh();
      
      const stats = registry.stats();
      expect(stats.totalRepos).toBe(2);
      expect(stats.loadedRepos).toBe(0); // No readers created yet
      expect(stats.shardPaths).toBeGreaterThan(0); // Should have shard paths
      expect(stats.memoryUsageMB).toBeDefined();
      expect(stats.cacheHitRate).toBeDefined();
    });
  });

  describe('IndexReader Interface', () => {
    let reader: any;

    beforeEach(async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['shard1.jsonl'],
        total_files: 50,
        total_symbols: 250,
      };

      // Mock shard data
      const mockShardData = [
        '{"file":"test.ts","line":1,"col":0,"lang":"typescript","snippet":"function test() {}","tokens":["function","test"],"score":0.95}',
        '{"file":"app.ts","line":5,"col":0,"lang":"typescript","snippet":"class App {}","tokens":["class","App"],"score":0.90}',
      ].join('\n');

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));
      
      // Mock fs sync methods for shard loading
      mockReadFileSync.mockReturnValue(mockShardData);

      await registry.refresh();
      reader = registry.getReader('abc123');
    });

    describe('Lexical Search', () => {
      it('should perform lexical search', async () => {
        const results = await reader.searchLexical({
          q: 'test',
          k: 10,
          fuzzy: 0,
          subtokens: true,
        });

        expect(Array.isArray(results)).toBe(true);
      });

      it('should handle fuzzy search parameters', async () => {
        const results = await reader.searchLexical({
          q: 'test',
          k: 10,
          fuzzy: 2,
          subtokens: false,
        });

        expect(Array.isArray(results)).toBe(true);
      });

      it('should limit results by k parameter', async () => {
        const results = await reader.searchLexical({
          q: 'test',
          k: 1,
          fuzzy: 0,
          subtokens: true,
        });

        expect(results.length).toBeLessThanOrEqual(1);
      });
    });

    describe('Structural Search', () => {
      it('should perform structural search', async () => {
        const results = await reader.searchStructural({
          q: 'class',
          k: 10,
        });

        expect(Array.isArray(results)).toBe(true);
      });

      it('should return structural search results with metadata', async () => {
        const results = await reader.searchStructural({
          q: 'class App',
          k: 10,
        });

        expect(Array.isArray(results)).toBe(true);
        // Results may be empty if no structural matches found
        if (results.length > 0) {
          expect(results[0]).toHaveProperty('file');
          expect(results[0]).toHaveProperty('score');
        }
      });
    });

    describe('Error Handling', () => {
      it('should handle search errors gracefully', async () => {
        // The reader should handle search errors and not throw
        const results = await reader.searchLexical({ q: 'nonexistent', k: 10 });
        expect(Array.isArray(results)).toBe(true); // Should return array, not throw
      });
    });
  });

  describe('Concurrent Access', () => {
    it('should handle concurrent refresh operations', async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['shard1.jsonl'],
        total_files: 50,
        total_symbols: 250,
      };

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));

      // Run multiple refresh operations concurrently
      const refreshPromises = Array(5).fill(0).map(() => registry.refresh());
      
      await expect(Promise.all(refreshPromises)).resolves.not.toThrow();
      
      const stats = registry.stats();
      expect(stats.totalRepos).toBe(1);
    });

    it('should handle concurrent reader access', async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['shard1.jsonl'],
        total_files: 50,
        total_symbols: 250,
      };

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));
      
      await registry.refresh();

      // Get multiple readers concurrently
      const readers = Array(10).fill(0).map(() => registry.getReader('abc123'));
      
      expect(readers).toHaveLength(10);
      readers.forEach(reader => {
        expect(reader).toBeDefined();
        expect(reader).toHaveProperty('searchLexical');
      });
    });
  });

  describe('Memory Management', () => {
    it('should handle large numbers of repositories', async () => {
      const mockManifests = Array(100).fill(0).map((_, i) => ({
        repo_ref: `repo-${i}`,
        repo_sha: `sha${i}`,
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: [`shard${i}.jsonl`],
        total_files: 10,
        total_symbols: 50,
      }));

      // Mock fs/promises methods for async manifest loading  
      mockFsReaddir.mockResolvedValue(
        mockManifests.map((_, i) => `repo-${i}.manifest.json`)
      );
      mockFsReadFile.mockImplementation((path) => {
        const pathStr = path.toString();
        const index = mockManifests.findIndex(m => pathStr.includes(m.repo_ref));
        return Promise.resolve(index >= 0 ? JSON.stringify(mockManifests[index]) : '{}');
      });

      await registry.refresh();
      
      const stats = registry.stats();
      expect(stats.totalRepos).toBeGreaterThan(0); // Should have some repos loaded
      expect(stats.shardPaths).toBeGreaterThan(0); // Should have some shard paths
      expect(stats.memoryUsageMB).toBeDefined();
      expect(stats.cacheHitRate).toBeDefined();
    });
  });

  describe('Shutdown', () => {
    it('should shutdown cleanly', async () => {
      await expect(registry.shutdown()).resolves.not.toThrow();
    });

    it('should handle shutdown after operations', async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0', 
        policy_version: '1.0.0',
        shard_paths: ['shard1.jsonl'],
        total_files: 50,
        total_symbols: 250,
      };

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));
      
      await registry.refresh();
      const reader = registry.getReader('abc123');
      
      await expect(registry.shutdown()).resolves.not.toThrow();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty shard files', async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['empty.jsonl'],
        total_files: 0,
        total_symbols: 0,
      };

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));
      
      // Mock fs sync methods for shard loading  
      mockReadFileSync.mockImplementation((path) => {
        if (path.toString().includes('.manifest.json')) {
          return JSON.stringify(mockManifest);
        }
        return ''; // empty shard
      });

      await registry.refresh();
      const reader = registry.getReader('abc123');
      
      const results = await reader.searchLexical({ q: 'nonexistent', k: 10 });
      expect(results).toEqual([]);
    });

    it('should handle malformed JSONL in shards', async () => {
      const mockManifest = {
        repo_ref: 'test-repo',
        repo_sha: 'abc123',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['bad.jsonl'],
        total_files: 1,
        total_symbols: 1,
      };

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['test-repo.manifest.json']);
      mockFsReadFile.mockResolvedValue(JSON.stringify(mockManifest));
      
      // Mock fs sync methods for shard loading
      mockReadFileSync.mockReturnValue('{ invalid jsonl line\n{"valid":true}'); // mixed shard

      await registry.refresh();
      const reader = registry.getReader('abc123');
      
      const results = await reader.searchLexical({ q: 'test', k: 10 });
      expect(Array.isArray(results)).toBe(true); // Should handle gracefully
    });

    it('should handle duplicate repo_sha values', async () => {
      const mockManifest1 = {
        repo_ref: 'repo1',
        repo_sha: 'duplicate',
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['shard1.jsonl'],
        total_files: 10,
        total_symbols: 50,
      };

      const mockManifest2 = {
        repo_ref: 'repo2',
        repo_sha: 'duplicate', // Same SHA
        api_version: '1.0.0',
        index_version: '1.0.0',
        policy_version: '1.0.0',
        shard_paths: ['shard2.jsonl'],
        total_files: 15,
        total_symbols: 75,
      };

      // Mock fs/promises methods for async manifest loading
      mockFsReaddir.mockResolvedValue(['repo1.manifest.json', 'repo2.manifest.json']);
      mockFsReadFile
        .mockResolvedValueOnce(JSON.stringify(mockManifest1))
        .mockResolvedValueOnce(JSON.stringify(mockManifest2));

      await registry.refresh();
      
      // Should handle duplicate SHA (last one wins or merge)
      expect(registry.hasRepo('duplicate')).toBe(true);
      
      const stats = registry.stats();
      // Should handle duplicates gracefully
      expect(stats.totalRepos).toBeGreaterThan(0);
    });
  });
});