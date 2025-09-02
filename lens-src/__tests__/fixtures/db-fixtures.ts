/**
 * Stable async database fixtures for testing
 * Phase A4 requirement for stable database testing infrastructure
 * Uses :memory: or per-test database isolation
 */

import { beforeEach, afterEach } from 'vitest';
import { SegmentStorage } from '../../storage/segments.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import type { SegmentType } from '../../types/core.js';

interface TestDatabase {
  storage: SegmentStorage;
  tempDir: string;
  cleanup: () => Promise<void>;
}

interface DatabaseFixture {
  setup(): Promise<TestDatabase>;
  teardown(db: TestDatabase): Promise<void>;
}

/**
 * Memory-based fixture for fast, isolated tests
 * Uses temporary directories that are cleaned up after each test
 */
export class MemoryDatabaseFixture implements DatabaseFixture {
  private activeDatabases: Set<TestDatabase> = new Set();

  async setup(): Promise<TestDatabase> {
    // Create unique temporary directory for this test
    const tempDir = await fs.promises.mkdtemp(
      path.join(os.tmpdir(), 'lens-test-')
    );

    // Initialize storage with temp directory
    const storage = new SegmentStorage(tempDir);

    const database: TestDatabase = {
      storage,
      tempDir,
      cleanup: async () => {
        this.activeDatabases.delete(database);
        await this.cleanupTempDir(tempDir);
      }
    };

    this.activeDatabases.add(database);
    return database;
  }

  async teardown(db: TestDatabase): Promise<void> {
    await db.cleanup();
  }

  private async cleanupTempDir(tempDir: string): Promise<void> {
    try {
      if (fs.existsSync(tempDir)) {
        await fs.promises.rm(tempDir, { recursive: true, force: true });
      }
    } catch (error) {
      console.warn(`Failed to cleanup temp dir ${tempDir}:`, error);
    }
  }

  /**
   * Emergency cleanup for all active databases
   * Called during test suite cleanup
   */
  async cleanupAll(): Promise<void> {
    const cleanupPromises = Array.from(this.activeDatabases).map(db => 
      this.teardown(db)
    );
    await Promise.all(cleanupPromises);
    this.activeDatabases.clear();
  }
}

/**
 * Per-test fixture that ensures complete isolation
 * Each test gets its own database instance
 */
export class IsolatedDatabaseFixture implements DatabaseFixture {
  private static instance: MemoryDatabaseFixture = new MemoryDatabaseFixture();

  async setup(): Promise<TestDatabase> {
    return IsolatedDatabaseFixture.instance.setup();
  }

  async teardown(db: TestDatabase): Promise<void> {
    return IsolatedDatabaseFixture.instance.teardown(db);
  }
}

/**
 * Shared fixture for tests that can share database state
 * More efficient but requires careful test design
 */
export class SharedDatabaseFixture implements DatabaseFixture {
  private static sharedDatabase: TestDatabase | null = null;
  private static referenceCount = 0;

  async setup(): Promise<TestDatabase> {
    SharedDatabaseFixture.referenceCount++;

    if (!SharedDatabaseFixture.sharedDatabase) {
      const tempDir = await fs.promises.mkdtemp(
        path.join(os.tmpdir(), 'lens-shared-')
      );

      const storage = new SegmentStorage(tempDir);

      SharedDatabaseFixture.sharedDatabase = {
        storage,
        tempDir,
        cleanup: async () => {
          if (SharedDatabaseFixture.sharedDatabase) {
            await fs.promises.rm(SharedDatabaseFixture.sharedDatabase.tempDir, {
              recursive: true,
              force: true
            });
            SharedDatabaseFixture.sharedDatabase = null;
          }
        }
      };
    }

    return SharedDatabaseFixture.sharedDatabase;
  }

  async teardown(db: TestDatabase): Promise<void> {
    SharedDatabaseFixture.referenceCount--;

    if (SharedDatabaseFixture.referenceCount <= 0) {
      if (SharedDatabaseFixture.sharedDatabase) {
        await SharedDatabaseFixture.sharedDatabase.cleanup();
        SharedDatabaseFixture.sharedDatabase = null;
      }
      SharedDatabaseFixture.referenceCount = 0;
    }
  }
}

/**
 * Test data seeding utilities
 */
export class TestDataSeeder {
  constructor(private storage: SegmentStorage) {}

  /**
   * Seed basic test segments
   */
  async seedBasicSegments(): Promise<void> {
    // Create lexical segment with test data
    const lexicalSegment = await this.storage.createSegment(
      'test-lexical',
      'lexical',
      1024 * 1024 // 1MB
    );

    // Seed with sample lexical data
    const sampleLexicalData = {
      terms: ['function', 'class', 'const', 'let', 'var'],
      positions: [
        { term: 'function', files: ['test1.js', 'test2.js'], positions: [[1, 1], [5, 10]] },
        { term: 'class', files: ['test1.js'], positions: [[10, 1]] },
      ]
    };

    // Note: writeToSegment method would need to be implemented in SegmentStorage
    // For now, this is a placeholder for test data seeding

    // Create symbol segment with test data
    const symbolSegment = await this.storage.createSegment(
      'test-symbols',
      'symbols',
      1024 * 1024 // 1MB
    );

    const sampleSymbolData = {
      symbols: [
        { name: 'TestClass', type: 'class', file: 'test1.js', line: 10, col: 1 },
        { name: 'testFunction', type: 'function', file: 'test1.js', line: 1, col: 1 },
        { name: 'CONSTANT', type: 'variable', file: 'test2.js', line: 5, col: 10 },
      ]
    };

    await this.storage.writeToSegment('test-symbols', 0, Buffer.from(JSON.stringify(sampleSymbolData)));
  }

  /**
   * Seed performance test data
   */
  async seedPerformanceData(itemCount: number = 10000): Promise<void> {
    const performanceSegment = await this.storage.createSegment(
      'test-performance',
      'lexical',
      16 * 1024 * 1024 // 16MB
    );

    // Generate large dataset
    const terms = Array.from({ length: itemCount }, (_, i) => `term${i}`);
    const positions = terms.map((term, i) => ({
      term,
      files: [`file${i % 100}.js`], // 100 files with multiple terms
      positions: Array.from({ length: Math.ceil(Math.random() * 10) + 1 }, () => [
        Math.floor(Math.random() * 1000) + 1,
        Math.floor(Math.random() * 100) + 1
      ])
    }));

    const performanceData = { terms, positions };
    await this.storage.writeToSegment('test-performance', 0, Buffer.from(JSON.stringify(performanceData)));
  }

  /**
   * Seed Unicode and edge case data
   */
  async seedUnicodeData(): Promise<void> {
    const unicodeSegment = await this.storage.createSegment(
      'test-unicode',
      'lexical',
      1024 * 1024 // 1MB
    );

    const unicodeData = {
      terms: [
        '测试', // Chinese
        'тест', // Cyrillic
        '🎉', // Emoji
        'café', // Accented characters
        '𝕳𝖊𝖑𝖑𝖔', // Mathematical script (surrogate pairs)
        'שלום', // Hebrew (RTL)
      ],
      positions: [
        { term: '测试', files: ['unicode.js'], positions: [[1, 1]] },
        { term: 'тест', files: ['unicode.js'], positions: [[2, 1]] },
        { term: '🎉', files: ['unicode.js'], positions: [[3, 1]] },
        { term: 'café', files: ['unicode.js'], positions: [[4, 1]] },
        { term: '𝕳𝖊𝖑𝖑𝖔', files: ['unicode.js'], positions: [[5, 1]] },
        { term: 'שלום', files: ['unicode.js'], positions: [[6, 1]] },
      ]
    };

    await this.storage.writeToSegment('test-unicode', 0, Buffer.from(JSON.stringify(unicodeData)));
  }
}

/**
 * Async test helpers
 */
export class AsyncTestHelpers {
  /**
   * Wait for all pending async operations to complete
   */
  static async waitForAsyncOperations(): Promise<void> {
    // Allow microtasks to complete
    await new Promise(resolve => setImmediate(resolve));
  }

  /**
   * Run operation with timeout
   */
  static async withTimeout<T>(
    operation: Promise<T>,
    timeoutMs: number = 5000,
    errorMessage: string = 'Operation timed out'
  ): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(errorMessage)), timeoutMs);
    });

    return Promise.race([operation, timeoutPromise]);
  }

  /**
   * Retry operation with exponential backoff
   */
  static async retry<T>(
    operation: () => Promise<T>,
    maxAttempts: number = 3,
    baseDelayMs: number = 100
  ): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;

        if (attempt === maxAttempts) {
          throw lastError;
        }

        // Exponential backoff
        const delayMs = baseDelayMs * Math.pow(2, attempt - 1);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }

    throw lastError!;
  }

  /**
   * Execute operations in parallel with concurrency limit
   */
  static async parallelLimit<T>(
    operations: (() => Promise<T>)[],
    concurrencyLimit: number = 5
  ): Promise<T[]> {
    const results: T[] = [];
    const executing: Promise<void>[] = [];

    for (const operation of operations) {
      const promise = operation().then(result => {
        results.push(result);
      });

      executing.push(promise);

      if (executing.length >= concurrencyLimit) {
        await Promise.race(executing);
        const completedIndex = executing.findIndex(p => 
          p === Promise.resolve(p).catch(() => {})
        );
        if (completedIndex >= 0) {
          executing.splice(completedIndex, 1);
        }
      }
    }

    await Promise.all(executing);
    return results;
  }
}

/**
 * Test suite setup helpers
 */
export function setupDatabaseFixtures(fixtureType: 'memory' | 'isolated' | 'shared' = 'isolated') {
  let fixture: DatabaseFixture;
  let database: TestDatabase;

  switch (fixtureType) {
    case 'memory':
    case 'isolated':
      fixture = new IsolatedDatabaseFixture();
      break;
    case 'shared':
      fixture = new SharedDatabaseFixture();
      break;
    default:
      throw new Error(`Unknown fixture type: ${fixtureType}`);
  }

  beforeEach(async () => {
    database = await fixture.setup();
  });

  afterEach(async () => {
    if (database) {
      await fixture.teardown(database);
    }
  });

  return {
    getDatabase: () => database,
    getStorage: () => database.storage,
    getSeeder: () => new TestDataSeeder(database.storage),
  };
}

// Global cleanup for memory fixtures
let globalMemoryFixture: MemoryDatabaseFixture | null = null;

export function setupGlobalDatabaseCleanup() {
  if (!globalMemoryFixture) {
    globalMemoryFixture = new MemoryDatabaseFixture();
    
    // Setup global cleanup
    if (typeof process !== 'undefined') {
      const cleanup = async () => {
        if (globalMemoryFixture) {
          await globalMemoryFixture.cleanupAll();
        }
      };

      process.on('exit', cleanup);
      process.on('SIGINT', cleanup);
      process.on('SIGTERM', cleanup);
      process.on('uncaughtException', cleanup);
    }
  }
}

// Export commonly used types and utilities
export type { TestDatabase, DatabaseFixture };