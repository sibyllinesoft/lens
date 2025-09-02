/**
 * Simplified stable async database fixtures for testing
 * Phase A4 requirement for stable database testing infrastructure
 */

import { beforeEach, afterEach } from 'vitest';
import { SegmentStorage } from '../../storage/segments.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

interface TestDatabase {
  storage: SegmentStorage;
  tempDir: string;
  cleanup: () => Promise<void>;
}

export class TestDatabaseFixture {
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

  async cleanupAll(): Promise<void> {
    const cleanupPromises = Array.from(this.activeDatabases).map(db => 
      this.teardown(db)
    );
    await Promise.all(cleanupPromises);
    this.activeDatabases.clear();
  }
}

export function setupDatabaseFixtures() {
  const fixture = new TestDatabaseFixture();
  let database: TestDatabase;

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
  };
}

export type { TestDatabase };