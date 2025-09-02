/**
 * Migration framework for lens index and schema changes
 * Implements Phase A2 requirements for version migration support
 */

import { ApiVersion, IndexVersion } from '../types/api.js';

export interface MigrationResult {
  success: boolean;
  message: string;
  fromVersion: IndexVersion;
  toVersion: IndexVersion;
  migratedItems?: number;
  warnings?: string[];
  errors?: string[];
}

export interface Migration {
  name: string;
  fromVersion: IndexVersion;
  toVersion: IndexVersion;
  description: string;
  migrate(): Promise<MigrationResult>;
}

/**
 * Migration from v1 to v1 (no-op for current version)
 */
class V1ToV1Migration implements Migration {
  name = 'v1-to-v1';
  fromVersion: IndexVersion = 'v1';
  toVersion: IndexVersion = 'v1';
  description = 'No-op migration for v1 to v1 (same version)';

  async migrate(): Promise<MigrationResult> {
    return {
      success: true,
      message: 'No migration needed - source and target versions are identical',
      fromVersion: this.fromVersion,
      toVersion: this.toVersion,
      migratedItems: 0,
    };
  }
}

// Registry of available migrations
const MIGRATION_REGISTRY = new Map<string, Migration>([
  ['v1-to-v1', new V1ToV1Migration()],
]);

/**
 * Migration manager for handling index migrations
 */
export class MigrationManager {
  /**
   * Get available migrations for a version transition
   */
  static getAvailableMigrations(from: IndexVersion, to: IndexVersion): Migration[] {
    const key = `${from}-to-${to}`;
    const migration = MIGRATION_REGISTRY.get(key);
    return migration ? [migration] : [];
  }

  /**
   * Check if migration is available
   */
  static canMigrate(from: IndexVersion, to: IndexVersion): boolean {
    return this.getAvailableMigrations(from, to).length > 0;
  }

  /**
   * Perform migration from one version to another
   */
  static async migrateIndex(
    from: IndexVersion,
    to: IndexVersion,
    options: {
      dryRun?: boolean;
      verbose?: boolean;
    } = {}
  ): Promise<MigrationResult> {
    const { dryRun = false, verbose = false } = options;

    // Validate versions
    if (!this.isValidVersion(from)) {
      return {
        success: false,
        message: `Unknown source version: ${from}`,
        fromVersion: from,
        toVersion: to,
        errors: [`Invalid source version: ${from}`],
      };
    }

    if (!this.isValidVersion(to)) {
      return {
        success: false,
        message: `Unknown target version: ${to}`,
        fromVersion: from,
        toVersion: to,
        errors: [`Invalid target version: ${to}`],
      };
    }

    // Find migration path
    const migrations = this.getAvailableMigrations(from, to);
    
    if (migrations.length === 0) {
      return {
        success: false,
        message: `No migration path available from ${from} to ${to}`,
        fromVersion: from,
        toVersion: to,
        errors: [`No migration available for ${from} -> ${to}`],
      };
    }

    // Execute migration
    const migration = migrations[0];
    
    if (verbose) {
      console.log(`Running migration: ${migration.name}`);
      console.log(`Description: ${migration.description}`);
      if (dryRun) {
        console.log('DRY RUN MODE - no changes will be made');
      }
    }

    if (dryRun) {
      return {
        success: true,
        message: `Dry run: ${migration.description}`,
        fromVersion: from,
        toVersion: to,
        migratedItems: 0,
        warnings: ['Dry run mode - no actual migration performed'],
      };
    }

    try {
      const result = await migration.migrate();
      
      if (verbose) {
        console.log(`Migration completed: ${result.success ? 'SUCCESS' : 'FAILED'}`);
        console.log(`Message: ${result.message}`);
        if (result.migratedItems !== undefined) {
          console.log(`Items migrated: ${result.migratedItems}`);
        }
      }

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown migration error';
      return {
        success: false,
        message: `Migration failed: ${errorMessage}`,
        fromVersion: from,
        toVersion: to,
        errors: [errorMessage],
      };
    }
  }

  /**
   * List all available migrations
   */
  static listMigrations(): Array<{
    name: string;
    fromVersion: IndexVersion;
    toVersion: IndexVersion;
    description: string;
  }> {
    return Array.from(MIGRATION_REGISTRY.values()).map(migration => ({
      name: migration.name,
      fromVersion: migration.fromVersion,
      toVersion: migration.toVersion,
      description: migration.description,
    }));
  }

  /**
   * Validate if a version is supported
   */
  private static isValidVersion(version: string): version is IndexVersion {
    return version === 'v1'; // Only v1 supported currently
  }
}

/**
 * CLI command interface for migrations
 */
export interface MigrateCommandOptions {
  from: string;
  to: string;
  dryRun?: boolean;
  verbose?: boolean;
}

/**
 * CLI command handler for migrations
 */
export async function handleMigrateCommand(options: MigrateCommandOptions): Promise<void> {
  const { from, to, dryRun = false, verbose = false } = options;

  // Validate version format
  if (!from.startsWith('v') || !to.startsWith('v')) {
    console.error('Error: Version must be in format vX (e.g., v1, v2)');
    process.exit(1);
  }

  // Strip 'v' prefix for internal handling
  const fromVersion = from.slice(1) as IndexVersion;
  const toVersion = to.slice(1) as IndexVersion;

  try {
    const result = await MigrationManager.migrateIndex(fromVersion, toVersion, {
      dryRun,
      verbose,
    });

    if (result.success) {
      console.log('✅ Migration completed successfully');
      console.log(result.message);
      
      if (result.warnings && result.warnings.length > 0) {
        console.log('\n⚠️  Warnings:');
        result.warnings.forEach(warning => console.log(`  - ${warning}`));
      }
    } else {
      console.error('❌ Migration failed');
      console.error(result.message);
      
      if (result.errors && result.errors.length > 0) {
        console.error('\n🚨 Errors:');
        result.errors.forEach(error => console.error(`  - ${error}`));
      }
      
      process.exit(1);
    }
  } catch (error) {
    console.error('❌ Unexpected migration error:', error);
    process.exit(1);
  }
}