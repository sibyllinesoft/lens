#!/usr/bin/env node
/**
 * Lens CLI - Command line interface for lens operations
 * Implements Phase A2 migration commands and other CLI operations
 */

import { Command } from 'commander';
import { handleMigrateCommand, MigrationManager } from './core/migration-manager.js';
import { SERVER_API_VERSION, SERVER_INDEX_VERSION } from './core/version-manager.js';
import { handleRCCommand } from './core/rc-release-manager.js';

const program = new Command();

program
  .name('lens')
  .description('Lens code search engine CLI')
  .version('1.0.0-rc.1');

// Migration command
program
  .command('migrate-index')
  .description('Migrate index from one version to another')
  .option('--from <version>', 'Source version (e.g., v1)', 'v1')
  .option('--to <version>', 'Target version (e.g., v1)', 'v1')
  .option('--dry-run', 'Show what would be migrated without making changes', false)
  .option('--verbose', 'Show detailed migration progress', false)
  .action(async (options) => {
    try {
      await handleMigrateCommand({
        from: options.from,
        to: options.to,
        dryRun: options.dryRun,
        verbose: options.verbose,
      });
    } catch (error) {
      console.error('Migration failed:', error);
      process.exit(1);
    }
  });

// List migrations command
program
  .command('list-migrations')
  .description('List all available migrations')
  .action(() => {
    const migrations = MigrationManager.listMigrations();
    
    if (migrations.length === 0) {
      console.log('No migrations available');
      return;
    }

    console.log('Available migrations:');
    console.log('==================');
    
    migrations.forEach(migration => {
      console.log(`\n📦 ${migration.name}`);
      console.log(`   From: ${migration.fromVersion}`);
      console.log(`   To: ${migration.toVersion}`);
      console.log(`   Description: ${migration.description}`);
    });
  });

// Version command
program
  .command('version')
  .description('Show lens version and compatibility information')
  .action(() => {
    console.log('🔍 Lens Code Search Engine');
    console.log('==========================');
    console.log(`CLI Version: 1.0.0-rc.1`);
    console.log(`API Version: ${SERVER_API_VERSION}`);
    console.log(`Index Version: ${SERVER_INDEX_VERSION}`);
  });

// Build command for Phase A3
program
  .command('build')
  .description('Build lens with security artifacts')
  .option('--sbom', 'Generate Software Bill of Materials', false)
  .option('--sast', 'Enable SAST security scanning', false)
  .option('--lock', 'Use locked dependency versions', false)
  .option('--container', 'Build container image', false)
  .action(async (options) => {
    const { execSync } = await import('child_process');
    const { join } = await import('path');
    
    console.log('🔨 Starting secure build...');
    
    // Build script arguments
    const args = [];
    if (options.sbom) args.push('--sbom');
    if (options.sast) args.push('--sast');
    if (options.lock) args.push('--lock');
    if (options.container) args.push('--container');
    
    try {
      const scriptPath = join(process.cwd(), 'scripts', 'build-secure.sh');
      const command = `"${scriptPath}" ${args.join(' ')}`;
      
      console.log(`Executing: ${command}`);
      
      execSync(command, {
        stdio: 'inherit',
        cwd: process.cwd(),
        env: {
          ...process.env,
          LENS_VERSION: '1.0.0-rc.1',
        }
      });
      
      console.log('✅ Secure build completed successfully');
    } catch (error) {
      console.error('❌ Build failed:', error);
      process.exit(1);
    }
  });

// Phase D RC release commands
program
  .command('cut-rc')
  .description('Cut RC release with all artifacts and security scanning')
  .option('--version <version>', 'RC version (e.g., v1.0.0-rc.1)', '1.0.0-rc.1')
  .option('--env <env>', 'Target environment (rc|production)', 'rc')
  .option('--output-dir <dir>', 'Output directory for artifacts', './release-output')
  .option('--no-sbom', 'Skip SBOM generation')
  .option('--no-sast', 'Skip SAST scanning')
  .option('--no-container', 'Skip container build')
  .option('--no-provenance', 'Skip build provenance')
  .action(async (options) => {
    await handleRCCommand('cut-rc', options);
  });

program
  .command('compat-drill')
  .description('Run compatibility drill against previous versions')
  .option('--version <version>', 'RC version', '1.0.0-rc.1')
  .option('--previous-versions <versions>', 'Comma-separated list of previous versions', 'v0.9.0,v0.9.1')
  .action(async (options) => {
    const previousVersions = options.previousVersions.split(',').map((v: string) => v.trim());
    await handleRCCommand('compat-drill', { ...options, previousVersions });
  });

program
  .command('nightly-validation')
  .description('Run comprehensive nightly validation across repo slices')
  .option('--duration <minutes>', 'Test duration in minutes', '120')
  .option('--repo-types <types>', 'Comma-separated repo types', 'backend,frontend,monorepo')
  .option('--languages <langs>', 'Comma-separated languages', 'typescript,javascript,python,go,rust')
  .option('--size-categories <sizes>', 'Comma-separated size categories', 'small,medium,large')
  .action(async (options) => {
    await handleRCCommand('nightly-validation', options);
  });

program
  .command('check-signoff')
  .description('Check three-night sign-off criteria for production promotion')
  .option('--version <version>', 'RC version', '1.0.0-rc.1')
  .action(async (options) => {
    await handleRCCommand('check-signoff', options);
  });

program
  .command('promote')
  .description('Promote RC to production release')
  .option('--version <version>', 'RC version to promote', '1.0.0-rc.1')
  .option('--force', 'Force promotion without sign-off checks', false)
  .action(async (options) => {
    if (!options.force) {
      console.log('⚠️  Production promotion requires sign-off verification...');
      console.log('Use --force to bypass checks (not recommended)');
    }
    await handleRCCommand('promote', options);
  });

// Error handling
program.exitOverride();

try {
  program.parse();
} catch (error) {
  console.error('CLI Error:', error);
  process.exit(1);
}