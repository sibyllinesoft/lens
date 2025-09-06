#!/usr/bin/env bun

/**
 * Phase 3 - Precision/Semantic Pack CLI
 * Command-line interface for Phase 3 execution and management
 * Target: +2-3 nDCG@10 points while maintaining Recall@50
 */

import { parseArgs } from 'node:util';
import { promises as fs } from 'fs';
import path from 'path';
import { Phase3PrecisionPack, type Phase3Config } from '../core/phase3-precision-pack.js';
import { Phase3PatternPackEngine } from '../core/phase3-pattern-packs.js';

interface CLIOptions {
  help: boolean;
  execute: boolean;
  config: boolean;
  patterns: boolean;
  rollback: boolean;
  'index-root': string;
  'output-dir': string;
  'api-url': string;
  'config-file': string;
  'source-file': string;
  'language': string;
  verbose: boolean;
}

const HELP_TEXT = `
🎯 Phase 3 - Precision/Semantic Pack CLI

USAGE:
  bun run src/scripts/phase3-cli.ts [OPTIONS]

COMMANDS:
  --execute              Execute complete Phase 3 implementation
  --config               Show Phase 3 configuration and acceptance gates  
  --patterns             Find patterns in source file (requires --source-file)
  --rollback             Perform Phase 3 rollback to previous state

OPTIONS:
  --index-root <path>    Index root directory (default: ./indexed-content)
  --output-dir <path>    Output directory for results (default: ./phase3-results)
  --api-url <url>        API base URL (default: http://localhost:3001)
  --config-file <path>   Custom configuration file (JSON)
  --source-file <path>   Source file for pattern analysis
  --language <lang>      Language for pattern analysis (typescript, python, etc.)
  --verbose              Enable verbose output
  --help                 Show this help message

EXAMPLES:
  # Execute Phase 3 with default configuration
  bun run src/scripts/phase3-cli.ts --execute
  
  # Execute with custom configuration
  bun run src/scripts/phase3-cli.ts --execute --config-file ./my-config.json
  
  # Show configuration and acceptance gates
  bun run src/scripts/phase3-cli.ts --config
  
  # Find patterns in a TypeScript file
  bun run src/scripts/phase3-cli.ts --patterns --source-file ./src/example.ts --language typescript
  
  # Perform rollback
  bun run src/scripts/phase3-cli.ts --rollback

PHASE 3 OVERVIEW:
  Phase 3 focuses on precision improvements through:
  
  🔧 Stage B Enhancements:
    - Pattern packs: ctor_impl, test_func_names, config_keys  
    - Expanded LSIF coverage (multi-workspace + vendored dirs)
    - 1.25x LRU bytes budget, 1.2x batch query size
  
  🧠 Stage C Improvements:
    - Isotonic calibration (isotonic_v1)
    - Lower NL threshold (0.35), higher ANN search (k=220, efSearch=96)
    - Enhanced features: path_prior_residual, subtoken_jaccard, struct_distance, docBM25
  
  📊 Target Metrics:
    - nDCG@10: +2-3 points improvement (≥0.758 from 0.743 baseline)
    - Recall@50: Maintain ≥0.856 (Phase 2 level)
    - Span coverage: ≥98%
    - Hard negative leakage: ≤+1.5% absolute
`;

function parseCliArgs(): CLIOptions {
  try {
    const { values } = parseArgs({
      args: process.argv.slice(2),
      options: {
        help: { type: 'boolean', default: false },
        execute: { type: 'boolean', default: false },
        config: { type: 'boolean', default: false },
        patterns: { type: 'boolean', default: false },
        rollback: { type: 'boolean', default: false },
        'index-root': { type: 'string', default: './indexed-content' },
        'output-dir': { type: 'string', default: './phase3-results' },
        'api-url': { type: 'string', default: 'http://localhost:3001' },
        'config-file': { type: 'string', default: '' },
        'source-file': { type: 'string', default: '' },
        'language': { type: 'string', default: 'typescript' },
        verbose: { type: 'boolean', default: false },
      },
      allowPositionals: false,
    });
    
    return values as CLIOptions;
  } catch (error) {
    console.error('❌ Invalid arguments:', error instanceof Error ? error.message : error);
    console.log(HELP_TEXT);
    process.exit(1);
  }
}

async function loadCustomConfig(configFile: string): Promise<Partial<Phase3Config> | undefined> {
  if (!configFile) return undefined;
  
  try {
    const configData = await fs.readFile(configFile, 'utf-8');
    const config = JSON.parse(configData);
    console.log(`📋 Loaded custom configuration from ${configFile}`);
    return config;
  } catch (error) {
    console.error(`❌ Failed to load config file ${configFile}:`, error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

async function executePhase3(options: CLIOptions): Promise<void> {
  console.log('🎯 Starting Phase 3 - Precision/Semantic Pack execution...');
  console.log('📊 Target: +2-3 nDCG@10 points while maintaining Recall@50\n');
  
  try {
    // Load custom configuration if provided
    const customConfig = await loadCustomConfig(options['config-file']);
    
    // Initialize Phase 3 orchestrator
    const phase3 = new Phase3PrecisionPack(
      options['index-root'],
      options['output-dir'],
      options['api-url']
    );
    
    // Display configuration
    if (options.verbose) {
      console.log('📋 Phase 3 Configuration:');
      const config = phase3.getDefaultConfig();
      console.log(JSON.stringify(config, null, 2));
      console.log();
    }
    
    console.log('⏳ Executing Phase 3 implementation...');
    const startTime = Date.now();
    
    // Execute Phase 3
    const results = await phase3.execute(customConfig);
    
    const duration = Date.now() - startTime;
    console.log(`✅ Phase 3 execution completed in ${duration}ms\n`);
    
    // Display results summary
    console.log('📊 Phase 3 Results Summary:');
    console.log('═══════════════════════════════');
    console.log(`📈 nDCG@10 Improvement: ${results.ndcg_improvement_points.toFixed(2)} points`);
    console.log(`   Baseline: ${results.baseline_ndcg_10.toFixed(3)}`);
    console.log(`   Current:  ${results.new_ndcg_10.toFixed(3)}`);
    console.log();
    console.log(`🎯 Recall@50 Status: ${results.recall_maintained ? '✅ MAINTAINED' : '❌ REGRESSED'}`);
    console.log(`   Baseline: ${results.baseline_recall_50.toFixed(3)}`);
    console.log(`   Current:  ${results.new_recall_50.toFixed(3)}`);
    console.log();
    console.log(`📋 Span Coverage: ${results.span_coverage_pct.toFixed(1)}%`);
    console.log(`⚠️  Hard Negative Leakage: ${results.hard_negative_leakage_pct.toFixed(1)}%`);
    console.log();
    console.log(`🚦 Acceptance Gates: ${results.acceptance_gates_passed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log(`🔧 Tripwires Status: ${results.tripwires_status === 'green' ? '✅ GREEN' : results.tripwires_status === 'yellow' ? '⚠️ YELLOW' : '🚨 RED'}`);
    console.log(`🚀 Promotion Ready: ${results.promotion_ready ? '✅ YES' : '❌ NO'}`);
    console.log();
    
    // Stage latencies
    console.log('⏱️  Stage Latencies:');
    console.log(`   Stage A p95: ${results.stage_latencies.stage_a_p95}ms`);
    console.log(`   Stage B p95: ${results.stage_latencies.stage_b_p95}ms`);
    console.log(`   Stage C p95: ${results.stage_latencies.stage_c_p95}ms`);
    console.log(`   E2E p95:     ${results.stage_latencies.e2e_p95}ms`);
    console.log();
    
    // Final decision
    if (results.promotion_ready) {
      console.log('🎉 Phase 3 SUCCESS - Ready for promotion!');
      console.log('📦 Evidence package generated in:', options['output-dir']);
      console.log('🏷️  Ready to tag: v1.2-precision-pack');
    } else {
      console.log('⚠️  Phase 3 acceptance criteria not fully met');
      console.log('🔄 Rollback capability available');
      
      if (results.tripwires_status === 'red') {
        console.log('🚨 TRIPWIRES TRIGGERED - Automatic rollback recommended');
      }
    }
    
  } catch (error) {
    console.error('❌ Phase 3 execution failed:', error instanceof Error ? error.message : error);
    console.log('🔄 Attempting rollback...');
    
    try {
      const phase3 = new Phase3PrecisionPack(options['index-root'], options['output-dir'], options['api-url']);
      await phase3.performRollback();
      console.log('✅ Rollback completed successfully');
    } catch (rollbackError) {
      console.error('🚨 Rollback failed:', rollbackError instanceof Error ? rollbackError.message : rollbackError);
    }
    
    process.exit(1);
  }
}

async function showConfiguration(options: CLIOptions): Promise<void> {
  console.log('📋 Phase 3 - Precision/Semantic Pack Configuration\n');
  
  try {
    const phase3 = new Phase3PrecisionPack(options['index-root'], options['output-dir'], options['api-url']);
    const engine = new Phase3PatternPackEngine();
    
    // Default configuration
    const config = phase3.getDefaultConfig();
    console.log('🔧 Default Configuration:');
    console.log('═══════════════════════════');
    console.log('Stage B (Symbol/AST Coverage):');
    console.log(`  Pattern Packs: ${config.stage_b.pattern_packs.join(', ')}`);
    console.log(`  LRU Budget: ${config.stage_b.lru_bytes_budget_multiplier}x`);
    console.log(`  Batch Size: ${config.stage_b.batch_query_size_multiplier}x`);
    console.log(`  Multi-workspace LSIF: ${config.stage_b.enable_multi_workspace_lsif}`);
    console.log(`  Vendored dirs LSIF: ${config.stage_b.enable_vendored_dirs_lsif}`);
    console.log();
    console.log('Stage C (Semantic Rerank):');
    console.log(`  Calibration: ${config.stage_c.calibration}`);
    console.log(`  NL Threshold: ${config.stage_c.gate.nl_threshold}`);
    console.log(`  Min Candidates: ${config.stage_c.gate.min_candidates}`);
    console.log(`  Confidence Cutoff: ${config.stage_c.gate.confidence_cutoff}`);
    console.log(`  ANN k: ${config.stage_c.ann.k}`);
    console.log(`  ANN efSearch: ${config.stage_c.ann.efSearch}`);
    console.log(`  Features: ${config.stage_c.features.join(', ')}`);
    console.log();
    
    // Acceptance gates
    const gates = phase3.getAcceptanceGates();
    console.log('🚦 Acceptance Gates:');
    console.log('══════════════════');
    console.log(`  nDCG@10 Min Improvement: +${gates.ndcg_10_min_improvement_points} points`);
    console.log(`  nDCG@10 Target: ≥${gates.ndcg_10_target.toFixed(3)}`);
    console.log(`  Recall@50 Maintenance: ≥${gates.recall_50_maintenance_threshold.toFixed(3)}`);
    console.log(`  Span Coverage: ≥${gates.span_coverage_min_pct}%`);
    console.log(`  Hard Negative Leakage: ≤${gates.hard_negative_leakage_max_pct}%`);
    console.log(`  Significance Level: p<${gates.significance_p_value}`);
    console.log();
    
    // Tripwire checks
    const tripwires = phase3.getTripwireChecks();
    console.log('🔧 Tripwire Checks:');
    console.log('═══════════════════');
    console.log(`  Span Coverage: ≥${tripwires.span_coverage_min_pct}%`);
    console.log(`  LSIF Coverage: ≥${tripwires.lsif_coverage_min_pct}%`);
    console.log(`  Semantic Timeout: ≤${tripwires.semantic_rerank_timeout_ms}ms p95`);
    console.log(`  Candidate Explosion: ≤${tripwires.candidate_explosion_max_multiplier}x`);
    console.log();
    
    // Pattern pack statistics
    const patternStats = engine.getStatistics();
    console.log('📦 Pattern Pack Statistics:');
    console.log('══════════════════════════');
    console.log(`  Total Packs: ${patternStats.total_packs}`);
    console.log(`  Total Patterns: ${patternStats.total_patterns}`);
    console.log(`  Languages Supported: ${patternStats.languages_supported.join(', ')}`);
    console.log('  Patterns by Language:');
    for (const [lang, count] of Object.entries(patternStats.patterns_by_language)) {
      console.log(`    ${lang}: ${count} patterns`);
    }
    
  } catch (error) {
    console.error('❌ Failed to retrieve configuration:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

async function findPatterns(options: CLIOptions): Promise<void> {
  if (!options['source-file']) {
    console.error('❌ --source-file is required for pattern analysis');
    process.exit(1);
  }
  
  console.log(`🔍 Finding patterns in ${options['source-file']} (${options.language})\n`);
  
  try {
    // Read source file
    const sourceCode = await fs.readFile(options['source-file'], 'utf-8');
    
    // Initialize pattern engine
    const engine = new Phase3PatternPackEngine();
    
    // Find patterns
    const patterns = await engine.findPatterns(
      sourceCode,
      options['source-file'],
      options.language
    );
    
    if (patterns.length === 0) {
      console.log('❌ No patterns found');
      return;
    }
    
    console.log(`✅ Found ${patterns.length} patterns:\n`);
    
    // Group patterns by type
    const patternsByType = patterns.reduce((acc, pattern) => {
      if (!acc[pattern.pattern_name]) {
        acc[pattern.pattern_name] = [];
      }
      acc[pattern.pattern_name].push(pattern);
      return acc;
    }, {} as Record<string, typeof patterns>);
    
    // Display results
    for (const [patternType, matches] of Object.entries(patternsByType)) {
      console.log(`📋 ${patternType} (${matches.length} matches):`);
      console.log('─'.repeat(50));
      
      matches.forEach((match, index) => {
        console.log(`  ${index + 1}. Line ${match.line}:${match.col}`);
        console.log(`     Match: ${match.match_text.replace(/\n/g, '\\n')}`);
        console.log(`     Symbol: ${match.symbol_kind}`);
        console.log(`     Confidence: ${(match.confidence * 100).toFixed(1)}%`);
        
        if (options.verbose && match.ast_context) {
          console.log(`     Context:\n${match.ast_context.split('\n').map(line => `       ${line}`).join('\n')}`);
        }
        console.log();
      });
    }
    
  } catch (error) {
    console.error('❌ Pattern analysis failed:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

async function performRollback(options: CLIOptions): Promise<void> {
  console.log('🔄 Performing Phase 3 rollback...\n');
  
  try {
    const phase3 = new Phase3PrecisionPack(options['index-root'], options['output-dir'], options['api-url']);
    
    await phase3.performRollback();
    
    console.log('✅ Phase 3 rollback completed successfully');
    console.log('🔧 Stage B and Stage C policies reverted to previous state');
    console.log('📊 System ready for re-execution or alternative approaches');
    
  } catch (error) {
    console.error('❌ Rollback failed:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

async function main(): Promise<void> {
  const options = parseCliArgs();
  
  if (options.help) {
    console.log(HELP_TEXT);
    return;
  }
  
  // Validate that at least one command is specified
  const commands = [options.execute, options.config, options.patterns, options.rollback];
  const commandCount = commands.filter(Boolean).length;
  
  if (commandCount === 0) {
    console.error('❌ No command specified. Use --help for usage information.');
    process.exit(1);
  }
  
  if (commandCount > 1) {
    console.error('❌ Multiple commands specified. Please specify only one command at a time.');
    process.exit(1);
  }
  
  try {
    if (options.execute) {
      await executePhase3(options);
    } else if (options.config) {
      await showConfiguration(options);
    } else if (options.patterns) {
      await findPatterns(options);
    } else if (options.rollback) {
      await performRollback(options);
    }
    
  } catch (error) {
    console.error('❌ CLI execution failed:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('🚨 Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('🚨 Unhandled rejection:', reason);
  process.exit(1);
});

// Execute main function if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { main, parseCliArgs, executePhase3, showConfiguration, findPatterns, performRollback };