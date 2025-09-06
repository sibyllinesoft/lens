/**
 * Gemma Migration System - Entry Point
 * Complete alignment, calibration, and ANN retuning system for Lens search
 */

export { VectorAlignment, ScoreAlignment, AlignmentValidator } from './alignment-system.js';
export { CalibrationSystem, BatchCalibrationProcessor } from './calibration-system.js';
export { HNSWTuner, QuantizationOptimizer } from './ann-tuning-system.js';
export { MatryoshkaRouter, RouterOptimizer } from './matryoshka-router.js';
export { SLAScoreboard } from './sla-scoreboard.js';
export { MigrationOrchestrator } from './migration-orchestrator.js';

// Type exports
export type { AlignmentConfig } from './alignment-system.js';
export type { CalibrationConfig } from './calibration-system.js';
export type { ANNConfig } from './ann-tuning-system.js';
export type { RouterConfig } from './matryoshka-router.js';
export type { ScoreboardConfig } from './sla-scoreboard.js';
export type { MigrationConfig, MigrationResult } from './migration-orchestrator.js';

/**
 * Quick-start migration execution
 */
import { MigrationOrchestrator, MigrationResult } from './migration-orchestrator.js';

export async function executeGemmaMigration(outputDir?: string): Promise<MigrationResult> {
  console.log('🚀 Starting Gemma Migration System');
  
  const orchestrator = new MigrationOrchestrator({
    phases: {
      alignment: true,
      calibration: true,
      annTuning: true,
      matryoshka: true,
      evaluation: true
    },
    models: {
      baseline: 'ada-002',
      candidates: ['gemma-768', 'gemma-256']
    },
    outputDir: outputDir || './gemma-migration-results',
    shadowMode: true
  });

  try {
    const result = await orchestrator.executeMigration();
    
    console.log('\n📋 Migration Summary:');
    console.log(`✅ Overall Success: ${result.overallSuccess}`);
    console.log(`📈 Promotion Ready: ${result.promotionRecommendation.ready}`);
    console.log(`🎯 Recommended Model: ${result.promotionRecommendation.model}`);
    console.log(`🔒 Config Hash: ${result.configHash}`);
    
    if (result.promotionRecommendation.blockers.length > 0) {
      console.log('\n⚠️  Promotion Blockers:');
      result.promotionRecommendation.blockers.forEach(blocker => {
        console.log(`  - ${blocker}`);
      });
    }
    
    console.log('\n📁 Generated Artifacts:');
    Object.entries(result.artifacts).forEach(([key, path]) => {
      if (path) console.log(`  ${key}: ${path}`);
    });
    
    return result;
    
  } catch (error) {
    console.error('❌ Migration failed:', error);
    throw error;
  }
}

/**
 * CLI interface (if run directly)
 */
if (require.main === module) {
  const outputDir = process.argv[2] || './gemma-migration-results';
  
  executeGemmaMigration(outputDir)
    .then(() => {
      console.log('\n🎉 Migration completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n💥 Migration failed:', error);
      process.exit(1);
    });
}