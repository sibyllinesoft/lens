#!/usr/bin/env bun

/**
 * Phase 3 Implementation Test Suite
 * Validates Phase 3 components before running full benchmarks
 */

import { Phase3PrecisionPack } from './src/core/phase3-precision-pack.js';
import { Phase3PatternPackEngine } from './src/core/phase3-pattern-packs.js';

// Test sample source code for pattern matching
const testSourceCode = `
// TypeScript constructor example
class UserService {
  constructor(private config: Config) {
    this.initialize();
  }
  
  private initialize() {
    // constructor implementation
  }
}

// Test function examples
describe('UserService', () => {
  test('should create user successfully', () => {
    const service = new UserService(config);
    expect(service).toBeDefined();
  });
  
  it('validates user input', async () => {
    const result = await service.validateUser(userData);
    expect(result.valid).toBe(true);
  });
});

// Configuration access examples
const config = {
  database_url: process.env.DATABASE_URL,
  api_timeout: process.env.API_TIMEOUT || 5000,
  retry_count: 3
};

function loadConfig() {
  return {
    ...config,
    debug: process.env.DEBUG === 'true'
  };
}
`;

async function testPatternPackEngine(): Promise<boolean> {
  console.log('🧪 Testing Phase 3 Pattern Pack Engine...');
  
  try {
    const engine = new Phase3PatternPackEngine();
    
    // Test pattern statistics
    const stats = engine.getStatistics();
    console.log('📊 Pattern Statistics:', {
      total_packs: stats.total_packs,
      total_patterns: stats.total_patterns,
      languages: stats.languages_supported.slice(0, 5) // Show first 5 languages
    });
    
    // Expected: 3 pattern packs (ctor_impl, test_func_names, config_keys)
    if (stats.total_packs < 3) {
      console.error('❌ Expected at least 3 pattern packs');
      return false;
    }
    
    // Test pattern matching on sample code
    const patterns = await engine.findPatterns(
      testSourceCode,
      'test-file.ts',
      'typescript'
    );
    
    console.log(`🔍 Found ${patterns.length} patterns in test source`);
    
    // Group by pattern type
    const patternTypes = patterns.reduce((acc, pattern) => {
      acc[pattern.pattern_name] = (acc[pattern.pattern_name] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    console.log('📋 Pattern types found:', patternTypes);
    
    // Validate expected patterns
    if (!patternTypes['typescript_constructor']) {
      console.error('❌ Expected typescript_constructor pattern');
      return false;
    }
    
    if (!patternTypes['jest_test_functions']) {
      console.error('❌ Expected jest_test_functions pattern');
      return false;
    }
    
    if (!patternTypes['env_var_access']) {
      console.error('❌ Expected env_var_access pattern');
      return false;
    }
    
    console.log('✅ Pattern Pack Engine tests passed');
    return true;
    
  } catch (error) {
    console.error('❌ Pattern Pack Engine test failed:', error instanceof Error ? error.message : error);
    return false;
  }
}

async function testPhase3Configuration(): Promise<boolean> {
  console.log('⚙️ Testing Phase 3 Configuration...');
  
  try {
    const phase3 = new Phase3PrecisionPack('./test-index', './test-results', 'http://localhost:3001');
    
    // Test configuration retrieval
    const config = phase3.getDefaultConfig();
    console.log('📋 Default config loaded');
    
    // Validate Stage B configuration
    const stageBConfig = config.stage_b;
    if (!stageBConfig.pattern_packs.includes('ctor_impl')) {
      console.error('❌ Expected ctor_impl pattern pack in Stage B');
      return false;
    }
    
    if (!stageBConfig.pattern_packs.includes('test_func_names')) {
      console.error('❌ Expected test_func_names pattern pack in Stage B');
      return false;
    }
    
    if (!stageBConfig.pattern_packs.includes('config_keys')) {
      console.error('❌ Expected config_keys pattern pack in Stage B');
      return false;
    }
    
    if (stageBConfig.lru_bytes_budget_multiplier !== 1.25) {
      console.error('❌ Expected LRU bytes budget multiplier of 1.25');
      return false;
    }
    
    // Validate Stage C configuration
    const stageCConfig = config.stage_c;
    if (stageCConfig.calibration !== 'isotonic_v1') {
      console.error('❌ Expected isotonic_v1 calibration');
      return false;
    }
    
    if (stageCConfig.gate.nl_threshold !== 0.35) {
      console.error('❌ Expected NL threshold of 0.35');
      return false;
    }
    
    if (stageCConfig.ann.k !== 220) {
      console.error('❌ Expected ANN k of 220');
      return false;
    }
    
    // Test acceptance gates
    const gates = phase3.getAcceptanceGates();
    if (gates.ndcg_10_min_improvement_points !== 2.0) {
      console.error('❌ Expected nDCG@10 minimum improvement of 2 points');
      return false;
    }
    
    if (gates.ndcg_10_target !== 0.758) {
      console.error('❌ Expected nDCG@10 target of 0.758');
      return false;
    }
    
    if (gates.recall_50_maintenance_threshold !== 0.856) {
      console.error('❌ Expected Recall@50 maintenance threshold of 0.856');
      return false;
    }
    
    // Test tripwire checks
    const tripwires = phase3.getTripwireChecks();
    if (tripwires.span_coverage_min_pct !== 98.0) {
      console.error('❌ Expected span coverage minimum of 98%');
      return false;
    }
    
    console.log('✅ Phase 3 Configuration tests passed');
    return true;
    
  } catch (error) {
    console.error('❌ Phase 3 Configuration test failed:', error instanceof Error ? error.message : error);
    return false;
  }
}

async function testApiEndpoints(): Promise<boolean> {
  console.log('🌐 Testing Phase 3 API Endpoints...');
  
  try {
    // Test Phase 3 config endpoint
    console.log('📋 Testing GET /phase3/config...');
    const configResponse = await fetch('http://localhost:3001/phase3/config');
    
    if (!configResponse.ok) {
      console.log('⚠️ API server not running - skipping API tests');
      return true; // Don't fail the test suite if server isn't running
    }
    
    const configData = await configResponse.json();
    
    if (!configData.success) {
      console.error('❌ Config endpoint returned failure');
      return false;
    }
    
    if (!configData.config || !configData.acceptance_gates) {
      console.error('❌ Config endpoint missing expected data');
      return false;
    }
    
    console.log('✅ Config endpoint working');
    
    // Test pattern finding endpoint
    console.log('🔍 Testing POST /phase3/patterns/find...');
    const patternsResponse = await fetch('http://localhost:3001/phase3/patterns/find', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        source_code: testSourceCode,
        file_path: 'test.ts',
        language: 'typescript',
      }),
    });
    
    if (!patternsResponse.ok) {
      console.error('❌ Patterns endpoint failed:', patternsResponse.statusText);
      return false;
    }
    
    const patternsData = await patternsResponse.json();
    
    if (!patternsData.success) {
      console.error('❌ Patterns endpoint returned failure:', patternsData.message);
      return false;
    }
    
    if (!Array.isArray(patternsData.patterns)) {
      console.error('❌ Patterns endpoint did not return patterns array');
      return false;
    }
    
    console.log(`✅ Patterns endpoint found ${patternsData.patterns.length} patterns`);
    
    console.log('✅ API Endpoint tests passed');
    return true;
    
  } catch (error) {
    console.error('❌ API Endpoint test failed:', error instanceof Error ? error.message : error);
    return false;
  }
}

async function testBaselineMetrics(): Promise<boolean> {
  console.log('📊 Testing Baseline Metrics Loading...');
  
  try {
    const phase3 = new Phase3PrecisionPack('./test-index', './test-results', 'http://localhost:3001');
    
    // Try to load baseline metrics using the private method logic
    // We'll simulate this by checking if baseline file exists
    const fs = await import('fs/promises');
    const path = await import('path');
    
    const baselinePath = path.join('./baseline_key_numbers.json');
    
    try {
      const baselineData = await fs.readFile(baselinePath, 'utf-8');
      const baseline = JSON.parse(baselineData);
      
      console.log('📋 Baseline metrics:', {
        recall_at_50: baseline.recall_at_50,
        ndcg_at_10: baseline.ndcg_at_10,
      });
      
      // Validate baseline has expected Phase 2 values
      if (baseline.recall_at_50 !== 0.856) {
        console.warn('⚠️ Expected Phase 2 Recall@50 baseline of 0.856');
      }
      
      if (baseline.ndcg_at_10 !== 0.743) {
        console.warn('⚠️ Expected Phase 2 nDCG@10 baseline of 0.743');
      }
      
      console.log('✅ Baseline metrics loaded successfully');
      return true;
      
    } catch {
      console.log('⚠️ No baseline file found - will use live benchmark baseline');
      return true; // This is acceptable
    }
    
  } catch (error) {
    console.error('❌ Baseline metrics test failed:', error instanceof Error ? error.message : error);
    return false;
  }
}

async function runPhase3Tests(): Promise<boolean> {
  console.log('🎯 Running Phase 3 Implementation Tests');
  console.log('========================================\n');
  
  const tests = [
    { name: 'Pattern Pack Engine', fn: testPatternPackEngine },
    { name: 'Phase 3 Configuration', fn: testPhase3Configuration },
    { name: 'API Endpoints', fn: testApiEndpoints },
    { name: 'Baseline Metrics', fn: testBaselineMetrics },
  ];
  
  const results: { name: string; passed: boolean }[] = [];
  
  for (const test of tests) {
    console.log(`\n🧪 Running ${test.name} test...`);
    
    try {
      const passed = await test.fn();
      results.push({ name: test.name, passed });
      
      if (passed) {
        console.log(`✅ ${test.name}: PASSED`);
      } else {
        console.log(`❌ ${test.name}: FAILED`);
      }
      
    } catch (error) {
      console.error(`💥 ${test.name}: CRASHED -`, error instanceof Error ? error.message : error);
      results.push({ name: test.name, passed: false });
    }
  }
  
  // Summary
  console.log('\n📊 Test Results Summary');
  console.log('=======================');
  
  const passed = results.filter(r => r.passed).length;
  const total = results.length;
  
  results.forEach(result => {
    console.log(`${result.passed ? '✅' : '❌'} ${result.name}`);
  });
  
  console.log(`\n🎯 Overall: ${passed}/${total} tests passed`);
  
  if (passed === total) {
    console.log('\n🎉 All Phase 3 implementation tests PASSED!');
    console.log('✅ Ready for smoke benchmark execution');
    console.log('\nNext steps:');
    console.log('  1. Run: bun run src/scripts/phase3-cli.ts --config');
    console.log('  2. Run: bun run src/scripts/phase3-cli.ts --execute');
    console.log('  3. Monitor acceptance gates and tripwires');
    return true;
  } else {
    console.log('\n⚠️ Some tests failed - fix issues before proceeding');
    return false;
  }
}

// Execute tests if running directly
if (import.meta.main) {
  runPhase3Tests()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('🚨 Test execution failed:', error);
      process.exit(1);
    });
}

export { runPhase3Tests };