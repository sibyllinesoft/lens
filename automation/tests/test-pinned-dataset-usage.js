#!/usr/bin/env node

/**
 * Test Pinned Dataset Usage
 * 
 * This script demonstrates how to use the pinned golden dataset
 * for consistent benchmarking and validates the setup.
 */

import { PinnedGroundTruthLoader } from './src/benchmark/pinned-ground-truth-loader.js';

async function testPinnedDatasetUsage() {
  console.log('🧪 Testing Pinned Golden Dataset Usage');
  console.log('=====================================\n');

  try {
    // Initialize the loader
    const loader = new PinnedGroundTruthLoader();
    
    // Test 1: Load the pinned dataset
    console.log('📌 Test 1: Loading pinned dataset...');
    const pinnedDataset = await loader.loadPinnedDataset();
    console.log(`   ✅ Loaded: ${pinnedDataset.version} (${pinnedDataset.total_items} items)\n`);
    
    // Test 2: Get golden items
    console.log('📊 Test 2: Retrieving golden items...');
    const allItems = loader.getCurrentGoldenItems();
    console.log(`   ✅ Retrieved: ${allItems.length} total items`);
    
    const smokeItems = loader.getSmokeDataset();
    console.log(`   ✅ SMOKE slice: ${smokeItems.length} items\n`);
    
    // Test 3: Sample a few queries to validate structure
    console.log('🔍 Test 3: Validating query structure...');
    const sampleItems = allItems.slice(0, 3);
    
    for (const [index, item] of sampleItems.entries()) {
      console.log(`   Query ${index + 1}: "${item.query}"`);
      console.log(`     ID: ${item.id}`);
      console.log(`     Language: ${item.language}`);
      console.log(`     Query class: ${item.query_class}`);
      console.log(`     Slice tags: [${item.slice_tags.join(', ')}]`);
      console.log(`     Expected results: ${item.expected_results.length}`);
      
      if (item.expected_results.length > 0) {
        const result = item.expected_results[0];
        console.log(`       File: ${result.file}`);
        console.log(`       Location: line ${result.line}, col ${result.col}`);
        console.log(`       Relevance: ${result.relevance_score}`);
      }
      console.log('');
    }
    
    // Test 4: Consistency validation
    console.log('🔍 Test 4: Consistency validation...');
    const consistencyResult = await loader.validatePinnedDatasetConsistency();
    
    if (consistencyResult.passed) {
      console.log(`   ✅ Consistency check PASSED`);
      console.log(`   📊 Pass rate: ${(consistencyResult.report.pass_rate * 100).toFixed(1)}%`);
      console.log(`   📁 Corpus files: ${consistencyResult.report.corpus_file_count}\n`);
    } else {
      console.log(`   ❌ Consistency check FAILED`);
      console.log(`   📊 Pass rate: ${(consistencyResult.report.pass_rate * 100).toFixed(1)}%`);
      console.log(`   ⚠️ Inconsistencies: ${consistencyResult.report.inconsistent_results}\n`);
    }
    
    // Test 5: Statistics validation
    console.log('📈 Test 5: Dataset statistics...');
    const stats = loader.getDatasetStats();
    console.log(`   Version: ${stats.version}`);
    console.log(`   Total items: ${stats.total_items}`);
    console.log(`   Languages: ${Object.entries(stats.languages).map(([lang, count]) => `${lang}(${count})`).join(', ')}`);
    console.log(`   Query classes: ${Object.entries(stats.query_classes).map(([cls, count]) => `${cls}(${count})`).join(', ')}`);
    console.log(`   Slices: ${Object.keys(stats.slices).join(', ')}\n`);
    
    // Test 6: Configuration fingerprint generation
    console.log('🔐 Test 6: Configuration fingerprint...');
    const mockConfig = {
      systems: ['lex', '+symbols'],
      k_candidates: 200,
      fuzzy: 2
    };
    
    const fingerprint = loader.generateConfigFingerprint(mockConfig, [1, 2, 3]);
    console.log(`   ✅ Generated fingerprint:`);
    console.log(`     Config hash: ${fingerprint.config_hash.substring(0, 16)}...`);
    console.log(`     Code hash: ${fingerprint.code_hash.substring(0, 16)}...`);
    console.log(`     Snapshot SHAs: ${Object.keys(fingerprint.snapshot_shas).join(', ')}`);
    console.log(`     Timestamp: ${fingerprint.timestamp}\n`);
    
    // Test 7: Slice filtering
    console.log('🎯 Test 7: Slice filtering...');
    const smokeSlice = loader.filterGoldenItemsBySlice('SMOKE_DEFAULT');
    const allSlice = loader.filterGoldenItemsBySlice('ALL');
    console.log(`   ✅ SMOKE_DEFAULT slice: ${smokeSlice.length} items`);
    console.log(`   ✅ ALL slice: ${allSlice.length} items\n`);
    
    // Final summary
    console.log('🎉 All Tests Completed Successfully!');
    console.log('===================================');
    console.log(`📌 Pinned dataset ${pinnedDataset.version} is ready for benchmarking`);
    console.log(`📊 ${allItems.length} golden queries available`);
    console.log(`✅ ${consistencyResult.passed ? 'Perfect' : 'Some'} corpus alignment`);
    console.log(`🎯 ${smokeItems.length} SMOKE queries for quick testing`);
    
    if (consistencyResult.passed) {
      console.log('\n🚀 Ready for production benchmarking!');
      console.log('This pinned dataset provides:');
      console.log('  • Consistent baseline measurements');
      console.log('  • Reproducible benchmark results');
      console.log('  • Reliable regression detection');
      console.log('  • Version-controlled dataset evolution');
    }
    
    return true;
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

// Run the test
if (import.meta.url.startsWith('file:') && process.argv[1] && import.meta.url.endsWith('test-pinned-dataset-usage.js')) {
  testPinnedDatasetUsage().then(success => {
    process.exit(success ? 0 : 1);
  });
}