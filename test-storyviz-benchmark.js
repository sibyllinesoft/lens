/**
 * Test storyviz corpus benchmark system
 */

import { promises as fs } from 'fs';
import path from 'path';

class StoryVizBenchmarkTester {
  constructor() {
    this.indexedDir = path.resolve('./indexed-content');
    this.goldenFile = path.resolve('./validation-data/golden-storyviz.json');
  }

  async testCorpusGoldenConsistency() {
    console.log('🔍 Testing corpus-golden consistency...');

    // Load golden data
    const goldenData = JSON.parse(await fs.readFile(this.goldenFile, 'utf-8'));
    const goldenItems = goldenData.golden_items;

    // Get list of indexed files
    const indexedFiles = new Set();
    const files = await fs.readdir(this.indexedDir);
    
    for (const file of files) {
      if (file.endsWith('.json')) continue; // Skip metadata files
      indexedFiles.add(file);
      
      // Also add the original path format
      const originalPath = file.replace(/_/g, '/');
      indexedFiles.add(originalPath);
    }

    console.log(`📁 Found ${indexedFiles.size} indexed files`);

    // Check consistency
    let validItems = 0;
    let invalidItems = 0;
    const inconsistencies = [];

    for (const item of goldenItems) {
      for (const expectedResult of item.expected_results) {
        const filePath = expectedResult.file;
        
        // Skip synthetic/semantic entries
        if (filePath === 'synthetic' || filePath === 'semantic_match') {
          validItems++;
          continue;
        }

        // Check if file exists in corpus
        const exists = indexedFiles.has(filePath) || 
                      indexedFiles.has(path.basename(filePath)) ||
                      Array.from(indexedFiles).some(f => f.includes(path.basename(filePath).replace('.py', '')));

        if (exists) {
          validItems++;
        } else {
          invalidItems++;
          inconsistencies.push({
            query: item.query,
            expected_file: filePath,
            issue: 'file_not_in_corpus'
          });
        }
      }
    }

    const totalItems = validItems + invalidItems;
    const passRate = validItems / Math.max(totalItems, 1);

    console.log(`📊 Consistency Results:`);
    console.log(`  Valid items: ${validItems}`);
    console.log(`  Invalid items: ${invalidItems}`);
    console.log(`  Pass rate: ${(passRate * 100).toFixed(1)}%`);

    if (inconsistencies.length > 0) {
      console.log(`⚠️ Sample inconsistencies:`);
      inconsistencies.slice(0, 5).forEach((inc, i) => {
        console.log(`  ${i + 1}. Query: "${inc.query}" -> Missing: ${inc.expected_file}`);
      });
    }

    return {
      passed: passRate > 0.8, // 80% pass rate threshold
      passRate,
      validItems,
      invalidItems,
      inconsistencies
    };
  }

  async testSampleQueries() {
    console.log('🔍 Testing sample queries...');

    // Load some sample queries
    const goldenData = JSON.parse(await fs.readFile(this.goldenFile, 'utf-8'));
    const sampleQueries = goldenData.golden_items.slice(0, 5);

    for (const item of sampleQueries) {
      console.log(`  Query: "${item.query}" (type: ${item.query_class})`);
      console.log(`    Expected: ${item.expected_results[0].file}:${item.expected_results[0].line}`);
      console.log(`    Language: ${item.language}`);
    }
  }
}

// Main execution
async function main() {
  const tester = new StoryVizBenchmarkTester();

  try {
    console.log('🧪 Running storyviz benchmark tests...');
    
    const consistencyResult = await tester.testCorpusGoldenConsistency();
    
    if (consistencyResult.passed) {
      console.log('✅ Corpus-golden consistency test PASSED');
      
      await tester.testSampleQueries();
      
      console.log('\n✅ All benchmark tests completed successfully!');
      console.log('🚀 Ready to run SMOKE benchmark');
    } else {
      console.log('❌ Corpus-golden consistency test FAILED');
      console.log('  Fix inconsistencies before running benchmarks');
    }
    
  } catch (error) {
    console.error('❌ Benchmark test failed:', error.message);
    process.exit(1);
  }
}

// Run if this is the main module
main();