#!/usr/bin/env node

/**
 * Add golden dataset loading capability to the GroundTruthBuilder
 * This fixes the issue where benchmarks return 0 total_queries
 */

import fs from 'fs/promises';
import path from 'path';

async function fixGroundTruthBuilder() {
  console.log('🔧 Adding golden dataset loading to GroundTruthBuilder...\n');
  
  const builderPath = './src/benchmark/ground-truth-builder.ts';
  
  try {
    // Read the current file
    const content = await fs.readFile(builderPath, 'utf-8');
    
    // Check if loadGoldenDataset method already exists
    if (content.includes('loadGoldenDataset')) {
      console.log('✅ loadGoldenDataset method already exists');
      return;
    }
    
    // Find the end of the class (before the final closing brace)
    const classEndIndex = content.lastIndexOf('}');
    if (classEndIndex === -1) {
      throw new Error('Could not find class end');
    }
    
    // The method to add
    const loadMethod = `
  /**
   * Load golden dataset from JSON file
   * This method loads the golden dataset that was created by create-lens-golden-data.js
   */
  async loadGoldenDataset(goldenPath?: string): Promise<void> {
    const defaultPaths = [
      path.join(process.cwd(), 'benchmark-results', 'golden-dataset.json'),
      path.join(process.cwd(), 'benchmark-results', 'smith-golden-dataset.json'),
      path.join(process.cwd(), 'sample-storyviz', 'golden-dataset.jsonl')
    ];
    
    const pathsToTry = goldenPath ? [goldenPath, ...defaultPaths] : defaultPaths;
    
    for (const testPath of pathsToTry) {
      try {
        console.log(\`🔍 Trying to load golden dataset from: \${testPath}\`);
        
        let goldenData: any[];
        
        if (testPath.endsWith('.jsonl')) {
          // Handle JSONL format
          const content = await fs.readFile(testPath, 'utf-8');
          goldenData = content.trim().split('\\n').map(line => JSON.parse(line));
        } else {
          // Handle JSON format
          const content = await fs.readFile(testPath, 'utf-8');
          goldenData = JSON.parse(content);
        }
        
        if (!Array.isArray(goldenData)) {
          console.warn(\`⚠️ Golden data in \${testPath} is not an array, skipping\`);
          continue;
        }
        
        // Filter for SMOKE_DEFAULT slice if requested
        const smokeItems = goldenData.filter((item: any) => 
          item.slice_tags && item.slice_tags.includes('SMOKE_DEFAULT')
        );
        
        console.log(\`📊 Loaded \${goldenData.length} total golden items from \${testPath}\`);
        console.log(\`🧪 Found \${smokeItems.length} SMOKE_DEFAULT items\`);
        
        // Set the golden items
        this.goldenItems = goldenData;
        
        console.log(\`✅ Successfully loaded golden dataset from \${testPath}\`);
        return;
        
      } catch (error) {
        console.log(\`❌ Could not load from \${testPath}: \${error.message}\`);
        continue;
      }
    }
    
    throw new Error('Could not load golden dataset from any of the attempted paths');
  }

  /**
   * Filter golden items by slice tags
   */
  filterGoldenItemsBySlice(sliceTags: string | string[]): GoldenDataItem[] {
    const targetTags = Array.isArray(sliceTags) ? sliceTags : [sliceTags];
    
    return this.goldenItems.filter(item => {
      if (!item.slice_tags) return false;
      return targetTags.some(tag => item.slice_tags.includes(tag));
    });
  }
`;

    // Insert the method before the final closing brace
    const beforeEnd = content.slice(0, classEndIndex);
    const afterEnd = content.slice(classEndIndex);
    
    const newContent = beforeEnd + loadMethod + afterEnd;
    
    // Write the updated file
    await fs.writeFile(builderPath, newContent);
    
    console.log('✅ Added loadGoldenDataset and filterGoldenItemsBySlice methods to GroundTruthBuilder');
    
  } catch (error) {
    console.error('❌ Failed to add golden dataset loading:', error);
    throw error;
  }
}

async function fixBenchmarkEndpoints() {
  console.log('🔧 Updating benchmark endpoints to load golden data...\n');
  
  const endpointsPath = './src/api/benchmark-endpoints.ts';
  
  try {
    let content = await fs.readFile(endpointsPath, 'utf-8');
    
    // Check if the fix is already applied
    if (content.includes('loadGoldenDataset')) {
      console.log('✅ Benchmark endpoints already load golden dataset');
      return;
    }
    
    // Find where groundTruthBuilder is created and add the loading call
    const builderCreation = 'const groundTruthBuilder = new GroundTruthBuilder(workingDir, outputDir);';
    
    if (!content.includes(builderCreation)) {
      console.log('⚠️ Could not find groundTruthBuilder creation line, skipping endpoints fix');
      return;
    }
    
    const replacement = `const groundTruthBuilder = new GroundTruthBuilder(workingDir, outputDir);
  
  // Load the golden dataset for benchmarking
  try {
    await groundTruthBuilder.loadGoldenDataset();
    console.log('📊 Loaded golden dataset for benchmarking');
  } catch (error) {
    console.error('❌ Failed to load golden dataset:', error);
    // Continue without golden data - benchmarks will return 0 queries
  }`;
    
    content = content.replace(builderCreation, replacement);
    
    await fs.writeFile(endpointsPath, content);
    
    console.log('✅ Updated benchmark endpoints to load golden dataset');
    
  } catch (error) {
    console.error('❌ Failed to update benchmark endpoints:', error);
    throw error;
  }
}

async function main() {
  try {
    console.log('🚀 Fixing benchmark golden dataset loading issue...\n');
    
    await fixGroundTruthBuilder();
    console.log();
    
    await fixBenchmarkEndpoints();
    console.log();
    
    console.log('✅ Fix completed! The benchmark system should now load golden datasets.');
    console.log('');
    console.log('💡 Next steps:');
    console.log('   1. Restart the Lens server: npm restart OR node dist/server.js');
    console.log('   2. Run the smoke benchmark: node run-smoke-benchmark.js');
    console.log('   3. Verify that total_queries > 0 in the results');
    console.log('');
    console.log('🎯 The system will now automatically load golden datasets from:');
    console.log('   - ./benchmark-results/golden-dataset.json (current lens dataset)');
    console.log('   - ./benchmark-results/smith-golden-dataset.json (if created)');
    console.log('   - ./sample-storyviz/golden-dataset.jsonl (legacy format)');
    
  } catch (error) {
    console.error('❌ Fix failed:', error);
    process.exit(1);
  }
}

main();