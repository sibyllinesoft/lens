#!/usr/bin/env node
/**
 * Create golden query data for benchmark testing
 * Generates test queries based on storyviz Python files
 */

import { promises as fs } from 'fs';
import { createHash } from 'crypto';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';

const STORYVIZ_DIR = '/media/nathan/Seagate Hub/Projects/lens/indexed-content/storyviz-sample';
const OUTPUT_DIR = '/media/nathan/Seagate Hub/Projects/lens/benchmark-results';
const REPO_SHA = 'a1b2c3d4e5f6789012345678901234567890abcd'; // Mock SHA for storyviz

async function createGoldenDataset() {
  console.log('🏗️  Creating golden dataset from storyviz Python files...\n');

  // Read all Python files
  const files = await fs.readdir(STORYVIZ_DIR);
  const pythonFiles = files.filter(f => f.endsWith('.py'));
  
  console.log(`Found ${pythonFiles.length} Python files:`, pythonFiles);

  const goldenItems = [];

  for (const file of pythonFiles) {
    const filePath = path.join(STORYVIZ_DIR, file);
    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.split('\n');

    // Extract functions, classes, and imports for queries
    const extractedItems = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Find function definitions
      const funcMatch = line.match(/^def\s+(\w+)\s*\(/);
      if (funcMatch) {
        extractedItems.push({
          type: 'function',
          name: funcMatch[1],
          line: i + 1,
          col: line.indexOf('def')
        });
      }
      
      // Find class definitions
      const classMatch = line.match(/^class\s+(\w+)[\s\(:]?/);
      if (classMatch) {
        extractedItems.push({
          type: 'class',
          name: classMatch[1],
          line: i + 1,
          col: line.indexOf('class')
        });
      }
      
      // Find import statements
      const importMatch = line.match(/^(?:from\s+\S+\s+)?import\s+(.+)/);
      if (importMatch) {
        const imports = importMatch[1].split(',').map(i => i.trim().split(' as ')[0]);
        for (const imp of imports) {
          extractedItems.push({
            type: 'import',
            name: imp,
            line: i + 1,
            col: line.indexOf(imp)
          });
        }
      }
    }

    // Create golden data items for each extracted element
    for (const item of extractedItems) {
      const goldenItem = {
        id: uuidv4(),
        query: item.name,
        query_class: item.type === 'function' ? 'identifier' : 
                    item.type === 'class' ? 'identifier' : 'identifier',
        language: 'py',
        source: 'synthetics',
        snapshot_sha: REPO_SHA,
        slice_tags: ['SMOKE_DEFAULT', 'ALL'],
        expected_results: [{
          file: `storyviz-sample/${file}`,
          line: item.line,
          col: item.col,
          relevance_score: 1.0,
          match_type: item.type === 'import' ? 'exact' : 'symbol',
          why: `${item.type} definition`
        }]
      };
      
      goldenItems.push(goldenItem);
    }
  }

  console.log(`\n✅ Generated ${goldenItems.length} golden data items`);

  // Save golden dataset
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  const outputPath = path.join(OUTPUT_DIR, 'golden-dataset.json');
  await fs.writeFile(outputPath, JSON.stringify(goldenItems, null, 2));
  
  console.log(`📁 Saved golden dataset to: ${outputPath}`);
  
  // Create a summary
  const summary = {
    total_items: goldenItems.length,
    languages: ['py'],
    query_classes: [...new Set(goldenItems.map(item => item.query_class))],
    slice_tags: ['SMOKE_DEFAULT', 'ALL'],
    files_covered: pythonFiles.length,
    sample_queries: goldenItems.slice(0, 5).map(item => ({
      query: item.query,
      expected_file: item.expected_results[0].file,
      line: item.expected_results[0].line
    }))
  };

  console.log('\n📊 Dataset Summary:');
  console.log(JSON.stringify(summary, null, 2));
  
  return {
    goldenItems,
    summary,
    outputPath
  };
}

async function main() {
  try {
    const result = await createGoldenDataset();
    console.log('\n🎯 Golden dataset created successfully!');
    console.log(`Ready for benchmark testing with ${result.goldenItems.length} queries.`);
  } catch (error) {
    console.error('❌ Failed to create golden dataset:', error.message);
    process.exit(1);
  }
}

// Use pathToFileURL to handle path encoding correctly
import { pathToFileURL } from 'url';

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}