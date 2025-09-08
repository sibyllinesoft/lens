#!/usr/bin/env node
/**
 * Create golden query data for benchmark testing
 * Generates test queries based on lens TypeScript/JavaScript files
 */

import { promises as fs } from 'fs';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';

const LENS_SRC_DIR = '/media/nathan/Seagate Hub/Projects/lens/indexed-content/lens-src';
const OUTPUT_DIR = '/media/nathan/Seagate Hub/Projects/lens/benchmark-results';
const REPO_SHA = '8a9f5a1'; // From the existing manifest

async function createLensGoldenDataset() {
  console.log('🏗️  Creating golden dataset from lens TypeScript files...\n');

  // Read all TypeScript and JavaScript files
  const allFiles = [];
  
  async function walkDir(dir) {
    const files = await fs.readdir(dir, { withFileTypes: true });
    for (const file of files) {
      const filePath = path.join(dir, file.name);
      if (file.isDirectory()) {
        await walkDir(filePath);
      } else if (file.name.endsWith('.ts') || file.name.endsWith('.js')) {
        allFiles.push(filePath);
      }
    }
  }
  
  await walkDir(LENS_SRC_DIR);
  
  console.log(`Found ${allFiles.length} TypeScript/JavaScript files`);

  const goldenItems = [];

  for (const filePath of allFiles) {
    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.split('\n');
    const relativePath = path.relative(LENS_SRC_DIR, filePath);

    // Extract functions, classes, interfaces, and imports for queries
    const extractedItems = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Find function definitions (both function and arrow functions)
      const funcMatch = line.match(/^(?:export\s+)?(?:async\s+)?function\s+(\w+)/) || 
                       line.match(/^(?:export\s+)?const\s+(\w+)\s*=.*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>/);
      if (funcMatch) {
        extractedItems.push({
          type: 'function',
          name: funcMatch[1],
          line: i + 1,
          col: line.indexOf(funcMatch[1])
        });
      }
      
      // Find class definitions
      const classMatch = line.match(/^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/);
      if (classMatch) {
        extractedItems.push({
          type: 'class',
          name: classMatch[1],
          line: i + 1,
          col: line.indexOf(classMatch[1])
        });
      }
      
      // Find interface definitions
      const interfaceMatch = line.match(/^(?:export\s+)?interface\s+(\w+)/);
      if (interfaceMatch) {
        extractedItems.push({
          type: 'interface',
          name: interfaceMatch[1],
          line: i + 1,
          col: line.indexOf(interfaceMatch[1])
        });
      }
      
      // Find type definitions
      const typeMatch = line.match(/^(?:export\s+)?type\s+(\w+)/);
      if (typeMatch) {
        extractedItems.push({
          type: 'type',
          name: typeMatch[1],
          line: i + 1,
          col: line.indexOf(typeMatch[1])
        });
      }
      
      // Find import statements
      const importMatch = line.match(/^import\s*{[^}]*?(\w+)[^}]*?}\s*from/) || 
                         line.match(/^import\s+(\w+)\s+from/);
      if (importMatch) {
        // Extract multiple imports from destructured imports
        if (line.includes('{')) {
          const importsStr = line.match(/{\s*([^}]+)\s*}/)?.[1];
          if (importsStr) {
            const imports = importsStr.split(',').map(imp => imp.trim().split(' as ')[0]);
            for (const imp of imports) {
              extractedItems.push({
                type: 'import',
                name: imp,
                line: i + 1,
                col: line.indexOf(imp)
              });
            }
          }
        } else {
          extractedItems.push({
            type: 'import',
            name: importMatch[1],
            line: i + 1,
            col: line.indexOf(importMatch[1])
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
                    item.type === 'class' ? 'identifier' :
                    item.type === 'interface' ? 'identifier' :
                    item.type === 'type' ? 'identifier' : 'identifier',
        language: 'ts',
        source: 'synthetics',
        snapshot_sha: REPO_SHA,
        slice_tags: ['SMOKE_DEFAULT', 'ALL'],
        expected_results: [{
          file: relativePath,
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
    languages: ['ts'],
    query_classes: [...new Set(goldenItems.map(item => item.query_class))],
    slice_tags: ['SMOKE_DEFAULT', 'ALL'],
    files_covered: allFiles.length,
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
    const result = await createLensGoldenDataset();
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