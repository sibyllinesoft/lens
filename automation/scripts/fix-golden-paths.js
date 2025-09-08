#!/usr/bin/env node
/**
 * Fix golden dataset file paths to match the indexed format
 */

import { promises as fs } from 'fs';
import path from 'path';

async function fixGoldenDataset() {
  const goldenPath = '/media/nathan/Seagate Hub/Projects/lens/benchmark-results/golden-dataset.json';
  
  console.log('🔧 Fixing golden dataset file paths...');
  
  const data = await fs.readFile(goldenPath, 'utf-8');
  const items = JSON.parse(data);
  
  console.log(`Found ${items.length} items to fix`);
  
  for (const item of items) {
    const originalFile = item.expected_results[0].file;
    
    // Convert relative path to match indexed format
    let fixedFile = originalFile;
    
    // If it's in api directory, just use the filename
    if (originalFile.startsWith('api/')) {
      fixedFile = originalFile.replace('api/', '');
    }
    // If it's in other directories, use ../directory/filename format
    else if (originalFile.includes('/')) {
      const parts = originalFile.split('/');
      if (parts.length === 2) {
        fixedFile = `../${parts[0]}/${parts[1]}`;
      }
    }
    
    item.expected_results[0].file = fixedFile;
  }
  
  // Write back the fixed dataset
  await fs.writeFile(goldenPath, JSON.stringify(items, null, 2));
  
  console.log(`✅ Fixed ${items.length} file paths`);
  
  // Show some examples
  const examples = items.slice(0, 5).map(item => ({
    query: item.query,
    file: item.expected_results[0].file
  }));
  
  console.log('\n📄 Example fixed paths:');
  console.log(JSON.stringify(examples, null, 2));
}

fixGoldenDataset().catch(console.error);