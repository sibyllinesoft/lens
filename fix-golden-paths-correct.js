#!/usr/bin/env node

/**
 * Fix the golden dataset file paths to match the actual indexed content structure
 */

import fs from 'fs/promises';
import path from 'path';

async function fixGoldenPaths() {
  console.log('🔧 Fixing golden dataset file paths...\n');
  
  try {
    // Read the golden dataset
    const goldenPath = './benchmark-results/golden-dataset.json';
    const goldenContent = await fs.readFile(goldenPath, 'utf-8');
    const goldenData = JSON.parse(goldenContent);
    
    console.log(`📊 Processing ${goldenData.length} golden items...`);
    
    // Get actual indexed file structure
    const indexedDir = './indexed-content/lens-src';
    const actualFiles = [];
    
    async function collectFiles(dir, relativePath = '') {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const currentRelative = path.join(relativePath, entry.name);
        
        if (entry.isDirectory()) {
          await collectFiles(fullPath, currentRelative);
        } else if (entry.isFile()) {
          const ext = path.extname(entry.name).toLowerCase();
          if (['.ts', '.js'].includes(ext)) {
            actualFiles.push(currentRelative);
          }
        }
      }
    }
    
    await collectFiles(indexedDir);
    console.log(`📁 Found ${actualFiles.length} actual indexed files`);
    
    // Create filename to path mapping
    const fileNameToPath = {};
    for (const filePath of actualFiles) {
      const fileName = path.basename(filePath);
      if (!fileNameToPath[fileName]) {
        fileNameToPath[fileName] = [];
      }
      fileNameToPath[fileName].push(filePath);
    }
    
    // Fix golden dataset paths
    let fixedCount = 0;
    let notFoundCount = 0;
    const notFoundFiles = new Set();
    
    for (const item of goldenData) {
      if (item.expected_results && item.expected_results.length > 0) {
        const expectedResult = item.expected_results[0];
        const originalPath = expectedResult.file;
        
        // Try different strategies to find the correct path
        let newPath = null;
        
        // Strategy 1: Direct match (file already correct)
        if (actualFiles.includes(originalPath)) {
          newPath = originalPath;
        }
        // Strategy 2: Just filename (most common case)
        else {
          const fileName = path.basename(originalPath);
          if (fileNameToPath[fileName] && fileNameToPath[fileName].length === 1) {
            newPath = fileNameToPath[fileName][0];
          }
          // Strategy 3: Multiple matches - try to find the best one
          else if (fileNameToPath[fileName] && fileNameToPath[fileName].length > 1) {
            // If original had '../benchmark/', prefer 'benchmark/' path
            if (originalPath.includes('../benchmark/')) {
              const benchmarkPath = fileNameToPath[fileName].find(p => p.startsWith('benchmark/'));
              if (benchmarkPath) {
                newPath = benchmarkPath;
              }
            }
            // Otherwise take the first match
            if (!newPath) {
              newPath = fileNameToPath[fileName][0];
            }
          }
        }
        
        if (newPath && newPath !== originalPath) {
          expectedResult.file = newPath;
          fixedCount++;
        } else if (!newPath) {
          notFoundCount++;
          notFoundFiles.add(originalPath);
        }
      }
    }
    
    console.log(`✅ Fixed ${fixedCount} file paths`);
    console.log(`❌ Could not fix ${notFoundCount} paths`);
    
    if (notFoundFiles.size > 0) {
      console.log('\n🔍 Files not found in indexed content:');
      for (const file of Array.from(notFoundFiles).slice(0, 5)) {
        console.log(`   - ${file}`);
      }
      if (notFoundFiles.size > 5) {
        console.log(`   ... and ${notFoundFiles.size - 5} more`);
      }
    }
    
    // Save the fixed dataset
    const backupPath = goldenPath + '.backup';
    await fs.copyFile(goldenPath, backupPath);
    await fs.writeFile(goldenPath, JSON.stringify(goldenData, null, 2));
    
    console.log(`\n💾 Saved fixed dataset to: ${goldenPath}`);
    console.log(`📋 Backup created at: ${backupPath}`);
    
    // Verify fix by testing a few queries
    console.log('\n🧪 Verifying fixes...');
    const sampleItems = goldenData.slice(0, 5);
    
    for (const item of sampleItems) {
      if (item.expected_results && item.expected_results.length > 0) {
        const expectedFile = item.expected_results[0].file;
        const fullPath = path.join(indexedDir, expectedFile);
        
        try {
          await fs.access(fullPath);
          console.log(`   ✅ "${item.query}" -> ${expectedFile} (exists)`);
        } catch (error) {
          console.log(`   ❌ "${item.query}" -> ${expectedFile} (missing)`);
        }
      }
    }
    
  } catch (error) {
    console.error('❌ Failed to fix golden paths:', error);
    process.exit(1);
  }
}

fixGoldenPaths();