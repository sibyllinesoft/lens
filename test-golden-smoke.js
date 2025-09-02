#!/usr/bin/env node

/**
 * Test if SMOKE_DEFAULT queries exist and can be found in the current golden dataset
 */

import fs from 'fs/promises';
import path from 'path';

async function testGoldenSmoke() {
  console.log('🔍 Testing SMOKE_DEFAULT queries in golden dataset...\n');
  
  try {
    // Read the golden dataset
    const goldenPath = './benchmark-results/golden-dataset.json';
    const goldenContent = await fs.readFile(goldenPath, 'utf-8');
    const goldenData = JSON.parse(goldenContent);
    
    console.log(`📊 Total golden items: ${goldenData.length}`);
    
    // Filter for SMOKE_DEFAULT
    const smokeItems = goldenData.filter(item => 
      item.slice_tags && item.slice_tags.includes('SMOKE_DEFAULT')
    );
    
    console.log(`🧪 SMOKE_DEFAULT items: ${smokeItems.length}`);
    
    // Show first few smoke items
    if (smokeItems.length > 0) {
      console.log('\n📋 Sample SMOKE_DEFAULT queries:');
      for (let i = 0; i < Math.min(5, smokeItems.length); i++) {
        const item = smokeItems[i];
        console.log(`   ${i + 1}. "${item.query}" (${item.language}) -> ${item.expected_results[0]?.file}:${item.expected_results[0]?.line}`);
      }
    } else {
      console.log('❌ No SMOKE_DEFAULT queries found!');
    }
    
    // Check languages
    const languages = [...new Set(smokeItems.map(item => item.language))];
    console.log(`\n🗣️ Languages in SMOKE_DEFAULT: ${languages.join(', ')}`);
    
    // Check file paths in expected results
    const expectedFiles = smokeItems.map(item => item.expected_results[0]?.file).filter(Boolean);
    const uniqueFiles = [...new Set(expectedFiles)];
    
    console.log(`📁 Expected result files: ${uniqueFiles.length} unique files`);
    
    if (uniqueFiles.length > 0) {
      console.log('\n🎯 Sample expected files:');
      for (let i = 0; i < Math.min(5, uniqueFiles.length); i++) {
        console.log(`   ${i + 1}. ${uniqueFiles[i]}`);
      }
      
      // Check if these files exist in indexed content
      console.log('\n🔍 Checking if expected files exist in indexed-content...');
      const indexedDir = './indexed-content/lens-src';
      
      let foundFiles = 0;
      let missingFiles = 0;
      
      for (const expectedFile of uniqueFiles.slice(0, 10)) { // Check first 10
        try {
          const fullPath = path.join(indexedDir, expectedFile);
          await fs.access(fullPath);
          foundFiles++;
        } catch (error) {
          missingFiles++;
          if (missingFiles <= 3) { // Only show first few missing files
            console.log(`   ❌ Missing: ${expectedFile}`);
          }
        }
      }
      
      console.log(`📊 File check: ${foundFiles} found, ${missingFiles} missing (of first 10 checked)`);
    }
    
    // Check repo_sha in golden data vs manifest
    if (smokeItems.length > 0) {
      const goldenSha = smokeItems[0].snapshot_sha;
      console.log(`\n🔖 Golden dataset SHA: ${goldenSha}`);
      
      // Check current manifest
      try {
        const response = await fetch('http://localhost:3000/manifest');
        const manifest = await response.json();
        console.log('📋 Current manifest:', manifest);
        
        const currentSha = manifest.master?.repo_sha;
        if (currentSha) {
          if (currentSha.startsWith(goldenSha.slice(0, 8)) || goldenSha.startsWith(currentSha.slice(0, 8))) {
            console.log('✅ SHA compatibility: MATCH');
          } else {
            console.log('⚠️  SHA compatibility: MISMATCH');
            console.log(`   Golden expects: ${goldenSha}`);
            console.log(`   Server has: ${currentSha}`);
          }
        }
      } catch (error) {
        console.log('⚠️  Could not check server manifest:', error.message);
      }
    }
    
  } catch (error) {
    console.error('❌ Failed to test golden smoke data:', error);
    process.exit(1);
  }
}

testGoldenSmoke();