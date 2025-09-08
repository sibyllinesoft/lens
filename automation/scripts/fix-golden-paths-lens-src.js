#!/usr/bin/env node

/**
 * Fix golden dataset paths to include lens-src/ prefix
 * The indexed content is in indexed-content/lens-src/ but golden dataset expects direct paths
 */

import { promises as fs } from 'fs';
import path from 'path';

async function fixGoldenPaths() {
  console.log('🔧 Fixing golden dataset paths to include lens-src/ prefix...\n');
  
  const goldenPath = './benchmark-results/golden-dataset.json';
  
  try {
    // Read the current golden dataset
    console.log(`📖 Reading golden dataset from: ${goldenPath}`);
    const content = await fs.readFile(goldenPath, 'utf-8');
    const goldenData = JSON.parse(content);
    
    console.log(`📊 Found ${goldenData.length} golden items`);
    
    // Fix the paths to include lens-src/ prefix
    let pathsFixed = 0;
    let filesAffected = new Set();
    
    for (const item of goldenData) {
      for (const result of item.expected_results) {
        const oldPath = result.file;
        
        // Add lens-src/ prefix if not already present
        if (!oldPath.startsWith('lens-src/')) {
          result.file = 'lens-src/' + oldPath;
          pathsFixed++;
          filesAffected.add(oldPath);
        }
      }
    }
    
    console.log(`🔧 Fixed ${pathsFixed} file paths`);
    console.log(`📁 Affected ${filesAffected.size} unique files`);
    
    // Show a few examples
    console.log('\\n📝 Example path fixes:');
    const examples = Array.from(filesAffected).slice(0, 5);
    for (const example of examples) {
      console.log(`   ${example} → lens-src/${example}`);
    }
    
    // Write the fixed dataset
    const updatedContent = JSON.stringify(goldenData, null, 2);
    await fs.writeFile(goldenPath, updatedContent);
    
    console.log(`\\n✅ Updated golden dataset saved to: ${goldenPath}`);
    console.log('🎯 Golden dataset paths should now match indexed content structure');
    
    return {
      success: true,
      pathsFixed,
      filesAffected: filesAffected.size
    };
    
  } catch (error) {
    console.error('❌ Failed to fix golden dataset paths:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

// Run if called directly
fixGoldenPaths().then(result => {
  if (result.success) {
    console.log('\\n🎉 Golden dataset paths successfully updated!');
    console.log('💡 You can now restart the Lens server and re-run the benchmark.');
  } else {
    console.log('\\n💥 Failed to update golden dataset paths.');
    process.exit(1);
  }
}).catch(error => {
  console.error('💥 Unexpected error:', error);
  process.exit(1);
});

export default fixGoldenPaths;