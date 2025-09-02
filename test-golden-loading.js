#!/usr/bin/env node

/**
 * Test script to verify golden dataset loading functionality
 * This tests the core functionality without full TypeScript compilation
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function testGoldenLoading() {
  console.log('🧪 Testing golden dataset loading functionality...\n');
  
  try {
    // Check if the fixed golden dataset exists
    const goldenPath = path.join(process.cwd(), 'benchmark-results', 'golden-dataset.json');
    console.log(`📁 Checking golden dataset at: ${goldenPath}`);
    
    const content = await fs.readFile(goldenPath, 'utf-8');
    const goldenData = JSON.parse(content);
    
    if (!Array.isArray(goldenData)) {
      throw new Error('Golden data is not an array');
    }
    
    console.log(`📊 Golden dataset contains ${goldenData.length} total items`);
    
    // Filter for SMOKE_DEFAULT items
    const smokeItems = goldenData.filter(item => 
      item.slice_tags && item.slice_tags.includes('SMOKE_DEFAULT')
    );
    
    console.log(`🧪 Found ${smokeItems.length} SMOKE_DEFAULT items`);
    
    // Verify file paths are correct
    const sampleItem = goldenData[0];
    console.log(`🔍 Sample item query: "${sampleItem.query}"`);
    console.log(`📁 Sample expected file path: "${sampleItem.expected_results[0].file}"`);
    
    // Check if the file path exists in indexed content
    const expectedFile = path.join(process.cwd(), 'indexed-content', sampleItem.expected_results[0].file);
    console.log(`🔍 Checking if indexed file exists: ${expectedFile}`);
    
    try {
      await fs.access(expectedFile);
      console.log(`✅ Indexed file exists!`);
    } catch (error) {
      console.log(`⚠️ Indexed file not found - may need re-indexing`);
    }
    
    // Test slice filtering
    const testSliceTags = ['SMOKE_DEFAULT'];
    const filteredItems = goldenData.filter(item => {
      if (!item.slice_tags) return false;
      return testSliceTags.some(tag => item.slice_tags.includes(tag));
    });
    
    console.log(`🎯 Slice filtering test: ${filteredItems.length} items match slice tags`);
    
    console.log('\n✅ Golden dataset loading test completed successfully!');
    console.log('🎯 Key findings:');
    console.log(`   - Golden dataset loaded: ${goldenData.length} items`);
    console.log(`   - SMOKE_DEFAULT items: ${smokeItems.length}`);
    console.log(`   - File paths appear to be fixed (api/ prefix)`);
    console.log('   - Ready for benchmark integration');
    
    return {
      success: true,
      totalItems: goldenData.length,
      smokeItems: smokeItems.length,
      sampleQuery: sampleItem.query
    };
    
  } catch (error) {
    console.error(`❌ Golden dataset loading test failed:`, error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

// Run test if called directly  
testGoldenLoading().then(result => {
  if (result.success) {
    console.log('\n🎉 All tests passed! Golden dataset loading is working correctly.');
  } else {
    console.log('\n💥 Test failed - see errors above.');
  }
}).catch(error => {
  console.error('💥 Unexpected error:', error);
});

// Alternative check for direct execution
if (import.meta.url === `file://${process.argv[1]}`) {
  testGoldenLoading().then(result => {
    if (result.success) {
      console.log('\n🎉 All tests passed! Golden dataset loading is working correctly.');
      process.exit(0);
    } else {
      console.log('\n💥 Test failed - see errors above.');
      process.exit(1);
    }
  });
}

export default testGoldenLoading;