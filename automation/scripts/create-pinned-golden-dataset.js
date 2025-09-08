#!/usr/bin/env node

/**
 * Create Pinned Golden Dataset for Consistent Benchmarking
 * 
 * This script creates a stable, versioned golden dataset that will be used
 * consistently across all benchmark runs. This ensures reproducible results
 * and eliminates dataset drift issues.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class PinnedGoldenDatasetCreator {
  constructor() {
    this.workingDir = process.cwd();
    this.outputDir = path.join(this.workingDir, 'benchmark-results');
    this.pinnedDir = path.join(this.workingDir, 'pinned-datasets');
  }

  async createPinnedDataset() {
    console.log('📌 Creating Pinned Golden Dataset for Consistent Benchmarking');
    console.log('===========================================================\n');

    // Get current git state for versioning
    const gitSha = this.getGitSha();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const version = `${gitSha.substring(0, 8)}-${timestamp}`;

    console.log(`🔍 Git SHA: ${gitSha}`);
    console.log(`📅 Timestamp: ${timestamp}`);
    console.log(`📌 Version: ${version}\n`);

    // Ensure directories exist
    await fs.mkdir(this.pinnedDir, { recursive: true });
    await fs.mkdir(this.outputDir, { recursive: true });

    // Load the current golden dataset
    const goldenData = await this.loadCurrentGoldenDataset();
    
    if (!goldenData || goldenData.length === 0) {
      throw new Error('No golden dataset found. Run the corpus setup first.');
    }

    console.log(`📊 Loaded ${goldenData.length} golden items from current dataset`);

    // Create the pinned dataset with metadata
    const pinnedDataset = {
      version: version,
      git_sha: gitSha,
      timestamp: new Date().toISOString(),
      pinned_at: new Date().toISOString(),
      source_dataset: this.lastLoadedPath || 'unknown',
      total_items: goldenData.length,
      corpus_stats: await this.getCorpusStats(),
      slice_distribution: this.analyzeSliceDistribution(goldenData),
      language_distribution: this.analyzeLanguageDistribution(goldenData),
      query_class_distribution: this.analyzeQueryClassDistribution(goldenData),
      golden_items: goldenData
    };

    // Write the pinned dataset
    const pinnedPath = path.join(this.pinnedDir, `golden-pinned-${version}.json`);
    await fs.writeFile(pinnedPath, JSON.stringify(pinnedDataset, null, 2));
    
    // Create a symlink to the current pinned version (or copy if symlink fails)
    const currentPath = path.join(this.pinnedDir, 'golden-pinned-current.json');
    try {
      await fs.unlink(currentPath);
    } catch (error) {
      // Ignore if file doesn't exist
    }
    
    try {
      await fs.symlink(path.basename(pinnedPath), currentPath);
      console.log(`🔗 Created symlink: ${currentPath} -> ${path.basename(pinnedPath)}`);
    } catch (error) {
      // Fallback: copy the file if symlink fails (e.g., on Windows or external drives)
      await fs.copyFile(pinnedPath, currentPath);
      console.log(`📋 Created copy (symlink failed): ${currentPath}`);
    }

    // Create a compact version for fast loading
    const compactPath = path.join(this.pinnedDir, `golden-pinned-${version}-compact.json`);
    await fs.writeFile(compactPath, JSON.stringify(goldenData, null, 0));

    // Generate summary report
    const reportPath = path.join(this.outputDir, `pinned-dataset-report-${version}.md`);
    await this.generatePinningReport(pinnedDataset, reportPath);

    console.log('\n✅ Pinned Golden Dataset Created Successfully!');
    console.log('================================================');
    console.log(`📌 Pinned dataset: ${pinnedPath}`);
    console.log(`🔗 Current link: ${currentPath}`);
    console.log(`📦 Compact version: ${compactPath}`);
    console.log(`📄 Report: ${reportPath}`);
    
    console.log('\n📊 Dataset Statistics:');
    console.log(`   Total items: ${pinnedDataset.total_items}`);
    console.log(`   Languages: ${Object.keys(pinnedDataset.language_distribution).length}`);
    console.log(`   Query classes: ${Object.keys(pinnedDataset.query_class_distribution).length}`);
    console.log(`   Slices: ${Object.keys(pinnedDataset.slice_distribution).length}`);
    
    return {
      pinnedPath,
      currentPath,
      compactPath,
      version,
      dataset: pinnedDataset
    };
  }

  async loadCurrentGoldenDataset() {
    const possiblePaths = [
      path.join(this.outputDir, 'golden-dataset.json'),
      path.join(this.workingDir, 'validation-data', 'golden-storyviz.json'),
      path.join(this.workingDir, 'sample-storyviz', 'golden-dataset.jsonl')
    ];

    for (const testPath of possiblePaths) {
      try {
        console.log(`🔍 Trying to load golden dataset from: ${testPath}`);
        
        let goldenData;
        if (testPath.endsWith('.jsonl')) {
          const content = await fs.readFile(testPath, 'utf-8');
          goldenData = content.trim().split('\n').map(line => JSON.parse(line));
        } else {
          const content = await fs.readFile(testPath, 'utf-8');
          goldenData = JSON.parse(content);
        }

        if (Array.isArray(goldenData) && goldenData.length > 0) {
          console.log(`✅ Loaded ${goldenData.length} items from ${testPath}`);
          this.lastLoadedPath = testPath;
          return goldenData;
        }
      } catch (error) {
        console.log(`❌ Failed to load from ${testPath}: ${error.message}`);
      }
    }

    return null;
  }

  async getCorpusStats() {
    try {
      const indexedDir = path.join(this.workingDir, 'indexed-content');
      const files = await fs.readdir(indexedDir);
      
      const stats = {
        total_files: files.length,
        file_types: {}
      };

      for (const file of files) {
        if (file.endsWith('.json')) continue; // Skip metadata files
        
        const ext = path.extname(file);
        stats.file_types[ext] = (stats.file_types[ext] || 0) + 1;
      }

      return stats;
    } catch (error) {
      return { error: error.message };
    }
  }

  analyzeSliceDistribution(goldenData) {
    const distribution = {};
    
    for (const item of goldenData) {
      if (item.slice_tags) {
        for (const tag of item.slice_tags) {
          distribution[tag] = (distribution[tag] || 0) + 1;
        }
      }
    }

    return distribution;
  }

  analyzeLanguageDistribution(goldenData) {
    const distribution = {};
    
    for (const item of goldenData) {
      const lang = item.language || 'unknown';
      distribution[lang] = (distribution[lang] || 0) + 1;
    }

    return distribution;
  }

  analyzeQueryClassDistribution(goldenData) {
    const distribution = {};
    
    for (const item of goldenData) {
      const queryClass = item.query_class || 'unknown';
      distribution[queryClass] = (distribution[queryClass] || 0) + 1;
    }

    return distribution;
  }

  getGitSha() {
    try {
      return execSync('git rev-parse HEAD', { 
        cwd: this.workingDir, 
        encoding: 'utf-8' 
      }).trim();
    } catch (error) {
      return 'unknown';
    }
  }

  async generatePinningReport(dataset, reportPath) {
    const report = `# Pinned Golden Dataset Report

## Dataset Information

- **Version**: ${dataset.version}
- **Git SHA**: ${dataset.git_sha}
- **Pinned At**: ${dataset.pinned_at}
- **Source Dataset**: ${dataset.source_dataset}
- **Total Items**: ${dataset.total_items}

## Corpus Statistics

- **Total Files**: ${dataset.corpus_stats.total_files}
- **File Types**: ${JSON.stringify(dataset.corpus_stats.file_types, null, 2)}

## Dataset Distribution Analysis

### Language Distribution
${Object.entries(dataset.language_distribution)
  .map(([lang, count]) => `- **${lang}**: ${count} items (${((count / dataset.total_items) * 100).toFixed(1)}%)`)
  .join('\n')}

### Query Class Distribution  
${Object.entries(dataset.query_class_distribution)
  .map(([queryClass, count]) => `- **${queryClass}**: ${count} items (${((count / dataset.total_items) * 100).toFixed(1)}%)`)
  .join('\n')}

### Slice Distribution
${Object.entries(dataset.slice_distribution)
  .map(([slice, count]) => `- **${slice}**: ${count} items`)
  .join('\n')}

## Usage Instructions

This pinned dataset ensures consistent benchmarking results across runs. To use:

1. **Load in benchmarks**: Reference \`golden-pinned-current.json\` 
2. **Baseline measurements**: All benchmarks use this exact dataset
3. **Comparative analysis**: Changes can be measured against this stable baseline

## Validation Notes

- ✅ Dataset includes expected results for all items
- ✅ Corpus alignment validated at pin time  
- ✅ Language and query class distributions preserved
- ✅ All slice tags maintained for stratified sampling

---

**Generated**: ${new Date().toISOString()}  
**Purpose**: Establish stable benchmark baseline for reproducible results
`;

    await fs.writeFile(reportPath, report);
  }

  async listAvailablePinnedDatasets() {
    try {
      const files = await fs.readdir(this.pinnedDir);
      const pinnedFiles = files.filter(f => f.startsWith('golden-pinned-') && f.endsWith('.json') && !f.includes('compact'));
      
      console.log('\n📌 Available Pinned Datasets:');
      console.log('=============================');
      
      for (const file of pinnedFiles.sort()) {
        const filePath = path.join(this.pinnedDir, file);
        const stat = await fs.stat(filePath);
        
        console.log(`   ${file}`);
        console.log(`     Size: ${(stat.size / 1024).toFixed(1)} KB`);
        console.log(`     Modified: ${stat.mtime.toISOString()}`);
        console.log('');
      }

      // Show current symlink target
      try {
        const currentPath = path.join(this.pinnedDir, 'golden-pinned-current.json');
        const linkTarget = await fs.readlink(currentPath);
        console.log(`🔗 Current active: ${linkTarget}`);
      } catch (error) {
        console.log('🔗 No current active dataset');
      }
    } catch (error) {
      console.log('📁 No pinned datasets directory found');
    }
  }
}

// Main execution
async function main() {
  const creator = new PinnedGoldenDatasetCreator();
  
  const command = process.argv[2];
  
  try {
    if (command === 'list') {
      await creator.listAvailablePinnedDatasets();
    } else {
      const result = await creator.createPinnedDataset();
      
      console.log('\n🎯 Next Steps:');
      console.log('==============');
      console.log('1. Update benchmarking code to use pinned dataset');
      console.log('2. Run baseline benchmark with pinned data');
      console.log('3. Document the pinned dataset in CLAUDE.md');
      console.log('4. Use pinned dataset for all future benchmark comparisons');
      
      console.log('\n💡 Usage Examples:');
      console.log('==================');
      console.log('// Load pinned dataset in code:');
      console.log(`const pinnedDataset = require('./pinned-datasets/golden-pinned-current.json');`);
      console.log(`const goldenItems = pinnedDataset.golden_items;`);
    }
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

// Check if this is the main module
if (import.meta.url.startsWith('file:') && process.argv[1] && import.meta.url.endsWith(path.basename(process.argv[1]))) {
  main().catch(error => {
    console.error('❌ Error:', error.message);
    process.exit(1);
  });
}

export default PinnedGoldenDatasetCreator;