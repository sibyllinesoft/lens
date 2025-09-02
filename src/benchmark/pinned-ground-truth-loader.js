/**
 * Pinned Ground Truth Loader for Consistent Benchmarking
 * 
 * This module loads and manages pinned golden datasets to ensure
 * consistent benchmark baselines across runs.
 */

import { promises as fs } from 'fs';
import path from 'path';
import { createHash } from 'crypto';

export class PinnedGroundTruthLoader {
  constructor(workingDir = process.cwd()) {
    this.pinnedDataset = null;
    this.workingDir = workingDir;
    this.pinnedDir = path.join(workingDir, 'pinned-datasets');
  }

  /**
   * Load the current pinned golden dataset
   */
  async loadPinnedDataset(version) {
    let datasetPath;

    if (version) {
      // Load specific version
      datasetPath = path.join(this.pinnedDir, `golden-pinned-${version}.json`);
    } else {
      // Load current active version
      datasetPath = path.join(this.pinnedDir, 'golden-pinned-current.json');
    }

    console.log(`📌 Loading pinned dataset from: ${datasetPath}`);

    try {
      const content = await fs.readFile(datasetPath, 'utf-8');
      this.pinnedDataset = JSON.parse(content);
      
      console.log(`✅ Loaded pinned dataset version: ${this.pinnedDataset.version}`);
      console.log(`   Total items: ${this.pinnedDataset.total_items}`);
      console.log(`   Pinned at: ${this.pinnedDataset.pinned_at}`);
      console.log(`   Git SHA: ${this.pinnedDataset.git_sha}`);

      return this.pinnedDataset;
    } catch (error) {
      throw new Error(`Failed to load pinned dataset from ${datasetPath}: ${error.message}`);
    }
  }

  /**
   * Get all golden items from the pinned dataset
   */
  getCurrentGoldenItems() {
    if (!this.pinnedDataset) {
      throw new Error('No pinned dataset loaded. Call loadPinnedDataset() first.');
    }

    return this.pinnedDataset.golden_items;
  }

  /**
   * Get golden items filtered by slice tags
   */
  filterGoldenItemsBySlice(sliceTags) {
    const allItems = this.getCurrentGoldenItems();
    const targetTags = Array.isArray(sliceTags) ? sliceTags : [sliceTags];
    
    return allItems.filter(item => {
      if (!item.slice_tags) return false;
      return targetTags.some(tag => item.slice_tags.includes(tag));
    });
  }

  /**
   * Get SMOKE_DEFAULT dataset (for smoke tests)
   */
  getSmokeDataset() {
    return this.filterGoldenItemsBySlice('SMOKE_DEFAULT');
  }

  /**
   * Get all available dataset slices
   */
  getAvailableSlices() {
    if (!this.pinnedDataset) {
      throw new Error('No pinned dataset loaded.');
    }

    return Object.keys(this.pinnedDataset.slice_distribution);
  }

  /**
   * Get dataset statistics
   */
  getDatasetStats() {
    if (!this.pinnedDataset) {
      throw new Error('No pinned dataset loaded.');
    }

    return {
      version: this.pinnedDataset.version,
      total_items: this.pinnedDataset.total_items,
      languages: this.pinnedDataset.language_distribution,
      query_classes: this.pinnedDataset.query_class_distribution,
      slices: this.pinnedDataset.slice_distribution
    };
  }

  /**
   * Generate configuration fingerprint using pinned dataset
   */
  generateConfigFingerprint(config, seedSet) {
    if (!this.pinnedDataset) {
      throw new Error('No pinned dataset loaded.');
    }

    const configHash = createHash('sha256')
      .update(JSON.stringify(config, null, 0))
      .digest('hex');
      
    // Include pinned dataset version in code hash for reproducibility
    const codeHash = createHash('sha256')
      .update(process.version + this.pinnedDataset.version)
      .digest('hex');

    const snapshotShas = {
      'pinned': this.pinnedDataset.git_sha
    };

    return {
      code_hash: codeHash,
      config_hash: configHash,
      snapshot_shas: snapshotShas,
      shard_layout: {
        pinned_dataset_version: this.pinnedDataset.version,
        total_items: this.pinnedDataset.total_items
      },
      timestamp: new Date().toISOString(),
      seed_set: seedSet
    };
  }

  /**
   * Validate that the pinned dataset is consistent with current corpus
   */
  async validatePinnedDatasetConsistency() {
    const goldenItems = this.getCurrentGoldenItems();
    const inconsistencies = [];
    let validItems = 0;
    
    // Get list of indexed files from the corpus (recursively)
    const indexedFiles = new Set();
    try {
      const indexedDir = path.join(this.workingDir, 'indexed-content');
      await this.walkDirectory(indexedDir, indexedFiles, indexedDir);
    } catch (error) {
      console.warn('Could not read indexed-content directory:', error);
    }
    
    // Check each golden item
    for (const item of goldenItems) {
      for (const expectedResult of item.expected_results) {
        const filePath = expectedResult.file;
        
        // Try different path variations to match against indexed files
        const pathVariations = [
          filePath,  // Exact path
          path.basename(filePath),  // Just filename
          filePath.replace(/\//g, '_'),  // Flattened path
          filePath.replace(/lens-src\//g, 'src/'),  // lens-src -> src
          filePath.replace(/^src\//, 'lens-src/'),  // src -> lens-src
        ];
        
        const exists = pathVariations.some(variation => indexedFiles.has(variation));
        
        if (!exists) {
          inconsistencies.push({
            golden_item_id: item.id,
            query: item.query,
            expected_file: filePath,
            line: expectedResult.line,
            col: expectedResult.col,
            issue: 'file_not_in_corpus',
            corpus_size: indexedFiles.size,
            tried_variations: pathVariations
          });
        } else {
          validItems++;
        }
      }
    }
    
    const totalExpected = goldenItems.reduce((sum, item) => sum + item.expected_results.length, 0);
    const passRate = validItems / Math.max(totalExpected, 1);
    const passed = inconsistencies.length === 0;
    
    const report = {
      pinned_dataset_version: this.pinnedDataset?.version || 'unknown',
      total_golden_items: goldenItems.length,
      total_expected_results: totalExpected,
      valid_results: validItems,
      inconsistent_results: inconsistencies.length,
      pass_rate: passRate,
      corpus_file_count: indexedFiles.size,
      inconsistencies
    };
    
    console.log(`📊 Pinned dataset consistency check:`);
    console.log(`   Dataset version: ${this.pinnedDataset?.version}`);
    console.log(`   Pass rate: ${(passRate * 100).toFixed(1)}% (${validItems}/${totalExpected})`);
    console.log(`   Status: ${passed ? '✅ PASSED' : `❌ FAILED (${inconsistencies.length} inconsistencies)`}`);
    
    return { passed, report };
  }

  /**
   * List all available pinned dataset versions
   */
  async listAvailablePinnedDatasets() {
    try {
      const files = await fs.readdir(this.pinnedDir);
      const pinnedFiles = files
        .filter(f => f.startsWith('golden-pinned-') && f.endsWith('.json') && !f.includes('compact') && f !== 'golden-pinned-current.json')
        .map(f => f.replace('golden-pinned-', '').replace('.json', ''))
        .sort();
      
      return pinnedFiles;
    } catch (error) {
      console.warn('Could not list pinned datasets:', error);
      return [];
    }
  }

  /**
   * Create a mock repo snapshot for compatibility with existing code
   */
  getCurrentSnapshots() {
    if (!this.pinnedDataset) {
      return [];
    }

    const snapshot = {
      repo_ref: 'pinned-dataset',
      repo_sha: this.pinnedDataset.git_sha,
      manifest: {}, // Not needed for pinned datasets
      timestamp: this.pinnedDataset.pinned_at,
      language_distribution: this.pinnedDataset.language_distribution,
      total_files: this.pinnedDataset.corpus_stats.total_files,
      total_lines: 0 // Not tracked in pinned dataset
    };

    return [snapshot];
  }

  /**
   * Get the loaded dataset (for inspection/debugging)
   */
  getPinnedDataset() {
    return this.pinnedDataset;
  }

  /**
   * Recursively walk directory to collect all files with relative paths
   */
  async walkDirectory(dir, fileSet, baseDir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        await this.walkDirectory(fullPath, fileSet, baseDir);
      } else if (entry.isFile()) {
        const relativePath = path.relative(baseDir, fullPath);
        if (relativePath.endsWith('.py') || relativePath.endsWith('.ts') || 
            relativePath.endsWith('.tsx') || relativePath.endsWith('.js')) {
          fileSet.add(relativePath);
        }
      }
    }
  }
}