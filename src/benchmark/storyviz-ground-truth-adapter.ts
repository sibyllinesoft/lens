/**
 * Ground Truth Adapter for StoryViz corpus
 * Provides golden dataset for benchmark system
 */

import { promises as fs } from 'fs';
import path from 'path';
import type { GoldenDataItem } from '../types/benchmark.js';

export class StoryVizGroundTruthAdapter {
  private goldenItems: GoldenDataItem[] = [];
  private loaded = false;

  constructor(private readonly dataDir: string = './src/benchmark') {}

  async loadGoldenDataset(): Promise<void> {
    if (this.loaded) return;

    try {
      const groundTruthFile = path.join(this.dataDir, 'storyviz-ground-truth.json');
      const data = JSON.parse(await fs.readFile(groundTruthFile, 'utf-8'));
      this.goldenItems = data.goldenItems || [];
      this.loaded = true;
      
      console.log(`📊 Loaded ${this.goldenItems.length} golden items from storyviz corpus`);
    } catch (error) {
      console.error('❌ Failed to load storyviz ground truth:', error);
      throw new Error(`Failed to load storyviz ground truth data: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }

  get currentGoldenItems(): GoldenDataItem[] {
    if (!this.loaded) {
      throw new Error('Ground truth data not loaded. Call loadGoldenDataset() first.');
    }
    return this.goldenItems;
  }

  generateConfigFingerprint(config: any, seeds: number[]): any {
    const crypto = require('crypto');
    const configString = JSON.stringify({
      ...config,
      seeds,
      corpus_type: 'storyviz',
      golden_items_count: this.goldenItems.length
    });
    
    return {
      config_hash: crypto.createHash('md5').update(configString).digest('hex'),
      corpus_type: 'storyviz',
      golden_items_count: this.goldenItems.length,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get golden items filtered by query class
   */
  getGoldenItemsByClass(queryClass: string): GoldenDataItem[] {
    return this.goldenItems.filter(item => item.query_class === queryClass);
  }

  /**
   * Get golden items filtered by language
   */
  getGoldenItemsByLanguage(language: string): GoldenDataItem[] {
    return this.goldenItems.filter(item => item.language === language);
  }

  /**
   * Get sample of golden items for smoke testing
   */
  getSmokeTestSample(count: number = 50): GoldenDataItem[] {
    // Stratified sampling by query_class and language
    const strata = new Map<string, GoldenDataItem[]>();
    
    for (const item of this.goldenItems) {
      const key = `${item.query_class}_${item.language}`;
      if (!strata.has(key)) {
        strata.set(key, []);
      }
      strata.get(key)!.push(item);
    }

    const sample: GoldenDataItem[] = [];
    const samplesPerStratum = Math.ceil(count / strata.size);

    for (const [_, items] of strata) {
      const stratumSample = items.slice(0, Math.min(samplesPerStratum, items.length));
      sample.push(...stratumSample);
    }

    return sample.slice(0, count);
  }

  /**
   * Get statistics about the golden dataset
   */
  getDatasetStatistics(): any {
    const stats = {
      total_items: this.goldenItems.length,
      by_query_class: {} as Record<string, number>,
      by_language: {} as Record<string, number>,
      by_match_type: {} as Record<string, number>
    };

    for (const item of this.goldenItems) {
      // Count by query class
      stats.by_query_class[item.query_class] = (stats.by_query_class[item.query_class] || 0) + 1;
      
      // Count by language
      stats.by_language[item.language] = (stats.by_language[item.language] || 0) + 1;
      
      // Count by match type (from first expected result)
      if (item.expected_results.length > 0 && item.expected_results[0]) {
        const matchType = item.expected_results[0].match_type || 'unknown';
        stats.by_match_type[matchType] = (stats.by_match_type[matchType] || 0) + 1;
      }
    }

    return stats;
  }
}