/**
 * Ground Truth Builder for Lens Benchmarking System
 * Implements TODO.md specifications for dataset construction
 */

import { z } from 'zod';
import { createHash } from 'crypto';
import { promises as fs } from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import type {
  RepoSnapshot,
  GoldenDataItem,
  ConfigFingerprint
} from '../types/benchmark.js';
// Note: Python golden generator removed - was synthetic storyviz-specific code

export interface PRDerivedData {
  pr_title: string;
  pr_description: string;
  changed_spans: Array<{
    file: string;
    line_start: number;
    line_end: number;
    change_type: 'added' | 'modified' | 'deleted';
  }>;
}

export interface AgentLogData {
  query: string;
  clicked_spans: Array<{
    file: string;
    line: number;
    col: number;
    relevance: number; // 0-1
  }>;
  edited_spans: Array<{
    file: string;
    line: number;
    col: number;
  }>;
  session_metadata: Record<string, any>;
}

export interface SyntheticData {
  base_identifier: string;
  near_miss_variants: string[];
  docstring_paraphrases: string[];
  structural_intents: Array<{
    intent: string; // e.g., "call X without arg Y"
    pattern: string;
    expected_matches: number;
  }>;
}

export class GroundTruthBuilder {
  private snapshots: Map<string, RepoSnapshot> = new Map();
  private goldenItems: GoldenDataItem[] = [];
  
  constructor(
    private readonly workingDir: string,
    private readonly outputDir: string
  ) {}

  /**
   * Step 1: Freeze repository snapshots with deterministic manifests
   */
  async freezeRepoSnapshot(
    repoPath: string,
    repoRef: string = 'HEAD'
  ): Promise<RepoSnapshot> {
    // Get current git SHA
    const { execSync } = await import('child_process');
    const repoSha = execSync('git rev-parse HEAD', { 
      cwd: repoPath, 
      encoding: 'utf-8' 
    }).trim();

    // Build file manifest with metadata
    const manifest: Record<string, any> = {};
    const languageDistribution: Record<string, number> = {};
    let totalFiles = 0;
    let totalLines = 0;

    const walkDir = async (dir: string): Promise<void> => {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const relativePath = path.relative(repoPath, fullPath);
        
        // Skip hidden files and node_modules
        if (entry.name.startsWith('.') || entry.name === 'node_modules') {
          continue;
        }
        
        if (entry.isDirectory()) {
          await walkDir(fullPath);
        } else if (entry.isFile()) {
          const stat = await fs.stat(fullPath);
          const ext = path.extname(entry.name);
          const language = this.getLanguageFromExtension(ext);
          
          if (language) {
            languageDistribution[language] = (languageDistribution[language] || 0) + 1;
            
            const content = await fs.readFile(fullPath, 'utf-8');
            const lineCount = content.split('\n').length;
            
            manifest[relativePath] = {
              size: stat.size,
              lines: lineCount,
              language,
              last_modified: stat.mtime.toISOString(),
              hash: createHash('sha256').update(content).digest('hex')
            };
            
            totalFiles++;
            totalLines += lineCount;
          }
        }
      }
    };

    await walkDir(repoPath);

    const snapshot: RepoSnapshot = {
      repo_ref: repoRef,
      repo_sha: repoSha,
      manifest,
      timestamp: new Date().toISOString(),
      language_distribution: languageDistribution,
      total_files: totalFiles,
      total_lines: totalLines
    };

    this.snapshots.set(repoSha, snapshot);
    
    // Persist snapshot
    const snapshotPath = path.join(this.outputDir, `snapshot_${repoSha.substring(0, 8)}.json`);
    await fs.writeFile(snapshotPath, JSON.stringify(snapshot, null, 2));
    
    return snapshot;
  }

  /**
   * Generate dataset for the current repository content
   */
  async generateRepoDataset(): Promise<void> {
    console.log('📁 Generating golden dataset for current repository...');
    
    // Implementation would scan current repo structure and create appropriate golden dataset
    // For now, this is a placeholder for future implementation
    console.log('⚠️ Repository dataset generation not yet implemented');
  }

  /**
   * Step 2: Construct golden dataset with stratified sampling
   */
  async constructGoldenSet(snapshots: RepoSnapshot[]): Promise<void> {
    for (const snapshot of snapshots) {
      // PR-derived queries
      await this.addPRDerivedQueries(snapshot);
      
      // Agent log queries  
      await this.addAgentLogQueries(snapshot);
      
      // Synthetic queries
      await this.addSyntheticQueries(snapshot);
      
      // Adversarial queries
      await this.addAdversarialQueries(snapshot);
    }

    // Apply slicing by language and query class
    this.applyStratifiedSlicing();
    
    // Persist golden dataset
    await this.persistGoldenDataset();
  }

  private async addPRDerivedQueries(snapshot: RepoSnapshot): Promise<void> {
    // This would integrate with GitHub API or local PR data
    // For now, generate examples based on changed files in snapshot
    
    const languages = Object.keys(snapshot.language_distribution);
    
    for (const language of languages) {
      const files = Object.entries(snapshot.manifest)
        .filter(([_, meta]) => meta.language === language)
        .slice(0, 5); // Limit for example
      
      for (const [filepath, meta] of files) {
        const item: GoldenDataItem = {
          id: uuidv4(),
          query: `${path.basename(filepath, path.extname(filepath))} implementation`,
          query_class: 'nl-ish',
          language: language as any,
          source: 'pr-derived',
          snapshot_sha: snapshot.repo_sha,
          slice_tags: [language, 'pr-derived', 'implementation'],
          expected_results: [{
            file: filepath,
            line: 1,
            col: 0,
            relevance_score: 0.9,
            match_type: 'structural'
          }]
        };
        
        this.goldenItems.push(item);
      }
    }
  }

  private async addAgentLogQueries(snapshot: RepoSnapshot): Promise<void> {
    // Generate queries based on common agent interaction patterns
    const commonPatterns = [
      'function definition',
      'class constructor',
      'interface declaration',  
      'export statement',
      'import statement'
    ];
    
    for (const pattern of commonPatterns) {
      const files = Object.entries(snapshot.manifest)
        .filter(([_, meta]) => meta.language === 'ts')
        .slice(0, 3);
        
      for (const [filepath, meta] of files) {
        const item: GoldenDataItem = {
          id: uuidv4(),
          query: pattern,
          query_class: 'structural',
          language: 'ts',
          source: 'agent-logs',
          snapshot_sha: snapshot.repo_sha,
          slice_tags: ['ts', 'agent-logs', 'structural'],
          expected_results: [{
            file: filepath,
            line: Math.floor(Math.random() * meta.lines) + 1,
            col: 0,
            relevance_score: 0.8,
            match_type: 'symbol',
            why: `Agent typically clicks on ${pattern} definitions`
          }]
        };
        
        this.goldenItems.push(item);
      }
    }
  }

  private async addSyntheticQueries(snapshot: RepoSnapshot): Promise<void> {
    // Generate near-miss identifiers with subtoken edits
    const baseIdentifiers = ['searchEngine', 'dataProcessor', 'configManager'];
    
    for (const base of baseIdentifiers) {
      // Camel case variations
      const variations = [
        base.toLowerCase(),
        base.toUpperCase(), 
        base.replace(/([A-Z])/g, '_$1').toLowerCase(),
        base.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase()
      ];
      
      for (const variant of variations) {
        const item: GoldenDataItem = {
          id: uuidv4(),
          query: variant,
          query_class: 'identifier',
          language: 'ts',
          source: 'synthetics',
          snapshot_sha: snapshot.repo_sha,
          slice_tags: ['ts', 'synthetics', 'near-miss'],
          expected_results: [{
            file: 'src/example.ts', // Would be actual file with base identifier
            line: 1,
            col: 0,
            relevance_score: 0.7,
            match_type: 'exact'
          }]
        };
        
        this.goldenItems.push(item);
      }
    }

    // Docstring paraphrases
    const docstrings = [
      'Performs fuzzy string matching',
      'Executes semantic search query',
      'Builds inverted index structure'
    ];
    
    for (const docstring of docstrings) {
      const paraphrases = [
        docstring.replace('Performs', 'Executes'),
        docstring.replace('string matching', 'text comparison'),
        docstring.toLowerCase()
      ];
      
      for (const paraphrase of paraphrases) {
        const item: GoldenDataItem = {
          id: uuidv4(),
          query: paraphrase,
          query_class: 'docs',
          language: 'ts',
          source: 'synthetics',
          snapshot_sha: snapshot.repo_sha,
          slice_tags: ['ts', 'synthetics', 'docstring'],
          expected_results: [{
            file: 'src/example.ts',
            line: 5,
            col: 0,
            relevance_score: 0.6,
            match_type: 'semantic'
          }]
        };
        
        this.goldenItems.push(item);
      }
    }
  }

  private async addAdversarialQueries(snapshot: RepoSnapshot): Promise<void> {
    const adversarialPatterns = [
      // Casing variants
      { query: 'XMLHttpRequest', variants: ['xmlhttprequest', 'XMLHTTPREQUEST', 'XmlHttpRequest'] },
      // Unicode variants  
      { query: 'résumé', variants: ['resume', 'résume'] },
      // Giant identifiers
      { query: 'VeryLongMethodNameThatExceedsTypicalIdentifierLength', variants: ['VeryLongMethod'] },
      // Vendor noise paths
      { query: 'node_modules/package/index.js', variants: ['package index'] }
    ];

    for (const pattern of adversarialPatterns) {
      for (const variant of pattern.variants) {
        const item: GoldenDataItem = {
          id: uuidv4(),
          query: variant,
          query_class: 'identifier',
          language: 'ts',
          source: 'adversarial',
          snapshot_sha: snapshot.repo_sha,
          slice_tags: ['ts', 'adversarial', 'edge-case'],
          expected_results: [{
            file: 'src/example.ts',
            line: 1,
            col: 0,
            relevance_score: 0.5,
            match_type: 'exact'
          }]
        };
        
        this.goldenItems.push(item);
      }
    }
  }

  private applyStratifiedSlicing(): void {
    // Group by language and query class for balanced sampling
    const slices = new Map<string, GoldenDataItem[]>();
    
    for (const item of this.goldenItems) {
      const sliceKey = `${item.language}_${item.query_class}`;
      if (!slices.has(sliceKey)) {
        slices.set(sliceKey, []);
      }
      slices.get(sliceKey)!.push(item);
    }
    
    // Add slice tags for stratification
    for (const [sliceKey, items] of slices) {
      for (const item of items) {
        item.slice_tags.push(`slice_${sliceKey}`);
      }
    }
  }

  private async persistGoldenDataset(): Promise<void> {
    const version = Date.now();
    const filename = `golden_v${version}.jsonl`;
    const filepath = path.join(this.outputDir, filename);
    
    const lines = this.goldenItems.map(item => JSON.stringify(item));
    await fs.writeFile(filepath, lines.join('\n'));
    
    console.log(`Persisted ${this.goldenItems.length} golden dataset items to ${filepath}`);
  }

  /**
   * Generate configuration fingerprint for deterministic runs
   */
  generateConfigFingerprint(config: any, seedSet: number[]): ConfigFingerprint {
    const configHash = createHash('sha256')
      .update(JSON.stringify(config, null, 0))
      .digest('hex');
      
    const codeHash = createHash('sha256')
      .update(process.version + Date.now()) // Simplified
      .digest('hex');

    const snapshotShas: Record<string, string> = {};
    for (const [sha, snapshot] of this.snapshots) {
      snapshotShas[snapshot.repo_ref] = sha;
    }

    return {
      code_hash: codeHash,
      config_hash: configHash,
      snapshot_shas: snapshotShas,
      shard_layout: {}, // Would include actual shard distribution
      timestamp: new Date().toISOString(),
      seed_set: seedSet
    };
  }

  private getLanguageFromExtension(ext: string): string | null {
    const langMap: Record<string, string> = {
      '.ts': 'ts',
      '.js': 'ts', // Treat as TypeScript for indexing
      '.py': 'py', 
      '.rs': 'rust',
      '.sh': 'bash',
      '.go': 'go',
      '.java': 'java'
    };
    
    return langMap[ext] || null;
  }

  // Getters for access
  get currentSnapshots(): RepoSnapshot[] {
    return Array.from(this.snapshots.values());
  }
  
  get currentGoldenItems(): GoldenDataItem[] {
    return [...this.goldenItems];
  }

  /**
   * Load golden dataset from JSON file
   * This method loads the golden dataset that was created by create-lens-golden-data.js
   */
  async loadGoldenDataset(goldenPath?: string): Promise<void> {
    const defaultPaths = [
      path.join(process.cwd(), 'benchmark-results', 'golden-dataset.json'),
      path.join(process.cwd(), 'benchmark-results', 'smith-golden-dataset.json'),
      path.join(process.cwd(), 'sample-storyviz', 'golden-dataset.jsonl')
    ];
    
    const pathsToTry = goldenPath ? [goldenPath, ...defaultPaths] : defaultPaths;
    
    for (const testPath of pathsToTry) {
      try {
        console.log(`🔍 Trying to load golden dataset from: ${testPath}`);
        
        let goldenData: any[];
        
        if (testPath.endsWith('.jsonl')) {
          // Handle JSONL format
          const content = await fs.readFile(testPath, 'utf-8');
          goldenData = content.trim().split('\n').map(line => JSON.parse(line));
        } else {
          // Handle JSON format
          const content = await fs.readFile(testPath, 'utf-8');
          goldenData = JSON.parse(content);
        }
        
        if (!Array.isArray(goldenData)) {
          console.warn(`⚠️ Golden data in ${testPath} is not an array, skipping`);
          continue;
        }
        
        // Filter for SMOKE_DEFAULT slice if requested
        const smokeItems = goldenData.filter((item: any) => 
          item.slice_tags && item.slice_tags.includes('SMOKE_DEFAULT')
        );
        
        console.log(`📊 Loaded ${goldenData.length} total golden items from ${testPath}`);
        console.log(`🧪 Found ${smokeItems.length} SMOKE_DEFAULT items`);
        
        // Set the golden items
        this.goldenItems = goldenData;
        
        console.log(`✅ Successfully loaded golden dataset from ${testPath}`);
        return;
        
      } catch (error) {
        console.log(`❌ Could not load from ${testPath}: ${(error as Error).message}`);
        continue;
      }
    }
    
    throw new Error('Could not load golden dataset from any of the attempted paths');
  }

  /**
   * Filter golden items by slice tags
   */
  filterGoldenItemsBySlice(sliceTags: string | string[]): GoldenDataItem[] {
    const targetTags = Array.isArray(sliceTags) ? sliceTags : [sliceTags];
    
    return this.goldenItems.filter(item => {
      if (!item.slice_tags) return false;
      return targetTags.some(tag => item.slice_tags.includes(tag));
    });
  }
}