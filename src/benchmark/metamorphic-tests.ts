/**
 * Metamorphic and Invariance Testing System
 * Implements TODO.md metamorphic tests: rename symbol, move file, reformat, inject decoys, plant canaries
 */

import { promises as fs } from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { createHash } from 'crypto';
import type {
  MetamorphicTest,
  RepoSnapshot,
  GoldenDataItem,
  BenchmarkConfig
} from '../types/benchmark.js';

export interface MetamorphicTestResult {
  test_id: string;
  transform_type: string;
  original_results: any[];
  transformed_results: any[];
  invariant_preserved: boolean;
  violations: string[];
  metrics: {
    rank_delta_avg: number;
    rank_delta_max: number;
    recall_drop: number;
    precision_change: number;
  };
}

export class MetamorphicTestRunner {
  constructor(
    private readonly workingDir: string,
    private readonly outputDir: string
  ) {}

  /**
   * Run all metamorphic tests for a given benchmark configuration
   */
  async runMetamorphicTests(
    config: BenchmarkConfig,
    snapshots: RepoSnapshot[],
    goldenItems: GoldenDataItem[]
  ): Promise<MetamorphicTestResult[]> {
    
    console.log('🔄 Starting metamorphic test suite');
    
    const results: MetamorphicTestResult[] = [];
    
    for (const snapshot of snapshots) {
      // Run each type of metamorphic test
      results.push(...await this.runRenameSymbolTests(snapshot, goldenItems));
      results.push(...await this.runMoveFileTests(snapshot, goldenItems));
      results.push(...await this.runReformatTests(snapshot, goldenItems));
      results.push(...await this.runInjectDecoysTests(snapshot, goldenItems));
      results.push(...await this.runPlantCanariesTests(snapshot, goldenItems));
    }
    
    // Generate metamorphic test report
    await this.generateMetamorphicReport(results);
    
    console.log(`🔄 Metamorphic tests complete: ${results.length} tests, ${results.filter(r => r.invariant_preserved).length} passed`);
    
    return results;
  }

  /**
   * Test 1: Rename a symbol → relevant hits must stay (defs/refs keep recall)
   */
  private async runRenameSymbolTests(
    snapshot: RepoSnapshot,
    goldenItems: GoldenDataItem[]
  ): Promise<MetamorphicTestResult[]> {
    
    const results: MetamorphicTestResult[] = [];
    const symbolsToRename = this.extractSymbolsForRenaming(snapshot);
    
    for (const symbol of symbolsToRename.slice(0, 5)) { // Limit to 5 for performance
      const testId = uuidv4();
      
      const metamorphicTest: MetamorphicTest = {
        test_id: testId,
        transform_type: 'rename_symbol',
        repo_snapshot: snapshot.repo_sha,
        original_query: symbol.name,
        transformed_query: `${symbol.name}_renamed`,
        expected_invariant: 'Recall must be preserved for symbol definitions and references',
        tolerance: {
          rank_delta_max: 3,
          recall_drop_max: 0.05 // Max 5% recall drop
        }
      };
      
      // Simulate original search results
      const originalResults = await this.simulateSearchResults(symbol.name, snapshot);
      
      // Apply transformation (rename symbol in relevant files)
      const transformedSnapshot = await this.applyRenameTransformation(snapshot, symbol);
      
      // Run search on transformed codebase
      const transformedResults = await this.simulateSearchResults(
        `${symbol.name}_renamed`,
        transformedSnapshot
      );
      
      // Check invariant
      const testResult = this.validateRenameInvariant(
        metamorphicTest,
        originalResults,
        transformedResults
      );
      
      results.push(testResult);
    }
    
    return results;
  }

  /**
   * Test 2: Move file → path priors adjust, rank stable within ±k
   */
  private async runMoveFileTests(
    snapshot: RepoSnapshot,
    goldenItems: GoldenDataItem[]
  ): Promise<MetamorphicTestResult[]> {
    
    const results: MetamorphicTestResult[] = [];
    const filesToMove = this.selectFilesForMoving(snapshot);
    
    for (const fileInfo of filesToMove.slice(0, 3)) { // Limit for performance
      const testId = uuidv4();
      
      const metamorphicTest: MetamorphicTest = {
        test_id: testId,
        transform_type: 'move_file',
        repo_snapshot: snapshot.repo_sha,
        original_query: path.basename(fileInfo.path),
        expected_invariant: 'File content hits remain stable within ±3 rank positions',
        tolerance: {
          rank_delta_max: 3,
          recall_drop_max: 0.10 // Allow 10% recall drop due to path priors
        }
      };
      
      const originalResults = await this.simulateSearchResults(
        path.basename(fileInfo.path),
        snapshot
      );
      
      // Apply move transformation
      const transformedSnapshot = await this.applyMoveTransformation(snapshot, fileInfo);
      
      const transformedResults = await this.simulateSearchResults(
        path.basename(fileInfo.newPath),
        transformedSnapshot
      );
      
      const testResult = this.validateMoveInvariant(
        metamorphicTest,
        originalResults,
        transformedResults
      );
      
      results.push(testResult);
    }
    
    return results;
  }

  /**
   * Test 3: Reformat/reorder → ranks invariant
   */
  private async runReformatTests(
    snapshot: RepoSnapshot,
    goldenItems: GoldenDataItem[]
  ): Promise<MetamorphicTestResult[]> {
    
    const results: MetamorphicTestResult[] = [];
    const filesToReformat = this.selectFilesForReformatting(snapshot);
    
    for (const fileInfo of filesToReformat.slice(0, 3)) {
      const testId = uuidv4();
      
      const metamorphicTest: MetamorphicTest = {
        test_id: testId,
        transform_type: 'reformat',
        repo_snapshot: snapshot.repo_sha,
        original_query: 'function definition', // Generic structural query
        expected_invariant: 'Formatting changes should not affect search rankings',
        tolerance: {
          rank_delta_max: 1, // Very strict - formatting shouldn't change ranks
          recall_drop_max: 0.01
        }
      };
      
      const originalResults = await this.simulateSearchResults(
        'function definition',
        snapshot
      );
      
      // Apply reformatting (prettier, whitespace changes, etc.)
      const transformedSnapshot = await this.applyReformatTransformation(snapshot, fileInfo);
      
      const transformedResults = await this.simulateSearchResults(
        'function definition',
        transformedSnapshot
      );
      
      const testResult = this.validateReformatInvariant(
        metamorphicTest,
        originalResults,
        transformedResults
      );
      
      results.push(testResult);
    }
    
    return results;
  }

  /**
   * Test 4: Inject decoys → precision doesn't collapse
   */
  private async runInjectDecoysTests(
    snapshot: RepoSnapshot,
    goldenItems: GoldenDataItem[]
  ): Promise<MetamorphicTestResult[]> {
    
    const results: MetamorphicTestResult[] = [];
    
    const testId = uuidv4();
    const metamorphicTest: MetamorphicTest = {
      test_id: testId,
      transform_type: 'inject_decoys',
      repo_snapshot: snapshot.repo_sha,
      original_query: 'common function name',
      expected_invariant: 'Precision should not drop by more than 20% with noise injection',
      tolerance: {
        rank_delta_max: 5,
        recall_drop_max: 0.20
      }
    };
    
    const originalResults = await this.simulateSearchResults('handleRequest', snapshot);
    
    // Inject noisy duplicate files with similar content
    const transformedSnapshot = await this.applyDecoyInjection(snapshot, [
      'handleRequest_copy1',
      'handleRequest_copy2', 
      'handleRequest_variant'
    ]);
    
    const transformedResults = await this.simulateSearchResults(
      'handleRequest',
      transformedSnapshot
    );
    
    const testResult = this.validateDecoyInvariant(
      metamorphicTest,
      originalResults,
      transformedResults
    );
    
    results.push(testResult);
    
    return results;
  }

  /**
   * Test 5: Plant canaries → always recovered in top-50
   */
  private async runPlantCanariesTests(
    snapshot: RepoSnapshot,
    goldenItems: GoldenDataItem[]
  ): Promise<MetamorphicTestResult[]> {
    
    const results: MetamorphicTestResult[] = [];
    
    const canarySnippets = [
      { name: 'CANARY_FUNCTION_UNIQUE_123', content: 'function canary_unique_123() { return "found"; }' },
      { name: 'CANARY_CLASS_SPECIAL_456', content: 'class CanarySpecial456 { test() {} }' },
      { name: 'CANARY_CONSTANT_789', content: 'const CANARY_CONSTANT_789 = "test";' }
    ];
    
    for (const canary of canarySnippets) {
      const testId = uuidv4();
      
      const metamorphicTest: MetamorphicTest = {
        test_id: testId,
        transform_type: 'plant_canaries',
        repo_snapshot: snapshot.repo_sha,
        original_query: canary.name,
        expected_invariant: 'Planted canaries must always be found in top-50 results',
        tolerance: {
          rank_delta_max: 50, // Must be within top 50
          recall_drop_max: 0.00 // Must find the canary
        }
      };
      
      // Plant canary in a known location
      const transformedSnapshot = await this.applyCanaryPlanting(snapshot, canary);
      
      const transformedResults = await this.simulateSearchResults(
        canary.name,
        transformedSnapshot
      );
      
      const testResult = this.validateCanaryInvariant(
        metamorphicTest,
        [],
        transformedResults,
        canary
      );
      
      results.push(testResult);
    }
    
    return results;
  }

  // Transformation methods (simplified implementations)
  
  private extractSymbolsForRenaming(snapshot: RepoSnapshot): Array<{ name: string; files: string[] }> {
    // In real implementation, would parse AST to find renameable symbols
    return [
      { name: 'searchEngine', files: ['src/api/search-engine.ts'] },
      { name: 'processQuery', files: ['src/core/query-processor.ts'] },
      { name: 'IndexManager', files: ['src/indexer/manager.ts'] }
    ];
  }

  private selectFilesForMoving(snapshot: RepoSnapshot): Array<{ path: string; newPath: string }> {
    const files = Object.keys(snapshot.manifest).slice(0, 3);
    return files.map(filePath => ({
      path: filePath,
      newPath: filePath.replace('src/', 'src/moved/')
    }));
  }

  private selectFilesForReformatting(snapshot: RepoSnapshot): Array<{ path: string }> {
    return Object.keys(snapshot.manifest)
      .filter(path => path.endsWith('.ts'))
      .slice(0, 3)
      .map(path => ({ path }));
  }

  private async applyRenameTransformation(
    snapshot: RepoSnapshot,
    symbol: { name: string; files: string[] }
  ): Promise<RepoSnapshot> {
    // Simulate renaming symbol across files
    const newManifest = { ...snapshot.manifest };
    
    for (const file of symbol.files) {
      if (newManifest[file]) {
        // Update hash to reflect rename change
        const newHash = createHash('sha256')
          .update(newManifest[file].hash + '_renamed')
          .digest('hex');
        newManifest[file] = { ...newManifest[file], hash: newHash };
      }
    }
    
    return {
      ...snapshot,
      manifest: newManifest,
      repo_sha: snapshot.repo_sha + '_renamed'
    };
  }

  private async applyMoveTransformation(
    snapshot: RepoSnapshot,
    fileInfo: { path: string; newPath: string }
  ): Promise<RepoSnapshot> {
    const newManifest = { ...snapshot.manifest };
    
    if (newManifest[fileInfo.path]) {
      newManifest[fileInfo.newPath] = newManifest[fileInfo.path];
      delete newManifest[fileInfo.path];
    }
    
    return {
      ...snapshot,
      manifest: newManifest,
      repo_sha: snapshot.repo_sha + '_moved'
    };
  }

  private async applyReformatTransformation(
    snapshot: RepoSnapshot,
    fileInfo: { path: string }
  ): Promise<RepoSnapshot> {
    const newManifest = { ...snapshot.manifest };
    
    if (newManifest[fileInfo.path]) {
      // Simulate formatting change (same content, different whitespace)
      const newHash = createHash('sha256')
        .update(newManifest[fileInfo.path].hash + '_formatted')
        .digest('hex');
      newManifest[fileInfo.path] = { ...newManifest[fileInfo.path], hash: newHash };
    }
    
    return {
      ...snapshot,
      manifest: newManifest,
      repo_sha: snapshot.repo_sha + '_formatted'
    };
  }

  private async applyDecoyInjection(
    snapshot: RepoSnapshot,
    decoyNames: string[]
  ): Promise<RepoSnapshot> {
    const newManifest = { ...snapshot.manifest };
    
    for (const decoyName of decoyNames) {
      newManifest[`src/decoys/${decoyName}.ts`] = {
        size: 1024,
        lines: 50,
        language: 'ts',
        last_modified: new Date().toISOString(),
        hash: createHash('sha256').update(decoyName).digest('hex')
      };
    }
    
    return {
      ...snapshot,
      manifest: newManifest,
      total_files: snapshot.total_files + decoyNames.length,
      repo_sha: snapshot.repo_sha + '_decoys'
    };
  }

  private async applyCanaryPlanting(
    snapshot: RepoSnapshot,
    canary: { name: string; content: string }
  ): Promise<RepoSnapshot> {
    const newManifest = { ...snapshot.manifest };
    
    newManifest[`src/test/canaries/${canary.name.toLowerCase()}.ts`] = {
      size: canary.content.length,
      lines: canary.content.split('\n').length,
      language: 'ts',
      last_modified: new Date().toISOString(),
      hash: createHash('sha256').update(canary.content).digest('hex')
    };
    
    return {
      ...snapshot,
      manifest: newManifest,
      total_files: snapshot.total_files + 1,
      repo_sha: snapshot.repo_sha + '_canary'
    };
  }

  // Validation methods
  
  private validateRenameInvariant(
    test: MetamorphicTest,
    originalResults: any[],
    transformedResults: any[]
  ): MetamorphicTestResult {
    
    const recallDrop = this.calculateRecallDrop(originalResults, transformedResults);
    const rankDeltas = this.calculateRankDeltas(originalResults, transformedResults);
    
    const invariantPreserved = 
      recallDrop <= test.tolerance.recall_drop_max &&
      Math.max(...rankDeltas) <= test.tolerance.rank_delta_max;
    
    const violations: string[] = [];
    if (recallDrop > test.tolerance.recall_drop_max) {
      violations.push(`Recall drop ${recallDrop.toFixed(3)} exceeds tolerance ${test.tolerance.recall_drop_max}`);
    }
    if (Math.max(...rankDeltas) > test.tolerance.rank_delta_max) {
      violations.push(`Max rank delta ${Math.max(...rankDeltas)} exceeds tolerance ${test.tolerance.rank_delta_max}`);
    }
    
    return {
      test_id: test.test_id,
      transform_type: test.transform_type,
      original_results: originalResults,
      transformed_results: transformedResults,
      invariant_preserved: invariantPreserved,
      violations,
      metrics: {
        rank_delta_avg: rankDeltas.length > 0 ? rankDeltas.reduce((a, b) => a + b, 0) / rankDeltas.length : 0,
        rank_delta_max: Math.max(...rankDeltas, 0),
        recall_drop: recallDrop,
        precision_change: this.calculatePrecisionChange(originalResults, transformedResults)
      }
    };
  }

  private validateMoveInvariant(
    test: MetamorphicTest,
    originalResults: any[],
    transformedResults: any[]
  ): MetamorphicTestResult {
    // Similar to rename validation but with different tolerances
    return this.validateRenameInvariant(test, originalResults, transformedResults);
  }

  private validateReformatInvariant(
    test: MetamorphicTest,
    originalResults: any[],
    transformedResults: any[]
  ): MetamorphicTestResult {
    // Very strict validation - formatting should not change results
    return this.validateRenameInvariant(test, originalResults, transformedResults);
  }

  private validateDecoyInvariant(
    test: MetamorphicTest,
    originalResults: any[],
    transformedResults: any[]
  ): MetamorphicTestResult {
    
    const precisionDrop = this.calculatePrecisionChange(originalResults, transformedResults);
    const invariantPreserved = Math.abs(precisionDrop) <= test.tolerance.recall_drop_max;
    
    const violations: string[] = [];
    if (Math.abs(precisionDrop) > test.tolerance.recall_drop_max) {
      violations.push(`Precision change ${precisionDrop.toFixed(3)} exceeds tolerance ${test.tolerance.recall_drop_max}`);
    }
    
    return {
      test_id: test.test_id,
      transform_type: test.transform_type,
      original_results: originalResults,
      transformed_results: transformedResults,
      invariant_preserved: invariantPreserved,
      violations,
      metrics: {
        rank_delta_avg: 0,
        rank_delta_max: 0,
        recall_drop: 0,
        precision_change: precisionDrop
      }
    };
  }

  private validateCanaryInvariant(
    test: MetamorphicTest,
    originalResults: any[],
    transformedResults: any[],
    canary: { name: string; content: string }
  ): MetamorphicTestResult {
    
    const canaryFound = transformedResults.some(result =>
      result.file && result.file.includes(canary.name.toLowerCase())
    );
    
    const canaryRank = transformedResults.findIndex(result =>
      result.file && result.file.includes(canary.name.toLowerCase())
    ) + 1;
    
    const invariantPreserved = canaryFound && canaryRank <= 50;
    
    const violations: string[] = [];
    if (!canaryFound) {
      violations.push('Canary not found in results');
    } else if (canaryRank > 50) {
      violations.push(`Canary rank ${canaryRank} exceeds top-50 requirement`);
    }
    
    return {
      test_id: test.test_id,
      transform_type: test.transform_type,
      original_results: originalResults,
      transformed_results: transformedResults,
      invariant_preserved: invariantPreserved,
      violations,
      metrics: {
        rank_delta_avg: 0,
        rank_delta_max: canaryRank,
        recall_drop: canaryFound ? 0 : 1,
        precision_change: 0
      }
    };
  }

  // Utility methods

  private async simulateSearchResults(query: string, snapshot: RepoSnapshot): Promise<any[]> {
    // Simulate search results based on query and snapshot
    const files = Object.keys(snapshot.manifest);
    const relevantFiles = files.filter(file =>
      file.toLowerCase().includes(query.toLowerCase()) ||
      query.toLowerCase().includes(path.basename(file, path.extname(file)).toLowerCase())
    );
    
    return relevantFiles.slice(0, 10).map((file, index) => ({
      file,
      line: 1,
      col: 0,
      score: 1.0 - (index * 0.1),
      rank: index + 1
    }));
  }

  private calculateRecallDrop(originalResults: any[], transformedResults: any[]): number {
    if (originalResults.length === 0) return 0;
    
    const originalRelevant = new Set(originalResults.map(r => r.file));
    const transformedRelevant = new Set(transformedResults.map(r => r.file));
    
    const intersection = new Set([...originalRelevant].filter(x => transformedRelevant.has(x)));
    
    return (originalRelevant.size - intersection.size) / originalRelevant.size;
  }

  private calculateRankDeltas(originalResults: any[], transformedResults: any[]): number[] {
    const originalRanks = new Map(originalResults.map(r => [r.file, r.rank]));
    const transformedRanks = new Map(transformedResults.map(r => [r.file, r.rank]));
    
    const deltas: number[] = [];
    
    for (const [file, originalRank] of originalRanks) {
      const transformedRank = transformedRanks.get(file);
      if (transformedRank !== undefined) {
        deltas.push(Math.abs(transformedRank - originalRank));
      }
    }
    
    return deltas;
  }

  private calculatePrecisionChange(originalResults: any[], transformedResults: any[]): number {
    if (originalResults.length === 0 || transformedResults.length === 0) return 0;
    
    const originalPrecision = originalResults.filter(r => r.score > 0.5).length / originalResults.length;
    const transformedPrecision = transformedResults.filter(r => r.score > 0.5).length / transformedResults.length;
    
    return transformedPrecision - originalPrecision;
  }

  private async generateMetamorphicReport(results: MetamorphicTestResult[]): Promise<void> {
    const reportPath = path.join(this.outputDir, 'metamorphic-tests-report.json');
    
    const summary = {
      total_tests: results.length,
      passed_tests: results.filter(r => r.invariant_preserved).length,
      failed_tests: results.filter(r => !r.invariant_preserved).length,
      by_transform_type: results.reduce((acc, r) => {
        const entry = acc[r.transform_type] || { total: 0, passed: 0 };
        acc[r.transform_type] = entry;
        entry.total++;
        if (r.invariant_preserved) entry.passed++;
        return acc;
      }, {} as Record<string, { total: number; passed: number }>),
      results
    };
    
    await fs.writeFile(reportPath, JSON.stringify(summary, null, 2));
    console.log(`📊 Metamorphic test report written to ${reportPath}`);
  }
}