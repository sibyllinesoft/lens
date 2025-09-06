/**
 * Coverage Tracker for LSIF/ctags Symbol Indexing Monitoring
 * Phase B2 Enhancement: Monitor and enforce symbol indexing coverage 
 * Tracks coverage metrics, identifies gaps, and provides insights for optimization
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SymbolDefinition, SymbolKind } from '../types/core.js';
import type { SupportedLanguage } from '../types/api.js';
import * as path from 'path';

export interface CoverageMetrics {
  totalFiles: number;
  indexedFiles: number;
  coveragePercentage: number;
  symbolCoverage: {
    [K in SymbolKind]: {
      total: number;
      indexed: number;
      coverage: number;
    };
  };
  languageCoverage: {
    [L in SupportedLanguage]: {
      files: number;
      indexed: number;
      coverage: number;
      symbols: number;
    };
  };
  lastUpdated: number;
}

export interface FileIndexingStatus {
  filePath: string;
  language: SupportedLanguage;
  indexed: boolean;
  lastIndexed?: number;
  symbolCount: number;
  indexingTime?: number;
  errors?: string[];
  coverage: {
    functions: number;
    classes: number;
    interfaces: number;
    types: number;
    variables: number;
    total: number;
  };
}

export interface CoverageGap {
  type: 'missing_file' | 'partial_symbols' | 'stale_index' | 'indexing_error';
  filePath: string;
  language: SupportedLanguage;
  description: string;
  impact: 'high' | 'medium' | 'low';
  suggestedAction: string;
  details?: Record<string, any>;
}

export interface CoverageReport {
  timestamp: number;
  metrics: CoverageMetrics;
  gaps: CoverageGap[];
  recommendations: string[];
  performanceInsights: {
    slowestFiles: Array<{ path: string; time: number; language: SupportedLanguage }>;
    errorProneFiles: Array<{ path: string; errors: number; lastError: string }>;
    coverageByDirectory: Array<{ directory: string; coverage: number; fileCount: number }>;
  };
}

export interface CoverageTrackerConfig {
  trackingEnabled: boolean;
  reportingInterval: number;
  staleThreshold: number;
  minimumCoverageThreshold: number;
  enablePerformanceTracking: boolean;
  enableGapAnalysis: boolean;
  maxReportHistory: number;
}

/**
 * Comprehensive coverage tracking and monitoring system
 */
export class CoverageTracker {
  private fileStatuses = new Map<string, FileIndexingStatus>();
  private coverageHistory: CoverageReport[] = [];
  private reportingTimer?: NodeJS.Timeout | undefined;
  
  private config: CoverageTrackerConfig;

  // Performance tracking
  private indexingTimes = new Map<string, number[]>();
  private errorCounts = new Map<string, number>();
  private lastErrors = new Map<string, string>();

  constructor(config: Partial<CoverageTrackerConfig> = {}) {
    this.config = {
      trackingEnabled: true,
      reportingInterval: 300000, // 5 minutes
      staleThreshold: 3600000,   // 1 hour
      minimumCoverageThreshold: 85, // 85%
      enablePerformanceTracking: true,
      enableGapAnalysis: true,
      maxReportHistory: 100,
      ...config
    };

    if (this.config.trackingEnabled) {
      this.startPeriodicReporting();
    }
  }

  /**
   * Record successful file indexing
   */
  recordFileIndexing(
    filePath: string,
    language: SupportedLanguage,
    symbols: SymbolDefinition[],
    indexingTimeMs?: number
  ): void {
    const span = LensTracer.createChildSpan('coverage_record_indexing', {
      'file.path': filePath,
      'file.language': language,
      'symbols.count': symbols.length
    });

    try {
      // Count symbols by kind
      const symbolCounts = this.countSymbolsByKind(symbols);
      
      const status: FileIndexingStatus = {
        filePath,
        language,
        indexed: true,
        lastIndexed: Date.now(),
        symbolCount: symbols.length,
        coverage: {
          functions: symbolCounts['function'] || 0,
          classes: symbolCounts['class'] || 0,
          interfaces: symbolCounts['interface'] || 0,
          types: symbolCounts['type'] || 0,
          variables: symbolCounts['variable'] || 0,
          total: symbols.length
        }
      };

      if (indexingTimeMs !== undefined) {
        status.indexingTime = indexingTimeMs;
      }

      this.fileStatuses.set(filePath, status);

      // Track performance metrics
      if (this.config.enablePerformanceTracking && indexingTimeMs) {
        this.recordIndexingPerformance(filePath, indexingTimeMs);
      }

      span.setAttributes({
        success: true,
        'indexing.time_ms': indexingTimeMs || 0,
        'symbols.total': symbols.length
      });

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Record indexing error
   */
  recordIndexingError(
    filePath: string,
    language: SupportedLanguage,
    error: string
  ): void {
    const span = LensTracer.createChildSpan('coverage_record_error', {
      'file.path': filePath,
      'file.language': language,
      'error.message': error
    });

    try {
      const existing = this.fileStatuses.get(filePath);
      const status: FileIndexingStatus = {
        filePath,
        language,
        indexed: false,
        symbolCount: 0,
        errors: existing?.errors ? [...existing.errors, error] : [error],
        coverage: {
          functions: 0,
          classes: 0,
          interfaces: 0,
          types: 0,
          variables: 0,
          total: 0
        }
      };

      this.fileStatuses.set(filePath, status);

      // Track error metrics
      const currentCount = this.errorCounts.get(filePath) || 0;
      this.errorCounts.set(filePath, currentCount + 1);
      this.lastErrors.set(filePath, error);

      span.setAttributes({
        success: true,
        'error.count': currentCount + 1
      });

    } catch (trackingError) {
      span.recordException(trackingError as Error);
      span.setAttributes({ success: false });
      throw trackingError;
    } finally {
      span.end();
    }
  }

  /**
   * Register files that should be indexed
   */
  registerFiles(filePaths: string[], language: SupportedLanguage): void {
    const span = LensTracer.createChildSpan('coverage_register_files', {
      'files.count': filePaths.length,
      'files.language': language
    });

    try {
      for (const filePath of filePaths) {
        if (!this.fileStatuses.has(filePath)) {
          const status: FileIndexingStatus = {
            filePath,
            language,
            indexed: false,
            symbolCount: 0,
            coverage: {
              functions: 0,
              classes: 0,
              interfaces: 0,
              types: 0,
              variables: 0,
              total: 0
            }
          };
          this.fileStatuses.set(filePath, status);
        }
      }

      span.setAttributes({ 
        success: true,
        'files.registered': filePaths.length 
      });

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Generate comprehensive coverage report
   */
  generateReport(): CoverageReport {
    const span = LensTracer.createChildSpan('coverage_generate_report');

    try {
      const metrics = this.calculateMetrics();
      const gaps = this.config.enableGapAnalysis ? this.identifyGaps() : [];
      const recommendations = this.generateRecommendations(metrics, gaps);
      const performanceInsights = this.generatePerformanceInsights();

      const report: CoverageReport = {
        timestamp: Date.now(),
        metrics,
        gaps,
        recommendations,
        performanceInsights
      };

      // Store in history
      this.coverageHistory.push(report);
      if (this.coverageHistory.length > this.config.maxReportHistory) {
        this.coverageHistory.shift();
      }

      span.setAttributes({
        success: true,
        'metrics.coverage_percentage': metrics.coveragePercentage,
        'gaps.count': gaps.length,
        'recommendations.count': recommendations.length
      });

      return report;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get current coverage metrics
   */
  getCurrentMetrics(): CoverageMetrics {
    return this.calculateMetrics();
  }

  /**
   * Get files with low or no coverage
   */
  getLowCoverageFiles(threshold: number = 50): FileIndexingStatus[] {
    return Array.from(this.fileStatuses.values())
      .filter(status => {
        const coverage = status.indexed ? 100 : 0;
        return coverage < threshold;
      })
      .sort((a, b) => (a.symbolCount || 0) - (b.symbolCount || 0));
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats(): {
    averageIndexingTime: number;
    slowestFiles: Array<{ path: string; time: number }>;
    totalIndexingTime: number;
    filesWithErrors: number;
  } {
    let totalTime = 0;
    let totalFiles = 0;
    const slowestFiles: Array<{ path: string; time: number }> = [];
    let filesWithErrors = 0;

    for (const [filePath, times] of this.indexingTimes) {
      const avgTime = times.reduce((sum, time) => sum + time, 0) / times.length;
      totalTime += avgTime;
      totalFiles++;
      
      slowestFiles.push({ path: filePath, time: avgTime });
      
      if (this.errorCounts.has(filePath)) {
        filesWithErrors++;
      }
    }

    slowestFiles.sort((a, b) => b.time - a.time);

    return {
      averageIndexingTime: totalFiles > 0 ? totalTime / totalFiles : 0,
      slowestFiles: slowestFiles.slice(0, 10),
      totalIndexingTime: totalTime,
      filesWithErrors
    };
  }

  /**
   * Export coverage data for external analysis
   */
  exportCoverageData(): {
    fileStatuses: FileIndexingStatus[];
    metrics: CoverageMetrics;
    history: CoverageReport[];
  } {
    return {
      fileStatuses: Array.from(this.fileStatuses.values()),
      metrics: this.calculateMetrics(),
      history: this.coverageHistory.slice()
    };
  }

  /**
   * Clear tracking data
   */
  clear(): void {
    this.fileStatuses.clear();
    this.coverageHistory = [];
    this.indexingTimes.clear();
    this.errorCounts.clear();
    this.lastErrors.clear();
    console.log('📊 Coverage tracking data cleared');
  }

  /**
   * Stop tracking and cleanup
   */
  shutdown(): void {
    if (this.reportingTimer) {
      clearInterval(this.reportingTimer);
      this.reportingTimer = undefined;
    }
    this.clear();
    console.log('💤 Coverage tracker shut down');
  }

  // Private methods

  private calculateMetrics(): CoverageMetrics {
    const now = Date.now();
    const statuses = Array.from(this.fileStatuses.values());
    
    const totalFiles = statuses.length;
    const indexedFiles = statuses.filter(s => s.indexed).length;
    const coveragePercentage = totalFiles > 0 ? Math.round((indexedFiles / totalFiles) * 100) : 0;

    // Calculate symbol coverage by kind
    const symbolCoverage = {} as CoverageMetrics['symbolCoverage'];
    const symbolKinds: SymbolKind[] = ['function', 'class', 'interface', 'type', 'variable', 'method', 'property', 'constant', 'enum'];
    
    for (const kind of symbolKinds) {
      const total = statuses.reduce((sum, s) => sum + (s.coverage[kind as keyof typeof s.coverage] || 0), 0);
      const indexed = statuses.filter(s => s.indexed).reduce((sum, s) => sum + (s.coverage[kind as keyof typeof s.coverage] || 0), 0);
      
      symbolCoverage[kind] = {
        total,
        indexed,
        coverage: total > 0 ? Math.round((indexed / total) * 100) : 100
      };
    }

    // Calculate language coverage
    const languageCoverage = {} as CoverageMetrics['languageCoverage'];
    const languages: SupportedLanguage[] = ['typescript', 'python', 'rust', 'bash', 'go', 'java'];
    
    for (const language of languages) {
      const langFiles = statuses.filter(s => s.language === language);
      const langIndexed = langFiles.filter(s => s.indexed);
      
      languageCoverage[language] = {
        files: langFiles.length,
        indexed: langIndexed.length,
        coverage: langFiles.length > 0 ? Math.round((langIndexed.length / langFiles.length) * 100) : 100,
        symbols: langIndexed.reduce((sum, s) => sum + s.symbolCount, 0)
      };
    }

    return {
      totalFiles,
      indexedFiles,
      coveragePercentage,
      symbolCoverage,
      languageCoverage,
      lastUpdated: now
    };
  }

  private identifyGaps(): CoverageGap[] {
    const gaps: CoverageGap[] = [];
    const now = Date.now();

    for (const [filePath, status] of this.fileStatuses) {
      // Missing file indexing
      if (!status.indexed) {
        gaps.push({
          type: 'missing_file',
          filePath,
          language: status.language,
          description: 'File has not been indexed',
          impact: 'high',
          suggestedAction: 'Run indexing on this file',
          details: { errors: status.errors }
        });
      }
      
      // Stale index
      else if (status.lastIndexed && (now - status.lastIndexed) > this.config.staleThreshold) {
        gaps.push({
          type: 'stale_index',
          filePath,
          language: status.language,
          description: `Index is stale (${Math.round((now - status.lastIndexed) / 60000)} minutes old)`,
          impact: 'medium',
          suggestedAction: 'Re-index file to ensure freshness'
        });
      }
      
      // Low symbol coverage
      else if (status.symbolCount === 0 && status.indexed) {
        gaps.push({
          type: 'partial_symbols',
          filePath,
          language: status.language,
          description: 'File indexed but no symbols found',
          impact: 'medium',
          suggestedAction: 'Check parsing logic for this language'
        });
      }
      
      // Indexing errors
      if (status.errors && status.errors.length > 0) {
        gaps.push({
          type: 'indexing_error',
          filePath,
          language: status.language,
          description: `${status.errors.length} indexing errors occurred`,
          impact: 'high',
          suggestedAction: 'Fix indexing errors and retry',
          details: { errors: status.errors }
        });
      }
    }

    return gaps.sort((a, b) => {
      const impactOrder = { high: 0, medium: 1, low: 2 };
      return impactOrder[a.impact] - impactOrder[b.impact];
    });
  }

  private generateRecommendations(metrics: CoverageMetrics, gaps: CoverageGap[]): string[] {
    const recommendations: string[] = [];

    // Overall coverage recommendations
    if (metrics.coveragePercentage < this.config.minimumCoverageThreshold) {
      recommendations.push(
        `Overall coverage is ${metrics.coveragePercentage}%, below threshold of ${this.config.minimumCoverageThreshold}%. ` +
        'Consider running batch indexing on unindexed files.'
      );
    }

    // Language-specific recommendations
    for (const [language, stats] of Object.entries(metrics.languageCoverage)) {
      if (stats.coverage < 90 && stats.files > 0) {
        recommendations.push(
          `${language} coverage is ${stats.coverage}% (${stats.indexed}/${stats.files} files). ` +
          'Consider improving language-specific parsing patterns.'
        );
      }
    }

    // Gap-based recommendations
    const errorGaps = gaps.filter(g => g.type === 'indexing_error');
    if (errorGaps.length > 0) {
      recommendations.push(
        `${errorGaps.length} files have indexing errors. ` +
        'Review error logs and fix parsing issues.'
      );
    }

    const staleGaps = gaps.filter(g => g.type === 'stale_index');
    if (staleGaps.length > 5) {
      recommendations.push(
        `${staleGaps.length} files have stale indexes. ` +
        'Consider implementing more frequent re-indexing or file watching.'
      );
    }

    // Performance recommendations
    const perfStats = this.getPerformanceStats();
    if (perfStats.averageIndexingTime > 1000) { // > 1 second
      recommendations.push(
        `Average indexing time is ${Math.round(perfStats.averageIndexingTime)}ms. ` +
        'Consider optimizing parsing performance or enabling batch processing.'
      );
    }

    return recommendations;
  }

  private generatePerformanceInsights(): CoverageReport['performanceInsights'] {
    const perfStats = this.getPerformanceStats();
    
    // Group by directory for coverage analysis
    const coverageByDirectory = new Map<string, { coverage: number; fileCount: number }>();
    
    for (const [filePath, status] of this.fileStatuses) {
      const directory = path.dirname(filePath);
      const existing = coverageByDirectory.get(directory) || { coverage: 0, fileCount: 0 };
      
      existing.fileCount++;
      if (status.indexed) {
        existing.coverage++;
      }
      
      coverageByDirectory.set(directory, existing);
    }

    const directoryCoverage = Array.from(coverageByDirectory.entries()).map(([directory, stats]) => ({
      directory,
      coverage: Math.round((stats.coverage / stats.fileCount) * 100),
      fileCount: stats.fileCount
    })).sort((a, b) => a.coverage - b.coverage);

    // Error-prone files
    const errorProneFiles = Array.from(this.errorCounts.entries()).map(([filePath, errorCount]) => ({
      path: filePath,
      errors: errorCount,
      lastError: this.lastErrors.get(filePath) || 'Unknown error'
    })).sort((a, b) => b.errors - a.errors).slice(0, 10);

    return {
      slowestFiles: perfStats.slowestFiles.map(f => ({
        ...f,
        language: this.fileStatuses.get(f.path)?.language || 'typescript'
      })),
      errorProneFiles,
      coverageByDirectory: directoryCoverage
    };
  }

  private countSymbolsByKind(symbols: SymbolDefinition[]): Record<string, number> {
    const counts: Record<string, number> = {};
    
    for (const symbol of symbols) {
      counts[symbol.kind] = (counts[symbol.kind] || 0) + 1;
    }
    
    return counts;
  }

  private recordIndexingPerformance(filePath: string, timeMs: number): void {
    const existing = this.indexingTimes.get(filePath) || [];
    existing.push(timeMs);
    
    // Keep only last 10 measurements per file
    if (existing.length > 10) {
      existing.shift();
    }
    
    this.indexingTimes.set(filePath, existing);
  }

  private startPeriodicReporting(): void {
    this.reportingTimer = setInterval(() => {
      try {
        const report = this.generateReport();
        console.log(`📊 Coverage Report: ${report.metrics.coveragePercentage}% (${report.metrics.indexedFiles}/${report.metrics.totalFiles} files)`);
        
        if (report.gaps.length > 0) {
          console.log(`⚠️  Found ${report.gaps.length} coverage gaps`);
        }
      } catch (error) {
        console.warn('Failed to generate periodic coverage report:', error);
      }
    }, this.config.reportingInterval);
  }
}

// Export coverage monitoring utilities
export const COVERAGE_THRESHOLDS = {
  minimum: 75,
  good: 85,
  excellent: 95
} as const;

export const COVERAGE_PRIORITIES = {
  functions: 'high',
  classes: 'high', 
  interfaces: 'medium',
  types: 'medium',
  variables: 'low'
} as const;