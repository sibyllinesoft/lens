/**
 * Lexical Phrase/Proximity Scorer for Sprint-2
 * Implements Section 5 of TODO.md: lexical improvements with impact-ordered postings
 */

import fs from 'fs/promises';
import path from 'path';

export interface PhraseConfig {
  min_phrase_length: number;
  max_phrase_length: number;
  proximity_window: number;
  impact_threshold: number;
  entropy_threshold: number;
  precompute_hot_ngrams: boolean;
}

export interface PostingsList {
  term: string;
  document_frequency: number;
  impact_score: number;
  positions: Array<{
    doc_id: string;
    positions: number[];
    term_frequency: number;
  }>;
}

export interface PhraseWindow {
  phrase: string;
  ngram_length: number;
  frequency: number;
  positions: Map<string, number[]>; // doc_id -> positions
  precomputed: boolean;
}

export interface ScoringResult {
  doc_id: string;
  phrase_score: number;
  proximity_score: number;
  combined_score: number;
  matched_phrases: string[];
  scoring_method: 'exact' | 'proximity' | 'panic_exactifier';
  latency_ms: number;
}

export class LexicalPhraseScorer {
  private config: PhraseConfig;
  private postingsIndex: Map<string, PostingsList> = new Map();
  private phraseWindows: Map<string, PhraseWindow> = new Map();
  private hotNgrams: Set<string> = new Set();
  private entropyCache: Map<string, number> = new Map();

  constructor(config: PhraseConfig) {
    this.config = config;
  }

  async initialize(corpusPath: string): Promise<void> {
    console.log('🚀 Initializing lexical phrase scorer...');
    
    // Build impact-ordered postings
    await this.buildImpactOrderedPostings(corpusPath);
    
    // Precompute hot n-grams if enabled
    if (this.config.precompute_hot_ngrams) {
      await this.precomputeHotNgrams();
    }
    
    console.log('✅ Phrase scorer initialized');
    console.log(`📊 Postings: ${this.postingsIndex.size}, Hot n-grams: ${this.hotNgrams.size}`);
  }

  async scoreQuery(
    query: string,
    candidates: Array<{ doc_id: string; content: string }>,
    slaDeadlineMs: number = 150
  ): Promise<ScoringResult[]> {
    const startTime = Date.now();
    const results: ScoringResult[] = [];
    
    // Calculate query entropy to decide scoring strategy
    const queryEntropy = this.calculateQueryEntropy(query);
    const useExactifier = queryEntropy > this.config.entropy_threshold;
    
    console.log(`🔍 Scoring query: "${query}" (entropy: ${queryEntropy.toFixed(3)}, exactifier: ${useExactifier})`);
    
    // Extract phrases from query
    const queryPhrases = this.extractPhrases(query);
    
    for (const candidate of candidates) {
      const remainingTime = slaDeadlineMs - (Date.now() - startTime);
      if (remainingTime <= 0) {
        console.warn('⚠️ SLA deadline reached, stopping scoring');
        break;
      }
      
      let scoringResult: ScoringResult;
      
      if (useExactifier) {
        scoringResult = this.scorePanicExactifier(query, queryPhrases, candidate, startTime);
      } else {
        scoringResult = await this.scorePhrasesProximity(queryPhrases, candidate, startTime);
      }
      
      results.push(scoringResult);
    }
    
    // Sort by combined score descending
    results.sort((a, b) => b.combined_score - a.combined_score);
    
    const totalTime = Date.now() - startTime;
    console.log(`✅ Scored ${results.length} candidates in ${totalTime}ms`);
    
    return results;
  }

  private async buildImpactOrderedPostings(corpusPath: string): Promise<void> {
    console.log('📚 Building impact-ordered postings index...');
    
    // In real implementation, this would process the actual corpus
    // For now, create mock postings with realistic structure
    
    const mockTerms = [
      'class', 'function', 'import', 'const', 'let', 'var', 'async', 'await',
      'interface', 'type', 'export', 'default', 'return', 'throw', 'try', 'catch',
      'UserManager', 'authenticate', 'process', 'data', 'response', 'request'
    ];
    
    for (let i = 0; i < mockTerms.length; i++) {
      const term = mockTerms[i];
      const frequency = Math.floor(Math.random() * 1000) + 10;
      const impactScore = this.calculateImpactScore(term, frequency);
      
      // Create mock positions for multiple documents
      const positions = [];
      const numDocs = Math.floor(Math.random() * 50) + 5;
      
      for (let docId = 0; docId < numDocs; docId++) {
        const docPositions = [];
        const termFreq = Math.floor(Math.random() * 10) + 1;
        
        for (let pos = 0; pos < termFreq; pos++) {
          docPositions.push(Math.floor(Math.random() * 1000));
        }
        
        positions.push({
          doc_id: `doc_${docId}`,
          positions: docPositions.sort((a, b) => a - b),
          term_frequency: termFreq
        });
      }
      
      this.postingsIndex.set(term, {
        term,
        document_frequency: numDocs,
        impact_score: impactScore,
        positions
      });
    }
    
    console.log(`📊 Built postings for ${this.postingsIndex.size} terms`);
  }

  private async precomputeHotNgrams(): Promise<void> {
    console.log('🔥 Precomputing hot n-gram windows...');
    
    // Find frequently occurring n-grams
    const ngramCounts = new Map<string, number>();
    
    // Mock hot n-grams that would be extracted from real corpus analysis
    const hotPhrases = [
      'class UserManager',
      'function authenticate',
      'async function',
      'import React',
      'const config',
      'interface ApiResponse',
      'export default',
      'try catch',
      'throw new Error'
    ];
    
    for (const phrase of hotPhrases) {
      const frequency = Math.floor(Math.random() * 500) + 100;
      ngramCounts.set(phrase, frequency);
      
      // Mark as hot if above threshold
      if (frequency > 200) {
        this.hotNgrams.add(phrase);
        
        // Precompute phrase window
        const phraseWindow: PhraseWindow = {
          phrase,
          ngram_length: phrase.split(' ').length,
          frequency,
          positions: this.precomputePhrasePositions(phrase),
          precomputed: true
        };
        
        this.phraseWindows.set(phrase, phraseWindow);
      }
    }
    
    console.log(`🔥 Precomputed ${this.phraseWindows.size} hot n-gram windows`);
  }

  private precomputePhrasePositions(phrase: string): Map<string, number[]> {
    const positions = new Map<string, number[]>();
    
    // Mock precomputed positions for hot phrases
    const numDocs = Math.floor(Math.random() * 30) + 10;
    
    for (let i = 0; i < numDocs; i++) {
      const docId = `doc_${i}`;
      const docPositions = [];
      const occurrences = Math.floor(Math.random() * 5) + 1;
      
      for (let j = 0; j < occurrences; j++) {
        docPositions.push(Math.floor(Math.random() * 800));
      }
      
      positions.set(docId, docPositions.sort((a, b) => a - b));
    }
    
    return positions;
  }

  private extractPhrases(query: string): Array<{
    phrase: string;
    length: number;
    terms: string[];
  }> {
    const phrases = [];
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    
    // Extract all valid phrase lengths
    for (let len = this.config.min_phrase_length; len <= Math.min(this.config.max_phrase_length, words.length); len++) {
      for (let start = 0; start <= words.length - len; start++) {
        const phraseTerms = words.slice(start, start + len);
        const phrase = phraseTerms.join(' ');
        
        phrases.push({
          phrase,
          length: len,
          terms: phraseTerms
        });
      }
    }
    
    return phrases;
  }

  private async scorePhrasesProximity(
    queryPhrases: Array<{ phrase: string; length: number; terms: string[] }>,
    candidate: { doc_id: string; content: string },
    startTime: number
  ): Promise<ScoringResult> {
    let phraseScore = 0;
    let proximityScore = 0;
    const matchedPhrases: string[] = [];
    
    const candidateContent = candidate.content.toLowerCase();
    
    for (const queryPhrase of queryPhrases) {
      // Check for exact phrase match first
      if (candidateContent.includes(queryPhrase.phrase)) {
        phraseScore += queryPhrase.length * 2; // Bonus for longer phrases
        matchedPhrases.push(queryPhrase.phrase);
        
        // Use precomputed window if available
        if (this.phraseWindows.has(queryPhrase.phrase)) {
          proximityScore += 5; // Bonus for hot n-gram
        }
        
        continue;
      }
      
      // Calculate proximity score for individual terms
      const termPositions = this.findTermPositions(queryPhrase.terms, candidateContent);
      if (termPositions.length > 0) {
        const proximityBonus = this.calculateProximityBonus(termPositions, queryPhrase.terms.length);
        proximityScore += proximityBonus;
        
        if (proximityBonus > 0) {
          matchedPhrases.push(`~${queryPhrase.phrase}`); // Prefix with ~ for proximity match
        }
      }
    }
    
    const combinedScore = phraseScore + (proximityScore * 0.7); // Weight phrase matches higher
    
    return {
      doc_id: candidate.doc_id,
      phrase_score: phraseScore,
      proximity_score: proximityScore,
      combined_score: combinedScore,
      matched_phrases: matchedPhrases,
      scoring_method: 'proximity',
      latency_ms: Date.now() - startTime
    };
  }

  private scorePanicExactifier(
    query: string,
    queryPhrases: Array<{ phrase: string; length: number; terms: string[] }>,
    candidate: { doc_id: string; content: string },
    startTime: number
  ): ScoringResult {
    // Panic exactifier - fall back to simpler exact matching under high entropy
    console.log('⚡ Using panic exactifier for high entropy query');
    
    let exactScore = 0;
    const matchedPhrases: string[] = [];
    const candidateContent = candidate.content.toLowerCase();
    const queryLower = query.toLowerCase();
    
    // Check for exact query match
    if (candidateContent.includes(queryLower)) {
      exactScore += 10;
      matchedPhrases.push(queryLower);
    }
    
    // Check individual terms
    const queryTerms = queryLower.split(/\s+/);
    for (const term of queryTerms) {
      if (candidateContent.includes(term)) {
        exactScore += 1;
        matchedPhrases.push(term);
      }
    }
    
    return {
      doc_id: candidate.doc_id,
      phrase_score: exactScore,
      proximity_score: 0,
      combined_score: exactScore,
      matched_phrases: matchedPhrases,
      scoring_method: 'panic_exactifier',
      latency_ms: Date.now() - startTime
    };
  }

  private findTermPositions(terms: string[], content: string): number[] {
    const positions = [];
    
    for (const term of terms) {
      let index = 0;
      while ((index = content.indexOf(term, index)) !== -1) {
        positions.push(index);
        index++;
      }
    }
    
    return positions.sort((a, b) => a - b);
  }

  private calculateProximityBonus(positions: number[], termCount: number): number {
    if (positions.length < termCount) return 0;
    
    // Find the minimum window that contains all terms
    let minWindowSize = Infinity;
    
    for (let i = 0; i <= positions.length - termCount; i++) {
      const windowSize = positions[i + termCount - 1] - positions[i];
      if (windowSize < minWindowSize) {
        minWindowSize = windowSize;
      }
    }
    
    // Convert window size to proximity bonus (closer = higher score)
    if (minWindowSize <= this.config.proximity_window) {
      return Math.max(1, this.config.proximity_window - minWindowSize);
    }
    
    return 0;
  }

  private calculateQueryEntropy(query: string): number {
    if (this.entropyCache.has(query)) {
      return this.entropyCache.get(query)!;
    }
    
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    const wordCounts = new Map<string, number>();
    
    for (const word of words) {
      wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
    }
    
    let entropy = 0;
    const totalWords = words.length;
    
    for (const count of wordCounts.values()) {
      const probability = count / totalWords;
      entropy -= probability * Math.log2(probability);
    }
    
    this.entropyCache.set(query, entropy);
    return entropy;
  }

  private calculateImpactScore(term: string, frequency: number): number {
    // Impact score based on term characteristics and frequency
    let impact = frequency;
    
    // Boost for code-specific terms
    if (term.match(/^[A-Z][a-z]+[A-Z]/)) { // CamelCase
      impact *= 1.5;
    }
    
    if (['class', 'function', 'interface', 'const', 'let'].includes(term)) {
      impact *= 1.3;
    }
    
    // Normalize to 0-1 range
    return Math.min(1.0, impact / 10000);
  }

  // Performance benchmarking methods
  async benchmarkPerformance(
    testQueries: string[],
    testCandidates: Array<{ doc_id: string; content: string }>,
    iterations: number = 100
  ): Promise<{
    avg_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    quality_metrics: {
      precision_at_10: number;
      recall_at_50: number;
      ndcg_at_20: number;
    };
    pareto_curves: Array<{
      quality_score: number;
      latency_ms: number;
      config_hash: string;
    }>;
  }> {
    console.log(`🏃 Running performance benchmark (${iterations} iterations)...`);
    
    const latencies: number[] = [];
    const qualityScores: number[] = [];
    
    for (let i = 0; i < iterations; i++) {
      const query = testQueries[i % testQueries.length];
      const startTime = Date.now();
      
      const results = await this.scoreQuery(query, testCandidates);
      const latency = Date.now() - startTime;
      
      latencies.push(latency);
      
      // Mock quality score (in real implementation, compare against ground truth)
      const qualityScore = this.calculateMockQualityScore(results);
      qualityScores.push(qualityScore);
    }
    
    latencies.sort((a, b) => a - b);
    
    const p95_idx = Math.floor(latencies.length * 0.95);
    const p99_idx = Math.floor(latencies.length * 0.99);
    
    const avgLatency = latencies.reduce((sum, l) => sum + l, 0) / latencies.length;
    const avgQuality = qualityScores.reduce((sum, q) => sum + q, 0) / qualityScores.length;
    
    return {
      avg_latency_ms: avgLatency,
      p95_latency_ms: latencies[p95_idx],
      p99_latency_ms: latencies[p99_idx],
      quality_metrics: {
        precision_at_10: avgQuality * 0.9, // Mock precision
        recall_at_50: avgQuality * 0.85,   // Mock recall  
        ndcg_at_20: avgQuality * 0.8       // Mock NDCG
      },
      pareto_curves: this.generateParetoCurves(latencies, qualityScores)
    };
  }

  private calculateMockQualityScore(results: ScoringResult[]): number {
    // Mock quality calculation - in real implementation, use ground truth
    const avgScore = results.reduce((sum, r) => sum + r.combined_score, 0) / results.length;
    const phraseMatches = results.reduce((sum, r) => sum + r.matched_phrases.length, 0);
    
    return Math.min(1.0, (avgScore + phraseMatches) / 20);
  }

  private generateParetoCurves(latencies: number[], qualityScores: number[]): Array<{
    quality_score: number;
    latency_ms: number;
    config_hash: string;
  }> {
    const curves = [];
    
    for (let i = 0; i < Math.min(10, latencies.length); i++) {
      curves.push({
        quality_score: qualityScores[i],
        latency_ms: latencies[i],
        config_hash: this.generateConfigHash()
      });
    }
    
    return curves;
  }

  private generateConfigHash(): string {
    const configString = JSON.stringify(this.config);
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(configString).digest('hex').substring(0, 12);
  }

  // Gate validation for Sprint-2 requirements
  async validateSprintGates(
    baselineMetrics: {
      lexical_slice_recall: number;
      avg_p95_latency_ms: number;
    },
    newMetrics: {
      lexical_slice_recall: number;
      avg_p95_latency_ms: number;
    }
  ): Promise<{
    gates_passed: boolean;
    recall_improvement: number;
    latency_increase: number;
    violations: string[];
  }> {
    const violations: string[] = [];
    
    const recallImprovement = newMetrics.lexical_slice_recall - baselineMetrics.lexical_slice_recall;
    const latencyIncrease = newMetrics.avg_p95_latency_ms - baselineMetrics.avg_p95_latency_ms;
    
    // Gate: +1-2pp on lexical slices
    if (recallImprovement < 0.01) {
      violations.push(`Recall improvement ${(recallImprovement * 100).toFixed(2)}pp < required +1pp`);
    }
    
    if (recallImprovement > 0.025) {
      violations.push(`Recall improvement ${(recallImprovement * 100).toFixed(2)}pp > expected +2.5pp (suspicious)`);
    }
    
    // Gate: ≤ +0.5ms p95
    if (latencyIncrease > 0.5) {
      violations.push(`P95 latency increase ${latencyIncrease.toFixed(2)}ms > allowed +0.5ms`);
    }
    
    const gatesPassed = violations.length === 0;
    
    console.log(`📊 Sprint-2 Gates: ${gatesPassed ? '✅ PASSED' : '❌ FAILED'}`);
    console.log(`   Recall: ${baselineMetrics.lexical_slice_recall.toFixed(3)} → ${newMetrics.lexical_slice_recall.toFixed(3)} (+${(recallImprovement * 100).toFixed(2)}pp)`);
    console.log(`   P95 Latency: ${baselineMetrics.avg_p95_latency_ms.toFixed(1)}ms → ${newMetrics.avg_p95_latency_ms.toFixed(1)}ms (+${latencyIncrease.toFixed(2)}ms)`);
    
    return {
      gates_passed: gatesPassed,
      recall_improvement: recallImprovement,
      latency_increase: latencyIncrease,
      violations
    };
  }
}

// Factory function with default configuration
export function createLexicalPhraseScorer(overrides: Partial<PhraseConfig> = {}): LexicalPhraseScorer {
  const defaultConfig: PhraseConfig = {
    min_phrase_length: 2,
    max_phrase_length: 5,
    proximity_window: 50,
    impact_threshold: 0.5,
    entropy_threshold: 2.0, // Above this, use panic exactifier
    precompute_hot_ngrams: true
  };

  const config = { ...defaultConfig, ...overrides };
  return new LexicalPhraseScorer(config);
}