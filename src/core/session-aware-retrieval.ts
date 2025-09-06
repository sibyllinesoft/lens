/**
 * Session-Aware Retrieval System
 * 
 * Maintains tiny state {topic_id, intent_hist, last_spans, repo_set} from past ~5 minutes.
 * Uses first-order semi-Markov to predict next intent/topic: P(next_topic|history).
 * Pre-spools compute: prefetch 1-2 shard entrypoints, raise per-file span cap for in-session files.
 * Bias Stage-B+ seeds toward recently selected symbols.
 * Session micro-cache keyed by (topic_id, repo, symbol), invalidate by index_version and span_hash.
 * 
 * Gates: Success@10 +0.5pp on multi-hop sessions, p95 ≤ +0.3ms, why-mix KL ≤ 0.02
 */

import type {
  SessionState,
  MarkovTransition,
  SessionAwareConfig,
  SessionMicroCache,
  SessionPrediction,
  SpanReference,
  SearchHit,
  AdvancedLeverMetrics
} from '../types/embedder-proof-levers.js';
import type { QueryIntent, SymbolKind } from '../types/core.js';

export class SessionAwareRetrievalSystem {
  private sessions: Map<string, SessionState> = new Map();
  private markovTransitions: Map<string, MarkovTransition[]> = new Map();
  private microCache: Map<string, SessionMicroCache> = new Map();
  private config: SessionAwareConfig;
  private metrics: AdvancedLeverMetrics['session_aware'];

  constructor(config: Partial<SessionAwareConfig> = {}) {
    this.config = {
      max_session_duration_minutes: 5,
      max_sessions_in_memory: 1000,
      prefetch_shard_count: 2,
      per_file_span_cap_multiplier: 2.0,
      markov_order: 1,
      session_cache_ttl_minutes: 10,
      min_transition_count: 5,
      ...config
    };

    this.metrics = {
      success_at_10_improvement: 0,
      p95_latency_impact_ms: 0,
      why_mix_kl_divergence: 0,
      cache_hit_rate: 0,
      session_prediction_accuracy: 0
    };

    // Start cleanup timer
    setInterval(() => this.cleanupExpiredSessions(), 60000); // Every minute
  }

  /**
   * Get or create session state for a query
   */
  public getOrCreateSession(
    sessionId: string,
    query: string,
    intent: QueryIntent,
    repoSha: string
  ): SessionState {
    const now = new Date();
    let session = this.sessions.get(sessionId);

    if (!session) {
      // Create new session
      session = {
        session_id: sessionId,
        topic_id: this.extractTopicId(query, intent),
        intent_history: [intent],
        last_spans: [],
        repo_set: new Set([repoSha]),
        created_at: now,
        last_accessed: now,
        ttl_minutes: this.config.max_session_duration_minutes
      };
      this.sessions.set(sessionId, session);
    } else {
      // Update existing session
      session.intent_history.push(intent);
      session.repo_set.add(repoSha);
      session.last_accessed = now;

      // Keep only recent intent history (sliding window)
      if (session.intent_history.length > 10) {
        session.intent_history = session.intent_history.slice(-10);
      }

      // Update topic if it has evolved
      const newTopicId = this.extractTopicId(query, intent);
      if (newTopicId !== session.topic_id) {
        this.recordTopicTransition(session.topic_id, newTopicId, session.intent_history);
        session.topic_id = newTopicId;
      }
    }

    return session;
  }

  /**
   * Predict next user intent/topic using semi-Markov model
   */
  public predictNextState(session: SessionState): SessionPrediction {
    const currentTopic = session.topic_id;
    const intentContext = session.intent_history.slice(-3); // Last 3 intents
    
    const transitions = this.markovTransitions.get(currentTopic) || [];
    const validTransitions = transitions.filter(t => 
      this.intentContextMatches(t.intent_context, intentContext) &&
      t.transition_count >= this.config.min_transition_count
    );

    if (validTransitions.length === 0) {
      // Fallback to default configuration
      return {
        next_topic: currentTopic,
        probability: 0.5,
        recommended_k: 20,
        per_file_cap: 50,
        ann_ef_search: 100,
        confidence: 0.1,
        reasoning: ['No sufficient transition history, using defaults']
      };
    }

    // Select most probable transition
    const bestTransition = validTransitions.reduce((best, current) => 
      current.probability > best.probability ? current : best
    );

    return {
      next_topic: bestTransition.to_topic,
      probability: bestTransition.probability,
      recommended_k: this.computeRecommendedK(bestTransition),
      per_file_cap: this.computePerFileCapacity(session, bestTransition),
      ann_ef_search: this.computeAnnEfSearch(bestTransition),
      confidence: this.computeConfidence(bestTransition),
      reasoning: [
        `Transition ${currentTopic} -> ${bestTransition.to_topic}`,
        `Probability: ${bestTransition.probability.toFixed(3)}`,
        `Based on ${bestTransition.transition_count} observations`
      ]
    };
  }

  /**
   * Update session with search results to learn from user interactions
   */
  public updateSessionWithResults(
    sessionId: string,
    results: SearchHit[],
    selectedResults: SearchHit[] = []
  ): void {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    // Update last spans with selected results
    const newSpans: SpanReference[] = selectedResults.map(hit => ({
      file_path: hit.file_path,
      line: hit.line,
      col: hit.col,
      span_len: hit.snippet?.length || 100,
      symbol_kind: hit.symbol_kind,
      access_count: 1,
      last_access: new Date()
    }));

    // Merge with existing spans, updating access counts
    for (const newSpan of newSpans) {
      const existing = session.last_spans.find(s => 
        s.file_path === newSpan.file_path && 
        s.line === newSpan.line
      );
      
      if (existing) {
        existing.access_count++;
        existing.last_access = newSpan.last_access;
      } else {
        session.last_spans.push(newSpan);
      }
    }

    // Keep only recent spans (sliding window)
    session.last_spans = session.last_spans
      .sort((a, b) => b.last_access.getTime() - a.last_access.getTime())
      .slice(0, 50);

    // Update micro-cache for future queries
    this.updateMicroCache(session, results);
  }

  /**
   * Get cached results for session-aware queries
   */
  public getCachedResults(
    topicId: string,
    repo: string,
    symbol: string,
    indexVersion: string
  ): SearchHit[] | null {
    const cacheKey = `${topicId}:${repo}:${symbol}`;
    const cached = this.microCache.get(cacheKey);

    if (!cached) return null;

    // Check if cache is still valid
    if (cached.index_version !== indexVersion) {
      this.microCache.delete(cacheKey);
      return null;
    }

    const now = new Date();
    const ageMinutes = (now.getTime() - cached.created_at.getTime()) / (1000 * 60);
    if (ageMinutes > this.config.session_cache_ttl_minutes) {
      this.microCache.delete(cacheKey);
      return null;
    }

    // Update cache statistics
    cached.hit_count++;
    this.metrics.cache_hit_rate = this.calculateCacheHitRate();

    return cached.results;
  }

  /**
   * Get session-aware biases for Stage-B+ processing
   */
  public getStageBoostBiases(session: SessionState): Map<string, number> {
    const biases = new Map<string, number>();
    
    // Boost recently accessed files
    for (const span of session.last_spans) {
      const ageMinutes = (new Date().getTime() - span.last_access.getTime()) / (1000 * 60);
      const recencyBoost = Math.exp(-ageMinutes / 2.0); // Exponential decay
      const accessBoost = Math.log(span.access_count + 1); // Log of access frequency
      
      biases.set(span.file_path, recencyBoost * accessBoost * 0.1); // Max 10% boost
    }

    return biases;
  }

  /**
   * Get prefetch recommendations for shard entrypoints
   */
  public getPrefetchRecommendations(session: SessionState): string[] {
    const prediction = this.predictNextState(session);
    
    // Identify most likely shards based on recent activity
    const fileFrequency = new Map<string, number>();
    for (const span of session.last_spans) {
      const count = fileFrequency.get(span.file_path) || 0;
      fileFrequency.set(span.file_path, count + span.access_count);
    }

    // Convert to shard recommendations (simplified - in practice would map files to shards)
    const sortedFiles = Array.from(fileFrequency.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, this.config.prefetch_shard_count)
      .map(([file]) => this.fileToShardId(file));

    return sortedFiles;
  }

  /**
   * Calculate why-mix KL divergence for quality gate
   */
  public calculateWhyMixKL(
    baselineDistribution: number[],
    sessionAwareDistribution: number[]
  ): number {
    if (baselineDistribution.length !== sessionAwareDistribution.length) {
      throw new Error('Distribution length mismatch');
    }

    let klDiv = 0;
    for (let i = 0; i < baselineDistribution.length; i++) {
      const p = baselineDistribution[i];
      const q = sessionAwareDistribution[i];
      
      if (p > 0 && q > 0) {
        klDiv += p * Math.log(p / q);
      }
    }

    this.metrics.why_mix_kl_divergence = klDiv;
    return klDiv;
  }

  /**
   * Get current metrics for monitoring
   */
  public getMetrics(): AdvancedLeverMetrics['session_aware'] {
    return { ...this.metrics };
  }

  // Private helper methods

  private extractTopicId(query: string, intent: QueryIntent): string {
    // Simplified topic extraction - in practice would use more sophisticated NLP
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    const contentWords = words.slice(0, 3).join('_');
    return `${intent}_${contentWords}`;
  }

  private recordTopicTransition(
    fromTopic: string,
    toTopic: string,
    intentHistory: QueryIntent[]
  ): void {
    if (fromTopic === toTopic) return;

    const key = fromTopic;
    const transitions = this.markovTransitions.get(key) || [];
    
    let existing = transitions.find(t => 
      t.to_topic === toTopic && 
      this.intentContextMatches(t.intent_context, intentHistory.slice(-3))
    );

    if (existing) {
      existing.transition_count++;
      existing.probability = this.updateProbability(transitions);
    } else {
      const newTransition: MarkovTransition = {
        from_topic: fromTopic,
        to_topic: toTopic,
        probability: 0,
        intent_context: intentHistory.slice(-3),
        transition_count: 1,
        confidence_interval: [0, 1]
      };
      
      transitions.push(newTransition);
      
      // Recalculate all probabilities for this from_topic
      const totalCount = transitions.reduce((sum, t) => sum + t.transition_count, 0);
      for (const t of transitions) {
        t.probability = t.transition_count / totalCount;
        t.confidence_interval = this.calculateConfidenceInterval(t.transition_count, totalCount);
      }
    }

    this.markovTransitions.set(key, transitions);
  }

  private intentContextMatches(context1: QueryIntent[], context2: QueryIntent[]): boolean {
    if (context1.length !== context2.length) return false;
    return context1.every((intent, i) => intent === context2[i]);
  }

  private updateProbability(transitions: MarkovTransition[]): number {
    const totalCount = transitions.reduce((sum, t) => sum + t.transition_count, 0);
    return transitions.reduce((sum, t) => sum + (t.transition_count / totalCount), 0);
  }

  private calculateConfidenceInterval(count: number, total: number): [number, number] {
    const p = count / total;
    const z = 1.96; // 95% confidence
    const se = Math.sqrt(p * (1 - p) / total);
    return [Math.max(0, p - z * se), Math.min(1, p + z * se)];
  }

  private computeRecommendedK(transition: MarkovTransition): number {
    // Higher K for uncertain transitions
    const baseK = 20;
    const uncertaintyMultiplier = 1 - transition.probability;
    return Math.round(baseK * (1 + uncertaintyMultiplier));
  }

  private computePerFileCapacity(session: SessionState, transition: MarkovTransition): number {
    // Higher capacity for files in session
    const baseCapacity = 50;
    return Math.round(baseCapacity * this.config.per_file_span_cap_multiplier);
  }

  private computeAnnEfSearch(transition: MarkovTransition): number {
    // Adjust ANN ef_search based on transition confidence
    const baseEf = 100;
    const confidenceBoost = transition.probability;
    return Math.round(baseEf * (1 + confidenceBoost * 0.5));
  }

  private computeConfidence(transition: MarkovTransition): number {
    const [lower, upper] = transition.confidence_interval;
    return 1 - (upper - lower); // Narrower interval = higher confidence
  }

  private updateMicroCache(session: SessionState, results: SearchHit[]): void {
    if (results.length === 0) return;

    const cacheKey = `${session.topic_id}:${Array.from(session.repo_set).join(',')}:${results[0].symbol_kind || 'unknown'}`;
    
    const cached: SessionMicroCache = {
      key: cacheKey,
      results: results.slice(0, 10), // Cache top 10 results
      index_version: 'current', // Would get from actual index version
      span_hash: this.calculateSpanHash(results),
      created_at: new Date(),
      hit_count: 0,
      cache_score: this.calculateCacheScore(results)
    };

    this.microCache.set(cacheKey, cached);

    // Limit cache size
    if (this.microCache.size > 1000) {
      const oldestKey = Array.from(this.microCache.keys())[0];
      this.microCache.delete(oldestKey);
    }
  }

  private calculateSpanHash(results: SearchHit[]): string {
    // Simple hash of result positions
    const positions = results.map(r => `${r.file_path}:${r.line}:${r.col}`).join('|');
    return btoa(positions).substring(0, 8); // Simple hash for demo
  }

  private calculateCacheScore(results: SearchHit[]): number {
    // Score based on result quality (simplified)
    return results.reduce((sum, r) => sum + r.score, 0) / results.length;
  }

  private calculateCacheHitRate(): number {
    if (this.microCache.size === 0) return 0;
    
    const totalHits = Array.from(this.microCache.values()).reduce((sum, cache) => sum + cache.hit_count, 0);
    const totalRequests = totalHits + this.microCache.size; // Simplified calculation
    
    return totalRequests > 0 ? totalHits / totalRequests : 0;
  }

  private fileToShardId(filePath: string): string {
    // Simplified mapping - in practice would use actual shard mapping
    return `shard_${filePath.split('/')[1] || 'default'}`;
  }

  private cleanupExpiredSessions(): void {
    const now = new Date();
    const expiredSessions: string[] = [];

    for (const [sessionId, session] of this.sessions) {
      const ageMinutes = (now.getTime() - session.last_accessed.getTime()) / (1000 * 60);
      if (ageMinutes > session.ttl_minutes) {
        expiredSessions.push(sessionId);
      }
    }

    for (const sessionId of expiredSessions) {
      this.sessions.delete(sessionId);
    }

    // Clean up old cache entries
    const expiredCacheKeys: string[] = [];
    for (const [key, cache] of this.microCache) {
      const ageMinutes = (now.getTime() - cache.created_at.getTime()) / (1000 * 60);
      if (ageMinutes > this.config.session_cache_ttl_minutes) {
        expiredCacheKeys.push(key);
      }
    }

    for (const key of expiredCacheKeys) {
      this.microCache.delete(key);
    }

    // Limit sessions in memory
    if (this.sessions.size > this.config.max_sessions_in_memory) {
      const sortedSessions = Array.from(this.sessions.entries())
        .sort(([, a], [, b]) => a.last_accessed.getTime() - b.last_accessed.getTime());
      
      const toRemove = sortedSessions.slice(0, this.sessions.size - this.config.max_sessions_in_memory);
      for (const [sessionId] of toRemove) {
        this.sessions.delete(sessionId);
      }
    }
  }
}

/**
 * Factory function to create session-aware retrieval system
 */
export function createSessionAwareRetrieval(
  config?: Partial<SessionAwareConfig>
): SessionAwareRetrievalSystem {
  return new SessionAwareRetrievalSystem(config);
}