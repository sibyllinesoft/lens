/**
 * Centrality Configuration Management
 * 
 * Manages hardening measures and configuration for centrality system deployment
 * with strict gating, caps, and query-type restrictions.
 */

interface StageAConfig {
  centralityPriorEnabled: boolean;
  queryTypes: ('NL' | 'symbol' | 'path')[];
  centralityLogOddsCap: number;
  alpha: number; // Stage-A parameter (0.2 baseline → 0.25 optimized)
}

interface SpanConfig {
  basePerFileSpanCap: number;
  highTopicSimCap: number;
  topicSimilarityThreshold: number;
}

interface MMRConfig {
  enabled: boolean;
  gamma: number;
  delta: number;
  overviewQueryOverride: boolean;
}

interface IncrementalGraphConfig {
  pushBasedUpdates: boolean;
  kHopNeighborhoodRecompute: boolean;
  fullPRWeeklySchedule: boolean;
  residualThresholdAlerts: boolean;
  hotSubgraphMonitoring: boolean;
}

interface PersonalizationConfig {
  pprSeedsRAPTORTopics: boolean;
  pprSeedsLSPSymbols: boolean;
  userClickPersonalization: boolean;
}

export class CentralityConfig {
  private stageAConfig: StageAConfig;
  private spanConfig: SpanConfig;
  private mmrConfig: MMRConfig;
  private incrementalGraphConfig: IncrementalGraphConfig;
  private personalizationConfig: PersonalizationConfig;

  constructor() {
    this.initializeDefaultConfig();
  }

  private initializeDefaultConfig(): void {
    // Hardening measures as specified
    this.stageAConfig = {
      centralityPriorEnabled: true,
      queryTypes: ['NL', 'symbol'], // Only NL and symbol, not path
      centralityLogOddsCap: 0.4,     // Cap net centrality log-odds ≤ 0.4
      alpha: 0.2                     // Base parameter, can be tuned to 0.25
    };

    this.spanConfig = {
      basePerFileSpanCap: 5,         // Preserve ≤ 5 spans per file
      highTopicSimCap: 8,            // Allow up to 8 only at high topic-similarity
      topicSimilarityThreshold: 0.7  // Threshold for high topic-similarity
    };

    this.mmrConfig = {
      enabled: false,                // Leave MMR off initially
      gamma: 0.10,                  // MMR diversity parameter
      delta: 0.05,                  // MMR relevance parameter
      overviewQueryOverride: true   // Enable only for "overview" queries
    };

    this.incrementalGraphConfig = {
      pushBasedUpdates: true,        // Push-based update budget
      kHopNeighborhoodRecompute: true, // Recompute only affected k-hop neighborhoods
      fullPRWeeklySchedule: true,    // Full PageRank weekly
      residualThresholdAlerts: true, // Alerts on hot subgraphs
      hotSubgraphMonitoring: true
    };

    this.personalizationConfig = {
      pprSeedsRAPTORTopics: true,    // PPR seeds = RAPTOR topics
      pprSeedsLSPSymbols: true,      // PPR seeds = LSP symbols  
      userClickPersonalization: false // No user-click personalization yet
    };
  }

  public async updateStageAConfig(config: Partial<StageAConfig>): Promise<void> {
    console.log('⚙️ Updating Stage-A centrality configuration...');
    
    this.stageAConfig = { ...this.stageAConfig, ...config };
    
    // Validate configuration constraints
    this.validateStageAConfig();
    
    // Apply configuration to the system
    await this.applyStageAConfig();
    
    console.log('✅ Stage-A configuration updated:', this.stageAConfig);
  }

  public async updateSpanConfig(config: Partial<SpanConfig>): Promise<void> {
    console.log('⚙️ Updating span configuration...');
    
    this.spanConfig = { ...this.spanConfig, ...config };
    
    // Apply span limits with topic-similarity awareness
    await this.applySpanConfig();
    
    console.log('✅ Span configuration updated:', this.spanConfig);
  }

  public async updateMMRConfig(config: Partial<MMRConfig>): Promise<void> {
    console.log('⚙️ Updating MMR configuration...');
    
    this.mmrConfig = { ...this.mmrConfig, ...config };
    
    // Apply MMR configuration with query type awareness
    await this.applyMMRConfig();
    
    console.log('✅ MMR configuration updated:', this.mmrConfig);
  }

  private validateStageAConfig(): void {
    // Ensure centrality log-odds cap is within safe bounds
    if (this.stageAConfig.centralityLogOddsCap > 0.4) {
      console.warn('⚠️ Centrality log-odds cap exceeds safe limit of 0.4, clamping');
      this.stageAConfig.centralityLogOddsCap = 0.4;
    }

    // Ensure only safe query types are enabled
    const safeQueryTypes: ('NL' | 'symbol' | 'path')[] = ['NL', 'symbol'];
    this.stageAConfig.queryTypes = this.stageAConfig.queryTypes.filter(
      type => safeQueryTypes.includes(type)
    );

    if (this.stageAConfig.queryTypes.length === 0) {
      throw new Error('At least one query type must be enabled for centrality');
    }
  }

  private async applyStageAConfig(): Promise<void> {
    // Implementation would interface with the actual centrality system
    const config = {
      'stage_a.centrality_prior.enabled': this.stageAConfig.centralityPriorEnabled,
      'stage_a.centrality_prior.query_types': this.stageAConfig.queryTypes,
      'stage_a.centrality_prior.log_odds_cap': this.stageAConfig.centralityLogOddsCap,
      'stage_a.alpha': this.stageAConfig.alpha
    };
    
    console.log('Applying Stage-A config:', config);
    // await configService.updateStageAConfig(config);
  }

  private async applySpanConfig(): Promise<void> {
    const config = {
      'per_file_span_cap.base': this.spanConfig.basePerFileSpanCap,
      'per_file_span_cap.high_topic_sim': this.spanConfig.highTopicSimCap,
      'topic_similarity_threshold': this.spanConfig.topicSimilarityThreshold
    };
    
    console.log('Applying span config:', config);
    // await configService.updateSpanConfig(config);
  }

  private async applyMMRConfig(): Promise<void> {
    const config = {
      'mmr.enabled': this.mmrConfig.enabled,
      'mmr.gamma': this.mmrConfig.gamma,
      'mmr.delta': this.mmrConfig.delta,
      'mmr.overview_query_override': this.mmrConfig.overviewQueryOverride
    };
    
    console.log('Applying MMR config:', config);
    // await configService.updateMMRConfig(config);
  }

  public async enableCentralityForQueryTypes(queryTypes: ('NL' | 'symbol' | 'path')[]): Promise<void> {
    await this.updateStageAConfig({ 
      centralityPriorEnabled: true,
      queryTypes: queryTypes.filter(type => ['NL', 'symbol'].includes(type))
    });
  }

  public async setCentralityLogOddsCap(cap: number): Promise<void> {
    const safeCap = Math.min(cap, 0.4); // Enforce hard limit
    await this.updateStageAConfig({ centralityLogOddsCap: safeCap });
  }

  public async adjustSpanCapForTopicSimilarity(topicSimilarity: number): Promise<number> {
    if (topicSimilarity >= this.spanConfig.topicSimilarityThreshold) {
      return this.spanConfig.highTopicSimCap; // 8 spans
    }
    return this.spanConfig.basePerFileSpanCap; // 5 spans
  }

  public async enableMMRForOverview(): Promise<void> {
    await this.updateMMRConfig({
      enabled: true,
      overviewQueryOverride: true
    });
  }

  public async tuneStageAParameter(alpha: number): Promise<void> {
    // Optimization: increase alpha from 0.2 to 0.25 for better positives-in-candidates
    const tunedAlpha = Math.min(alpha, 0.3); // Safety limit
    await this.updateStageAConfig({ alpha: tunedAlpha });
  }

  public async setupIncrementalGraphUpkeep(): Promise<void> {
    console.log('⚙️ Setting up incremental graph upkeep...');
    
    // Configure push-based updates for affected k-hop neighborhoods
    const graphConfig = {
      'graph.push_based_updates': this.incrementalGraphConfig.pushBasedUpdates,
      'graph.k_hop_neighborhood_recompute': this.incrementalGraphConfig.kHopNeighborhoodRecompute,
      'graph.full_pr_weekly': this.incrementalGraphConfig.fullPRWeeklySchedule,
      'graph.residual_threshold_alerts': this.incrementalGraphConfig.residualThresholdAlerts,
      'graph.hot_subgraph_monitoring': this.incrementalGraphConfig.hotSubgraphMonitoring
    };
    
    console.log('Applying incremental graph config:', graphConfig);
    // await configService.updateGraphConfig(graphConfig);
    
    console.log('✅ Incremental graph upkeep configured');
  }

  public async setupPersonalizationScope(): Promise<void> {
    console.log('⚙️ Setting up personalization scope...');
    
    const personalizationConfig = {
      'ppr.seeds.raptor_topics': this.personalizationConfig.pprSeedsRAPTORTopics,
      'ppr.seeds.lsp_symbols': this.personalizationConfig.pprSeedsLSPSymbols,
      'ppr.user_click_personalization': this.personalizationConfig.userClickPersonalization
    };
    
    console.log('Applying personalization config:', personalizationConfig);
    // await configService.updatePersonalizationConfig(personalizationConfig);
    
    console.log('✅ Personalization scope configured: RAPTOR topics ∪ LSP symbols');
  }

  public getHardeningConfig(): {
    stageA: StageAConfig;
    span: SpanConfig;
    mmr: MMRConfig;
    incrementalGraph: IncrementalGraphConfig;
    personalization: PersonalizationConfig;
  } {
    return {
      stageA: this.stageAConfig,
      span: this.spanConfig,  
      mmr: this.mmrConfig,
      incrementalGraph: this.incrementalGraphConfig,
      personalization: this.personalizationConfig
    };
  }

  public async validateHardeningMeasures(): Promise<{
    valid: boolean;
    violations: string[];
  }> {
    const violations: string[] = [];
    
    // Check centrality log-odds cap
    if (this.stageAConfig.centralityLogOddsCap > 0.4) {
      violations.push(`Centrality log-odds cap ${this.stageAConfig.centralityLogOddsCap} exceeds limit 0.4`);
    }
    
    // Check query type restrictions
    const unsafeTypes = this.stageAConfig.queryTypes.filter(type => !['NL', 'symbol'].includes(type));
    if (unsafeTypes.length > 0) {
      violations.push(`Unsafe query types enabled: ${unsafeTypes.join(', ')}`);
    }
    
    // Check span cap safety
    if (this.spanConfig.basePerFileSpanCap > 5) {
      violations.push(`Base span cap ${this.spanConfig.basePerFileSpanCap} exceeds safe limit 5`);
    }
    
    // Check MMR is disabled initially
    if (this.mmrConfig.enabled && !this.mmrConfig.overviewQueryOverride) {
      violations.push('MMR enabled without overview query override restriction');
    }
    
    return {
      valid: violations.length === 0,
      violations
    };
  }
}