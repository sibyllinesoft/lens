/**
 * Semantic Card Types and Interfaces for RAPTOR System
 * 
 * A SemanticCard represents the semantic fingerprint of a file/function,
 * extracting roles, resources, shapes, domain tokens, and effects to enable
 * hierarchical clustering and semantic search.
 */

export type Language = "py" | "ts" | "js";

export type Role = 
  | "handler" 
  | "service" 
  | "repo" 
  | "job" 
  | "validator" 
  | "adapter";

export type Effect = 
  | "fs" 
  | "net" 
  | "db" 
  | "cache" 
  | "auth" 
  | "email" 
  | "crypto";

export interface Resources {
  routes: string[];
  sql: string[];
  topics: string[];
  buckets: string[];
  featureFlags: string[];
}

export interface Shapes {
  typeNames: string[];
  jsonKeys: string[];
}

export interface PathHints {
  ngrams: string[];
  depth: number;
  recentlyTouched: boolean;
}

export interface SemanticCard {
  file_id: string;
  file_sha: string;
  lang: Language;
  roles: Role[];
  resources: Resources;
  shapes: Shapes;
  domainTokens: string[];
  effects: Effect[];
  utilAffinity: number;        // [0..1], edges into known util namespaces
  pathHints: PathHints;
  e_sem: number[];            // facet embedding (fixed dim)
  B: number;                  // businessness score
}

export interface SemanticCardCacheKey {
  file_sha: string;
  strings_hash: string;
}

export interface SemanticCardExtractorConfig {
  embeddingDim: number;
  pmiThreshold: number;
  utilityPackages: string[];  // Known utility/framework packages
  domainStopWords: string[];  // Filter out generic tokens
}

export interface SemanticCardStats {
  total_cards: number;
  by_language: Record<Language, number>;
  by_role: Record<Role, number>;
  avg_businessness: number;
  avg_util_affinity: number;
  embedding_coverage: number;
}

export interface SemanticCardValidation {
  file_id: string;
  is_valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Utility functions for semantic cards
 */
export class SemanticCardUtils {
  static getCacheKey(fileSha: string, content: string): SemanticCardCacheKey {
    // Simple hash of string literals and identifiers for cache invalidation
    const stringsHash = this.hashStrings(content);
    return { file_sha: fileSha, strings_hash: stringsHash };
  }

  static validateCard(card: SemanticCard): SemanticCardValidation {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!card.file_id || card.file_id.length === 0) {
      errors.push("file_id is required");
    }

    if (!card.file_sha || card.file_sha.length !== 40) {
      errors.push("file_sha must be 40 character SHA");
    }

    if (!["py", "ts", "js"].includes(card.lang)) {
      errors.push("lang must be py, ts, or js");
    }

    if (card.utilAffinity < 0 || card.utilAffinity > 1) {
      errors.push("utilAffinity must be in range [0,1]");
    }

    if (!Array.isArray(card.e_sem) || card.e_sem.length === 0) {
      errors.push("e_sem embedding vector is required");
    }

    if (typeof card.B !== 'number' || isNaN(card.B)) {
      errors.push("B (businessness score) must be a number");
    }

    // Warnings for potentially problematic cards
    if (card.roles.length === 0) {
      warnings.push("No roles detected - may be utility code");
    }

    if (card.domainTokens.length === 0) {
      warnings.push("No domain tokens extracted - may lack business context");
    }

    if (card.utilAffinity > 0.8) {
      warnings.push("High utility affinity - may be framework/library code");
    }

    return {
      file_id: card.file_id,
      is_valid: errors.length === 0,
      errors,
      warnings
    };
  }

  static computeBusinessnessScore(
    pmiDomain: number,
    resourceCount: number, 
    shapeSpecificity: number,
    hasBusinessRole: boolean,
    utilAffinity: number,
    stats: { pmiMean: number, pmiStd: number, resourceMean: number, resourceStd: number, shapeMean: number, shapeStd: number, utilMean: number, utilStd: number }
  ): number {
    // Z-score normalization
    const zPmi = (pmiDomain - stats.pmiMean) / Math.max(stats.pmiStd, 0.01);
    const zResources = (resourceCount - stats.resourceMean) / Math.max(stats.resourceStd, 0.01);
    const zShapes = (shapeSpecificity - stats.shapeMean) / Math.max(stats.shapeStd, 0.01);
    const zUtil = (utilAffinity - stats.utilMean) / Math.max(stats.utilStd, 0.01);

    const roleBonus = hasBusinessRole ? 1 : 0;
    const utilPenalty = -0.5 * zUtil;

    return zPmi + zResources + zShapes + roleBonus + utilPenalty;
  }

  static isBusinessRole(role: Role): boolean {
    return ["handler", "service", "repo"].includes(role);
  }

  static computeShapeSpecificity(shapes: Shapes): number {
    // More specific type names and JSON keys indicate business logic
    const typeSpecificity = shapes.typeNames.reduce((sum, name) => {
      // Longer, more specific type names get higher scores
      return sum + Math.min(name.length / 10, 2);
    }, 0);

    const keySpecificity = shapes.jsonKeys.reduce((sum, key) => {
      // Domain-specific keys vs generic ones
      const isGeneric = ["id", "name", "type", "data", "value", "config"].includes(key.toLowerCase());
      return sum + (isGeneric ? 0.1 : 1);
    }, 0);

    return typeSpecificity + keySpecificity;
  }

  private static hashStrings(content: string): string {
    // Extract string literals and compute hash
    const stringRegex = /["'`][^"'`]*["'`]/g;
    const strings = content.match(stringRegex) || [];
    const combined = strings.join('|');
    
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return hash.toString(16);
  }
}

export default SemanticCard;