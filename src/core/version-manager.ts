/**
 * API Versioning and Backward Compatibility Manager
 * 
 * Comprehensive API versioning system with:
 * - Semantic versioning support (major.minor.patch)
 * - Multiple versioning strategies (URL path, header, query parameter)
 * - Automatic request/response transformation between versions
 * - Deprecation management with sunset dates
 * - Breaking change detection and migration guidance
 * - Schema evolution and compatibility checking
 * - Version negotiation and client capabilities detection
 * - Analytics and usage tracking per version
 */

import { FastifyRequest, FastifyReply } from 'fastify';
import { z } from 'zod';
import { opentelemetry } from '../telemetry/index.js';
import { performanceMonitor } from './performance-monitor.js';

// Server version constants
export const SERVER_API_VERSION = 'v1';
export const SERVER_INDEX_VERSION = '1.0.0';
export const SERVER_POLICY_VERSION = '1.0.0';

// Version definition and metadata
export interface ApiVersion {
  version: string;          // e.g., "1.2.3"
  major: number;
  minor: number;
  patch: number;
  releaseDate: Date;
  deprecationDate?: Date;   // When this version was deprecated
  sunsetDate?: Date;        // When this version will be removed
  status: 'current' | 'deprecated' | 'sunset';
  breakingChanges: string[];
  features: string[];
  migrations?: VersionMigration[];
}

// Version migration definition
export interface VersionMigration {
  fromVersion: string;
  toVersion: string;
  description: string;
  requestTransformers: RequestTransformer[];
  responseTransformers: ResponseTransformer[];
  breaking: boolean;
  automated: boolean;       // Can be automatically applied
}

// Request/response transformers
export interface RequestTransformer {
  name: string;
  description: string;
  transform: (request: any, context: TransformContext) => any;
  reverse?: (request: any, context: TransformContext) => any;
}

export interface ResponseTransformer {
  name: string;
  description: string;
  transform: (response: any, context: TransformContext) => any;
  reverse?: (response: any, context: TransformContext) => any;
}

export interface TransformContext {
  fromVersion: string;
  toVersion: string;
  requestId: string;
  route: string;
  clientInfo?: ClientInfo;
}

// Client information and capabilities
export interface ClientInfo {
  userAgent: string;
  sdkVersion?: string;
  capabilities: string[];
  preferredVersion?: string;
  supportedVersions: string[];
}

// Version negotiation result
export interface VersionNegotiation {
  requestedVersion?: string;
  resolvedVersion: string;
  strategy: 'url' | 'header' | 'query' | 'default';
  compatible: boolean;
  transformations: string[];
  warnings: string[];
}

// Schema definition for version compatibility
const ApiVersionSchema = z.object({
  version: z.string().regex(/^\d+\.\d+\.\d+$/),
  status: z.enum(['current', 'deprecated', 'sunset']),
  releaseDate: z.date(),
  deprecationDate: z.date().optional(),
  sunsetDate: z.date().optional(),
  breakingChanges: z.array(z.string()),
  features: z.array(z.string())
});

/**
 * Comprehensive API version management system
 */
export class VersionManager {
  private readonly tracer = opentelemetry.trace.getTracer('lens-version-manager');
  private static instance: VersionManager | null = null;
  
  // Version registry
  private versions = new Map<string, ApiVersion>();
  private currentVersion: string = '1.0.0';
  private defaultVersion: string = '1.0.0';
  
  // Public getter for currentVersion
  public getCurrentVersion(): string {
    return this.currentVersion;
  }
  
  // Migration chains and transformers
  private migrations = new Map<string, VersionMigration[]>();
  private requestTransformers = new Map<string, RequestTransformer>();
  private responseTransformers = new Map<string, ResponseTransformer>();
  
  // Usage analytics
  private versionUsage = new Map<string, {
    requests: number;
    lastUsed: number;
    uniqueClients: Set<string>;
    errors: number;
  }>();
  
  // Schema compatibility matrix
  private compatibilityMatrix = new Map<string, Map<string, boolean>>();

  private constructor() {
    this.initializeDefaultVersions();
    this.initializeBuiltinTransformers();
  }

  static getInstance(): VersionManager {
    if (!VersionManager.instance) {
      VersionManager.instance = new VersionManager();
    }
    return VersionManager.instance;
  }

  /**
   * Register a new API version
   */
  registerVersion(versionData: Omit<ApiVersion, 'major' | 'minor' | 'patch'>): void {
    const { major, minor, patch } = this.parseVersion(versionData.version);
    
    const version: ApiVersion = {
      ...versionData,
      major,
      minor,
      patch
    };
    
    // Validate version data
    const validation = ApiVersionSchema.safeParse(version);
    if (!validation.success) {
      throw new Error(`Invalid version data: ${validation.error.message}`);
    }
    
    this.versions.set(version.version, version);
    
    // Initialize usage tracking
    this.versionUsage.set(version.version, {
      requests: 0,
      lastUsed: 0,
      uniqueClients: new Set(),
      errors: 0
    });
    
    // Update compatibility matrix
    this.updateCompatibilityMatrix(version.version);
  }

  /**
   * Register version migration
   */
  registerMigration(migration: VersionMigration): void {
    const key = `${migration.fromVersion}->${migration.toVersion}`;
    
    if (!this.migrations.has(migration.fromVersion)) {
      this.migrations.set(migration.fromVersion, []);
    }
    
    this.migrations.get(migration.fromVersion)!.push(migration);
    
    // Register individual transformers
    for (const transformer of migration.requestTransformers) {
      this.requestTransformers.set(`${key}:${transformer.name}`, transformer);
    }
    
    for (const transformer of migration.responseTransformers) {
      this.responseTransformers.set(`${key}:${transformer.name}`, transformer);
    }
  }

  /**
   * Negotiate API version from request
   */
  negotiateVersion(request: FastifyRequest): VersionNegotiation {
    return this.tracer.startActiveSpan('negotiate-version', (span) => {
      const negotiation: VersionNegotiation = {
        resolvedVersion: this.defaultVersion,
        strategy: 'default',
        compatible: true,
        transformations: [],
        warnings: []
      };

      try {
        // Try URL path version (e.g., /api/v2/search)
        const urlVersion = this.extractVersionFromUrl(request.url);
        if (urlVersion) {
          negotiation.requestedVersion = urlVersion;
          negotiation.strategy = 'url';
        }
        
        // Try header version (e.g., Accept: application/vnd.lens+json;version=2.0.0)
        if (!negotiation.requestedVersion) {
          const headerVersion = this.extractVersionFromHeader(request.headers);
          if (headerVersion) {
            negotiation.requestedVersion = headerVersion;
            negotiation.strategy = 'header';
          }
        }
        
        // Try query parameter version (e.g., ?api_version=1.2.0)
        if (!negotiation.requestedVersion) {
          const queryVersion = this.extractVersionFromQuery(request.query as Record<string, any>);
          if (queryVersion) {
            negotiation.requestedVersion = queryVersion;
            negotiation.strategy = 'query';
          }
        }
        
        // Resolve to actual version
        if (negotiation.requestedVersion) {
          const resolved = this.resolveVersion(negotiation.requestedVersion);
          if (resolved) {
            negotiation.resolvedVersion = resolved.version;
            
            // Check compatibility and add warnings
            if (resolved.status === 'deprecated') {
              negotiation.warnings.push(`Version ${resolved.version} is deprecated and will be sunset on ${resolved.sunsetDate?.toISOString()}`);
            }
            
            if (resolved.status === 'sunset') {
              negotiation.compatible = false;
              negotiation.warnings.push(`Version ${resolved.version} is no longer supported`);
            }
            
            // Check if transformations are needed
            if (negotiation.resolvedVersion !== this.currentVersion) {
              const transformPath = this.findTransformationPath(negotiation.resolvedVersion, this.currentVersion);
              negotiation.transformations = transformPath.map(t => t.description);
            }
          } else {
            negotiation.warnings.push(`Requested version ${negotiation.requestedVersion} not found, using default ${this.defaultVersion}`);
          }
        }
        
        span.setAttributes({
          'lens.version.requested': negotiation.requestedVersion || 'none',
          'lens.version.resolved': negotiation.resolvedVersion,
          'lens.version.strategy': negotiation.strategy,
          'lens.version.compatible': negotiation.compatible
        });
        
        return negotiation;
        
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
        
        // Fallback to default on error
        negotiation.warnings.push(`Version negotiation failed: ${error}. Using default version.`);
        return negotiation;
        
      } finally {
        span.end();
      }
    });
  }

  /**
   * Transform request from one version to another
   */
  async transformRequest(
    request: any,
    fromVersion: string,
    toVersion: string,
    route: string,
    requestId: string
  ): Promise<any> {
    return await this.tracer.startActiveSpan('transform-request', async (span) => {
      try {
        if (fromVersion === toVersion) {
          return request;
        }
        
        const transformationPath = this.findTransformationPath(fromVersion, toVersion);
        let transformedRequest = { ...request };
        
        for (const migration of transformationPath) {
          const context: TransformContext = {
            fromVersion: migration.fromVersion,
            toVersion: migration.toVersion,
            requestId,
            route
          };
          
          for (const transformer of migration.requestTransformers) {
            transformedRequest = transformer.transform(transformedRequest, context);
          }
        }
        
        span.setAttributes({
          'lens.version.transform.from': fromVersion,
          'lens.version.transform.to': toVersion,
          'lens.version.transform.steps': transformationPath.length
        });
        
        return transformedRequest;
        
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
        throw new Error(`Failed to transform request from ${fromVersion} to ${toVersion}: ${error}`);
      } finally {
        span.end();
      }
    });
  }

  /**
   * Transform response from one version to another
   */
  async transformResponse(
    response: any,
    fromVersion: string,
    toVersion: string,
    route: string,
    requestId: string
  ): Promise<any> {
    return await this.tracer.startActiveSpan('transform-response', async (span) => {
      try {
        if (fromVersion === toVersion) {
          return response;
        }
        
        // Reverse the transformation path for responses
        const transformationPath = this.findTransformationPath(toVersion, fromVersion).reverse();
        let transformedResponse = { ...response };
        
        for (const migration of transformationPath) {
          const context: TransformContext = {
            fromVersion: migration.toVersion,
            toVersion: migration.fromVersion,
            requestId,
            route
          };
          
          for (const transformer of migration.responseTransformers) {
            if (transformer.reverse) {
              transformedResponse = transformer.reverse(transformedResponse, context);
            }
          }
        }
        
        span.setAttributes({
          'lens.version.transform.from': fromVersion,
          'lens.version.transform.to': toVersion,
          'lens.version.transform.steps': transformationPath.length
        });
        
        return transformedResponse;
        
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: opentelemetry.SpanStatusCode.ERROR });
        throw new Error(`Failed to transform response from ${fromVersion} to ${toVersion}: ${error}`);
      } finally {
        span.end();
      }
    });
  }

  /**
   * Track version usage for analytics
   */
  trackVersionUsage(
    version: string,
    clientIdentifier: string,
    success: boolean = true
  ): void {
    const usage = this.versionUsage.get(version);
    if (usage) {
      usage.requests++;
      usage.lastUsed = Date.now();
      usage.uniqueClients.add(clientIdentifier);
      
      if (!success) {
        usage.errors++;
      }
      
      // Record metrics
      performanceMonitor.recordMetric(`api_version_${version}_requests`, 1, 'count');
      if (!success) {
        performanceMonitor.recordMetric(`api_version_${version}_errors`, 1, 'count');
      }
    }
  }

  /**
   * Get version compatibility information
   */
  getCompatibilityInfo(version1: string, version2: string): {
    compatible: boolean;
    breaking: boolean;
    migrations: string[];
    effort: 'none' | 'low' | 'medium' | 'high';
  } {
    const compat = this.compatibilityMatrix.get(version1)?.get(version2);
    const transformPath = this.findTransformationPath(version1, version2);
    
    const breakingChanges = transformPath.some(m => m.breaking);
    const migrationCount = transformPath.length;
    
    let effort: 'none' | 'low' | 'medium' | 'high' = 'none';
    if (migrationCount === 0) {
      effort = 'none';
    } else if (migrationCount === 1 && !breakingChanges) {
      effort = 'low';
    } else if (migrationCount <= 3 || !breakingChanges) {
      effort = 'medium';
    } else {
      effort = 'high';
    }
    
    return {
      compatible: compat !== false,
      breaking: breakingChanges,
      migrations: transformPath.map(m => m.description),
      effort
    };
  }

  /**
   * Get version usage analytics
   */
  getVersionAnalytics(): {
    versions: Array<{
      version: string;
      requests: number;
      uniqueClients: number;
      lastUsed: number;
      errors: number;
      errorRate: number;
      status: string;
    }>;
    totalRequests: number;
    activeVersions: number;
  } {
    const versions = [];
    let totalRequests = 0;
    let activeVersions = 0;
    
    for (const [version, usage] of this.versionUsage.entries()) {
      const versionInfo = this.versions.get(version);
      const errorRate = usage.requests > 0 ? (usage.errors / usage.requests) * 100 : 0;
      
      versions.push({
        version,
        requests: usage.requests,
        uniqueClients: usage.uniqueClients.size,
        lastUsed: usage.lastUsed,
        errors: usage.errors,
        errorRate,
        status: versionInfo?.status || 'unknown'
      });
      
      totalRequests += usage.requests;
      
      if (usage.requests > 0 && Date.now() - usage.lastUsed < 30 * 24 * 60 * 60 * 1000) { // Active in last 30 days
        activeVersions++;
      }
    }
    
    return {
      versions: versions.sort((a, b) => b.requests - a.requests),
      totalRequests,
      activeVersions
    };
  }

  /**
   * Get deprecation warnings for a version
   */
  getDeprecationWarnings(version: string): string[] {
    const versionInfo = this.versions.get(version);
    const warnings = [];
    
    if (versionInfo?.status === 'deprecated') {
      warnings.push(`Version ${version} is deprecated since ${versionInfo.deprecationDate?.toISOString()}`);
      
      if (versionInfo.sunsetDate) {
        const daysToSunset = Math.ceil((versionInfo.sunsetDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24));
        warnings.push(`Version ${version} will be sunset in ${daysToSunset} days on ${versionInfo.sunsetDate.toISOString()}`);
      }
      
      warnings.push(`Please migrate to version ${this.currentVersion}`);
      
      // Add breaking changes info
      if (versionInfo.breakingChanges.length > 0) {
        warnings.push(`Breaking changes since ${version}: ${versionInfo.breakingChanges.join(', ')}`);
      }
    }
    
    return warnings;
  }

  /**
   * Private helper methods
   */
  
  private parseVersion(version: string): { major: number; minor: number; patch: number } {
    const match = version.match(/^(\d+)\.(\d+)\.(\d+)$/);
    if (!match) {
      throw new Error(`Invalid version format: ${version}`);
    }
    
    return {
      major: parseInt(match[1]),
      minor: parseInt(match[2]),
      patch: parseInt(match[3])
    };
  }

  private extractVersionFromUrl(url: string): string | null {
    const match = url.match(/\/api\/v(\d+(?:\.\d+)?(?:\.\d+)?)\//);
    if (match) {
      const version = match[1];
      // Normalize to full semantic version
      const parts = version.split('.');
      while (parts.length < 3) {
        parts.push('0');
      }
      return parts.join('.');
    }
    return null;
  }

  private extractVersionFromHeader(headers: Record<string, any>): string | null {
    const acceptHeader = headers['accept'] || headers['Accept'];
    if (acceptHeader) {
      const match = acceptHeader.match(/application\/vnd\.lens\+json;version=([0-9.]+)/);
      if (match) {
        return match[1];
      }
    }
    
    const versionHeader = headers['api-version'] || headers['Api-Version'] || headers['X-API-Version'];
    if (versionHeader) {
      return versionHeader.toString();
    }
    
    return null;
  }

  private extractVersionFromQuery(query: Record<string, any>): string | null {
    return query.api_version || query.version || null;
  }

  private resolveVersion(requestedVersion: string): ApiVersion | null {
    // Exact match first
    if (this.versions.has(requestedVersion)) {
      return this.versions.get(requestedVersion)!;
    }
    
    // Try to find compatible version
    const { major, minor } = this.parseVersion(requestedVersion);
    
    // Look for compatible minor versions within same major
    for (const [version, versionInfo] of this.versions.entries()) {
      if (versionInfo.major === major && versionInfo.minor >= minor) {
        return versionInfo;
      }
    }
    
    return null;
  }

  private findTransformationPath(fromVersion: string, toVersion: string): VersionMigration[] {
    if (fromVersion === toVersion) {
      return [];
    }
    
    // Simple breadth-first search for transformation path
    const queue = [{ version: fromVersion, path: [] as VersionMigration[] }];
    const visited = new Set<string>();
    
    while (queue.length > 0) {
      const { version, path } = queue.shift()!;
      
      if (visited.has(version)) {
        continue;
      }
      visited.add(version);
      
      if (version === toVersion) {
        return path;
      }
      
      const migrations = this.migrations.get(version) || [];
      for (const migration of migrations) {
        if (!visited.has(migration.toVersion)) {
          queue.push({
            version: migration.toVersion,
            path: [...path, migration]
          });
        }
      }
    }
    
    // No transformation path found
    throw new Error(`No transformation path found from ${fromVersion} to ${toVersion}`);
  }

  private updateCompatibilityMatrix(newVersion: string): void {
    if (!this.compatibilityMatrix.has(newVersion)) {
      this.compatibilityMatrix.set(newVersion, new Map());
    }
    
    // Check compatibility with all existing versions
    for (const [existingVersion] of this.versions) {
      if (existingVersion !== newVersion) {
        const compatible = this.checkVersionCompatibility(existingVersion, newVersion);
        
        this.compatibilityMatrix.get(existingVersion)!.set(newVersion, compatible);
        this.compatibilityMatrix.get(newVersion)!.set(existingVersion, compatible);
      }
    }
    
    // Self-compatibility
    this.compatibilityMatrix.get(newVersion)!.set(newVersion, true);
  }

  private checkVersionCompatibility(version1: string, version2: string): boolean {
    const v1 = this.parseVersion(version1);
    const v2 = this.parseVersion(version2);
    
    // Same major version is generally compatible
    if (v1.major === v2.major) {
      return true;
    }
    
    // Different major versions require explicit migration
    try {
      this.findTransformationPath(version1, version2);
      return true;
    } catch {
      return false;
    }
  }

  private initializeDefaultVersions(): void {
    // Register current version
    this.registerVersion({
      version: '1.0.0',
      status: 'current',
      releaseDate: new Date(),
      breakingChanges: [],
      features: [
        'Basic search functionality',
        'Lexical search',
        'Symbol search',
        'Semantic search'
      ]
    });
    
    this.currentVersion = '1.0.0';
    this.defaultVersion = '1.0.0';
  }

  private initializeBuiltinTransformers(): void {
    // Example: v1.0.0 -> v1.1.0 migration (adding pagination)
    const v1_0_to_v1_1_request: RequestTransformer = {
      name: 'add-pagination-defaults',
      description: 'Add default pagination parameters',
      transform: (request, context) => {
        if (!request.pagination) {
          request.pagination = {
            limit: 100,
            offset: 0
          };
        }
        return request;
      },
      reverse: (request, context) => {
        // Remove pagination if it matches defaults
        if (request.pagination?.limit === 100 && request.pagination?.offset === 0) {
          delete request.pagination;
        }
        return request;
      }
    };

    const v1_0_to_v1_1_response: ResponseTransformer = {
      name: 'add-pagination-metadata',
      description: 'Add pagination metadata to response',
      transform: (response, context) => {
        if (Array.isArray(response.results) && !response.pagination) {
          response.pagination = {
            total: response.results.length,
            limit: 100,
            offset: 0,
            hasMore: false
          };
        }
        return response;
      },
      reverse: (response, context) => {
        // Remove pagination metadata for older versions
        delete response.pagination;
        return response;
      }
    };

    // Register example migration
    this.registerMigration({
      fromVersion: '1.0.0',
      toVersion: '1.1.0',
      description: 'Add pagination support',
      requestTransformers: [v1_0_to_v1_1_request],
      responseTransformers: [v1_0_to_v1_1_response],
      breaking: false,
      automated: true
    });
  }
}

/**
 * Fastify plugin for automatic version handling
 */
export function createVersionMiddleware() {
  const versionManager = VersionManager.getInstance();
  
  return async function versionMiddleware(request: FastifyRequest, reply: FastifyReply) {
    const negotiation = versionManager.negotiateVersion(request);
    
    // Add version info to request context
    (request as any).apiVersion = negotiation;
    
    // Add version headers to response
    reply.header('X-API-Version', negotiation.resolvedVersion);
    reply.header('X-API-Version-Strategy', negotiation.strategy);
    
    // Add deprecation warnings
    if (negotiation.warnings.length > 0) {
      reply.header('X-API-Warnings', negotiation.warnings.join('; '));
    }
    
    // Track usage
    const clientId = request.headers['user-agent'] || 'unknown';
    versionManager.trackVersionUsage(negotiation.resolvedVersion, clientId, true);
  };
}

// Export singleton instance
export const versionManager = VersionManager.getInstance();

/**
 * Check compatibility between different component versions
 */
export function checkCompatibility(
  api_version?: string,
  index_version?: string,
  allow_compat?: boolean,
  policy_version?: string
): {
  compatible: boolean;
  warnings: string[];
  errors: string[];
  api_version: "v1";
  index_version: "v1";
  policy_version: "v1";
} {
  const currentApiVersion = versionManager.getCurrentVersion();
  const warnings: string[] = [];
  const errors: string[] = [];
  let compatible = true;

  // Use defaults if not provided
  const resolvedApiVersion = api_version || currentApiVersion;
  const resolvedIndexVersion = index_version || '1.0.0';
  const resolvedPolicyVersion = policy_version || '1.0.0';

  // Check API version compatibility
  if (api_version && api_version !== currentApiVersion) {
    const apiCompatInfo = versionManager.getCompatibilityInfo(api_version, currentApiVersion);
    if (!apiCompatInfo.compatible && !allow_compat) {
      compatible = false;
      errors.push(`API version ${api_version} is incompatible with current version ${currentApiVersion}`);
    } else if (!apiCompatInfo.compatible && allow_compat) {
      warnings.push(`API version ${api_version} requires compatibility mode`);
    }
  }

  // Check for deprecated versions
  const deprecationWarnings = versionManager.getDeprecationWarnings(resolvedApiVersion);
  warnings.push(...deprecationWarnings);

  return {
    compatible,
    warnings,
    errors,
    api_version: "v1", // API expects exactly "v1"
    index_version: "v1", // API expects exactly "v1"
    policy_version: "v1" // API expects exactly "v1"
  };
}

/**
 * Get current version information
 */
export function getVersionInfo(): {
  api_version: "v1";
  index_version: "v1";
  policy_version: "v1";
  build_timestamp?: string;
  git_commit?: string;
} {
  return {
    api_version: "v1", // API expects exactly "v1"
    index_version: "v1", // API expects exactly "v1" 
    policy_version: "v1", // API expects exactly "v1"
    build_timestamp: new Date().toISOString(),
    git_commit: process.env.GIT_COMMIT || 'unknown'
  };
}