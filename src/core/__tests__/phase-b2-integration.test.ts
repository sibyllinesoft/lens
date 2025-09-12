/**
 * Phase B2 Integration Tests
 * End-to-end testing of all Phase B2 optimizations working together
 * Validates 40% Stage-B performance improvement target
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest } from 'bun:test';
import { EnhancedSymbolSearchEngine } from '../../indexer/enhanced-symbols.js';
import { OptimizedASTCache, PERFORMANCE_PRESETS } from '../optimized-ast-cache.js';
import { StructuralPatternEngine, PATTERN_PRESETS } from '../structural-pattern-engine.js';
import { CoverageTracker } from '../coverage-tracker.js';
import { FeatureFlagManager, PhaseB2Flags } from '../feature-flags.js';
import { SegmentStorage } from '../../storage/segments.js';

describe('Phase B2 Integration', () => {
  let segmentStorage: SegmentStorage;
  let featureFlags: FeatureFlagManager;
  
  // Test data representing real-world TypeScript files
  const realWorldFiles = {
    'react-component.tsx': `
import React, { useState, useEffect, useMemo } from 'react';
import { DataProcessor, ProcessorConfig } from './data-processor';
import { ApiClient } from '../api/client';
import type { User, UserProfile, ApiResponse } from '../types';

interface UserListProps {
  users: User[];
  onUserSelect: (user: User) => void;
  loading?: boolean;
  error?: string;
}

export const UserList: React.FC<UserListProps> = ({
  users,
  onUserSelect,
  loading = false,
  error
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedUsers, setSelectedUsers] = useState<User[]>([]);
  
  const filteredUsers = useMemo(() => {
    return users.filter(user => 
      user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [users, searchTerm]);
  
  const handleUserClick = async (user: User) => {
    try {
      const profile = await ApiClient.getUserProfile(user.id);
      onUserSelect({ ...user, profile });
      setSelectedUsers(prev => [...prev, user]);
    } catch (err) {
      console.error('Failed to load user profile:', err);
    }
  };
  
  useEffect(() => {
    const processor = new DataProcessor({
      transformer: (user: User) => ({ ...user, processed: true }),
      batchSize: 50
    });
    
    if (users.length > 0) {
      processor.processItems(users);
    }
  }, [users]);
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return (
    <div className="user-list">
      <input
        type="text"
        placeholder="Search users..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
      />
      <div className="user-grid">
        {filteredUsers.map(user => (
          <UserCard
            key={user.id}
            user={user}
            onClick={() => handleUserClick(user)}
            selected={selectedUsers.includes(user)}
          />
        ))}
      </div>
    </div>
  );
};

interface UserCardProps {
  user: User;
  onClick: () => void;
  selected: boolean;
}

const UserCard: React.FC<UserCardProps> = ({ user, onClick, selected }) => {
  return (
    <div 
      className={\`user-card \${selected ? 'selected' : ''}\`}
      onClick={onClick}
    >
      <img src={user.avatar} alt={user.name} />
      <h3>{user.name}</h3>
      <p>{user.email}</p>
      <span className="role">{user.role}</span>
    </div>
  );
};

export default UserList;
    `,

    'api-service.ts': `
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { EventEmitter } from 'events';
import { Logger } from '../utils/logger';
import { retry, RetryConfig } from '../utils/retry';
import { cache, CacheConfig } from '../utils/cache';

export interface ApiConfig {
  baseURL: string;
  timeout: number;
  retryConfig: RetryConfig;
  cacheConfig?: CacheConfig;
  authToken?: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
  retryable: boolean;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    hasNext: boolean;
  };
}

export class ApiService extends EventEmitter {
  private client: AxiosInstance;
  private logger: Logger;
  private retryConfig: RetryConfig;
  
  constructor(private config: ApiConfig) {
    super();
    
    this.logger = new Logger('ApiService');
    this.retryConfig = config.retryConfig;
    
    this.client = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(config.authToken && { 'Authorization': \`Bearer \${config.authToken}\` })
      }
    });
    
    this.setupInterceptors();
  }
  
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.request<T>('GET', url, undefined, config);
  }
  
  async post<T, D = any>(url: string, data?: D, config?: AxiosRequestConfig): Promise<T> {
    return this.request<T>('POST', url, data, config);
  }
  
  async put<T, D = any>(url: string, data?: D, config?: AxiosRequestConfig): Promise<T> {
    return this.request<T>('PUT', url, data, config);
  }
  
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.request<T>('DELETE', url, undefined, config);
  }
  
  async paginate<T>(
    url: string,
    options: { page?: number; limit?: number } = {}
  ): Promise<PaginatedResponse<T>> {
    const { page = 1, limit = 20 } = options;
    const response = await this.get<PaginatedResponse<T>>(
      \`\${url}?page=\${page}&limit=\${limit}\`
    );
    return response;
  }
  
  private async request<T>(
    method: string,
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    const cacheKey = this.getCacheKey(method, url, data);
    
    // Try cache first for GET requests
    if (method === 'GET' && this.config.cacheConfig) {
      const cached = await cache.get<T>(cacheKey);
      if (cached) {
        this.logger.debug(\`Cache hit for \${method} \${url}\`);
        return cached;
      }
    }
    
    try {
      const response = await retry(
        () => this.client.request<T>({
          method,
          url,
          data,
          ...config
        }),
        this.retryConfig
      );
      
      const result = response.data;
      
      // Cache successful GET responses
      if (method === 'GET' && this.config.cacheConfig) {
        await cache.set(cacheKey, result, this.config.cacheConfig);
      }
      
      this.emit('request', { method, url, status: response.status });
      return result;
      
    } catch (error) {
      const apiError = this.transformError(error);
      this.emit('error', apiError);
      this.logger.error(\`API request failed: \${method} \${url}\`, apiError);
      throw apiError;
    }
  }
  
  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        this.logger.debug(\`Making request: \${config.method?.toUpperCase()} \${config.url}\`);
        return config;
      },
      (error) => {
        this.logger.error('Request interceptor error:', error);
        return Promise.reject(error);
      }
    );
    
    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        this.logger.debug(\`Response received: \${response.status} \${response.config.url}\`);
        return response;
      },
      (error) => {
        this.logger.error('Response interceptor error:', error);
        return Promise.reject(error);
      }
    );
  }
  
  private getCacheKey(method: string, url: string, data?: any): string {
    const dataHash = data ? JSON.stringify(data) : '';
    return \`\${method}:\${url}:\${dataHash}\`;
  }
  
  private transformError(error: any): ApiError {
    if (error.response) {
      // Server responded with error status
      return {
        code: \`HTTP_\${error.response.status}\`,
        message: error.response.data?.message || error.message,
        details: error.response.data,
        retryable: error.response.status >= 500
      };
    } else if (error.request) {
      // Request was made but no response received
      return {
        code: 'NETWORK_ERROR',
        message: 'Network error - no response received',
        details: error.request,
        retryable: true
      };
    } else {
      // Something else happened
      return {
        code: 'UNKNOWN_ERROR',
        message: error.message || 'Unknown error occurred',
        details: error,
        retryable: false
      };
    }
  }
}

export const createApiService = (config: ApiConfig): ApiService => {
  return new ApiService(config);
};

export default ApiService;
    `,

    'data-utils.ts': `
import { z } from 'zod';
import { EventEmitter } from 'events';

// Validation schemas
export const UserSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1).max(100),
  email: z.string().email(),
  role: z.enum(['admin', 'user', 'moderator']),
  avatar: z.string().url().optional(),
  createdAt: z.date(),
  updatedAt: z.date(),
  profile: z.object({
    bio: z.string().optional(),
    location: z.string().optional(),
    website: z.string().url().optional(),
    preferences: z.record(z.any()).optional()
  }).optional()
});

export type User = z.infer<typeof UserSchema>;

// Data processing utilities
export class DataValidator {
  static validate<T>(schema: z.ZodSchema<T>, data: unknown): T {
    try {
      return schema.parse(data);
    } catch (error) {
      if (error instanceof z.ZodError) {
        throw new ValidationError('Data validation failed', error.errors);
      }
      throw error;
    }
  }
  
  static validateArray<T>(schema: z.ZodSchema<T>, data: unknown[]): T[] {
    return data.map((item, index) => {
      try {
        return schema.parse(item);
      } catch (error) {
        if (error instanceof z.ZodError) {
          throw new ValidationError(\`Validation failed at index \${index}\`, error.errors);
        }
        throw error;
      }
    });
  }
}

export class ValidationError extends Error {
  constructor(message: string, public errors: z.ZodError['errors']) {
    super(message);
    this.name = 'ValidationError';
  }
}

// Data transformation utilities
export class DataTransformer {
  static mapUsers(users: any[]): User[] {
    return DataValidator.validateArray(UserSchema, users);
  }
  
  static filterByRole(users: User[], role: User['role']): User[] {
    return users.filter(user => user.role === role);
  }
  
  static sortByName(users: User[]): User[] {
    return [...users].sort((a, b) => a.name.localeCompare(b.name));
  }
  
  static groupByRole(users: User[]): Record<User['role'], User[]> {
    return users.reduce((groups, user) => {
      const role = user.role;
      groups[role] = groups[role] || [];
      groups[role].push(user);
      return groups;
    }, {} as Record<User['role'], User[]>);
  }
  
  static async transformAsync<T, U>(
    items: T[],
    transformer: (item: T) => Promise<U>,
    concurrency = 5
  ): Promise<U[]> {
    const results: U[] = [];
    
    for (let i = 0; i < items.length; i += concurrency) {
      const chunk = items.slice(i, i + concurrency);
      const chunkResults = await Promise.all(
        chunk.map(transformer)
      );
      results.push(...chunkResults);
    }
    
    return results;
  }
}

// Data aggregation utilities  
export class DataAggregator extends EventEmitter {
  private cache = new Map<string, any>();
  
  async aggregate<T, R>(
    data: T[],
    aggregator: (items: T[]) => R,
    cacheKey?: string
  ): Promise<R> {
    if (cacheKey && this.cache.has(cacheKey)) {
      this.emit('cache-hit', cacheKey);
      return this.cache.get(cacheKey);
    }
    
    const result = aggregator(data);
    
    if (cacheKey) {
      this.cache.set(cacheKey, result);
      this.emit('cache-set', cacheKey);
    }
    
    return result;
  }
  
  async aggregateUsers(users: User[]): Promise<{
    total: number;
    byRole: Record<User['role'], number>;
    averageNameLength: number;
    recentUsers: User[];
  }> {
    return this.aggregate(
      users,
      (users) => ({
        total: users.length,
        byRole: users.reduce((counts, user) => {
          counts[user.role] = (counts[user.role] || 0) + 1;
          return counts;
        }, {} as Record<User['role'], number>),
        averageNameLength: users.reduce((sum, user) => sum + user.name.length, 0) / users.length,
        recentUsers: users
          .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
          .slice(0, 10)
      }),
      'user-aggregation'
    );
  }
  
  clearCache(): void {
    this.cache.clear();
    this.emit('cache-cleared');
  }
}

// Export utility functions
export const createUser = (data: Partial<User>): User => {
  const now = new Date();
  return DataValidator.validate(UserSchema, {
    id: data.id || crypto.randomUUID(),
    name: data.name || 'Unknown User',
    email: data.email || 'user@example.com',
    role: data.role || 'user',
    createdAt: data.createdAt || now,
    updatedAt: data.updatedAt || now,
    ...data
  });
};

export const isValidUser = (data: unknown): data is User => {
  try {
    UserSchema.parse(data);
    return true;
  } catch {
    return false;
  }
};

export const userToString = (user: User): string => {
  return \`\${user.name} <\${user.email}> (\${user.role})\`;
};

export default {
  DataValidator,
  DataTransformer,
  DataAggregator,
  ValidationError,
  createUser,
  isValidUser,
  userToString
};
    `
  };

  beforeEach(() => {
    // Mock segment storage
    segmentStorage = {
      listSegments: jest.fn().mockReturnValue([]),
      openSegment: jest.fn(),
      readFromSegment: jest.fn(),
    } as any;

    // Initialize feature flags for testing
    featureFlags = new FeatureFlagManager({
      enableABTesting: true,
      enablePerformanceTracking: true,
    });
  });

  afterEach(() => {
    featureFlags.shutdown();
  });

  describe('Complete Phase B2 Workflow', () => {
    it('should demonstrate end-to-end 40% performance improvement', async () => {
      // Test configuration
      const baselineTargetMs = 7;
      const optimizedTargetMs = 4;
      const expectedImprovementPercent = 40;

      console.log('🚀 Starting Phase B2 Complete Workflow Test');
      console.log(`Target: ${expectedImprovementPercent}% improvement (${baselineTargetMs}ms → ${optimizedTargetMs}ms)`);

      // Initialize enhanced engine with all optimizations enabled
      const enhancedEngine = new EnhancedSymbolSearchEngine(segmentStorage, {
        cacheConfig: PERFORMANCE_PRESETS.performance,
        enableStructuralPatterns: true,
        enableCoverageTracking: true,
        batchProcessingEnabled: true,
        preloadHotFiles: false, // Disable for consistent testing
        stageBTargetMs: optimizedTargetMs,
      });

      await enhancedEngine.initialize();

      try {
        // Phase 1: Batch indexing test
        console.log('\n📊 Phase 1: Batch Indexing Performance');
        
        const indexingFiles = Object.entries(realWorldFiles).map(([filename, content]) => ({
          filePath: `/src/${filename}`,
          content,
          language: filename.endsWith('.tsx') || filename.endsWith('.ts') ? 'typescript' as const : 'javascript' as const
        }));

        const indexingStart = Date.now();
        await enhancedEngine.batchIndexFiles(indexingFiles);
        const totalIndexingTime = Date.now() - indexingStart;
        const avgIndexingTime = totalIndexingTime / indexingFiles.length;

        console.log(`  • Total files: ${indexingFiles.length}`);
        console.log(`  • Total time: ${totalIndexingTime}ms`);
        console.log(`  • Average per file: ${avgIndexingTime.toFixed(2)}ms`);
        console.log(`  • Target per file: <${optimizedTargetMs}ms`);

        expect(avgIndexingTime).toBeLessThan(optimizedTargetMs);

        // Phase 2: Search performance test
        console.log('\n📊 Phase 2: Search Performance');
        
        const searchQueries = [
          'UserList',
          'ApiService',
          'DataValidator',
          'interface',
          'async function',
          'useState',
          'EventEmitter'
        ];

        const searchContext = { workspace_root: '/src', language: 'typescript' as const };
        const searchTimes: number[] = [];

        for (const query of searchQueries) {
          const searchStart = Date.now();
          const results = await enhancedEngine.searchSymbols(query, searchContext, 20);
          const searchTime = Date.now() - searchStart;
          searchTimes.push(searchTime);

          console.log(`  • "${query}": ${searchTime}ms (${results.length} results)`);
        }

        const avgSearchTime = searchTimes.reduce((sum, t) => sum + t, 0) / searchTimes.length;
        console.log(`  • Average search time: ${avgSearchTime.toFixed(2)}ms`);

        expect(avgSearchTime).toBeLessThan(optimizedTargetMs);

        // Phase 3: Combined Stage-B performance
        console.log('\n📊 Phase 3: Combined Stage-B Performance');
        
        const combinedStageB = (avgIndexingTime + avgSearchTime) / 2;
        const actualImprovement = ((baselineTargetMs - combinedStageB) / baselineTargetMs) * 100;

        console.log(`  • Combined Stage-B: ${combinedStageB.toFixed(2)}ms`);
        console.log(`  • Baseline target: ${baselineTargetMs}ms`);
        console.log(`  • Optimized target: ${optimizedTargetMs}ms`);
        console.log(`  • Actual improvement: ${actualImprovement.toFixed(1)}%`);
        console.log(`  • Target improvement: ${expectedImprovementPercent}%`);

        expect(combinedStageB).toBeLessThan(optimizedTargetMs);
        expect(actualImprovement).toBeGreaterThan(expectedImprovementPercent);

        // Phase 4: Verify all components are working
        console.log('\n📊 Phase 4: Component Integration Verification');
        
        const metrics = enhancedEngine.getEnhancedMetrics();
        const coverageReport = enhancedEngine.getCoverageReport();

        console.log(`  • Cache hit rate: ${metrics.cache.hitRate}%`);
        console.log(`  • Cache size: ${metrics.cache.cacheSize} files`);
        console.log(`  • Pattern engines: ${metrics.patterns.length} patterns`);
        console.log(`  • Coverage: ${coverageReport.metrics.coveragePercentage}% (${coverageReport.metrics.indexedFiles}/${coverageReport.metrics.totalFiles})`);
        console.log(`  • Symbols processed: ${metrics.performance.symbolsProcessed}`);

        // Verify each component is contributing
        expect(metrics.cache.cacheSize).toBeGreaterThan(0);
        expect(metrics.patterns.length).toBeGreaterThan(0);
        expect(coverageReport.metrics.coveragePercentage).toBe(100);
        expect(metrics.performance.symbolsProcessed).toBeGreaterThan(0);

        console.log('\n✅ Phase B2 Complete Workflow: All targets achieved!');

      } finally {
        await enhancedEngine.shutdown();
      }
    });

    it('should demonstrate scalability improvements', async () => {
      console.log('📈 Testing Phase B2 Scalability Improvements');

      const engine = new EnhancedSymbolSearchEngine(segmentStorage, {
        cacheConfig: PERFORMANCE_PRESETS.performance,
        enableStructuralPatterns: true,
        enableCoverageTracking: true,
        batchProcessingEnabled: true,
        stageBTargetMs: 4,
      });

      await engine.initialize();

      try {
        // Generate larger dataset
        const scaleTestFiles = [];
        for (let i = 0; i < 20; i++) {
          for (const [filename, content] of Object.entries(realWorldFiles)) {
            scaleTestFiles.push({
              filePath: `/scale/batch${i}/${filename}`,
              content: content + `\n// Variation ${i}`,
              language: 'typescript' as const
            });
          }
        }

        console.log(`  • Testing with ${scaleTestFiles.length} files`);

        // Test batch processing scalability
        const batchStart = Date.now();
        await engine.batchIndexFiles(scaleTestFiles);
        const batchTime = Date.now() - batchStart;

        const throughput = (scaleTestFiles.length / batchTime) * 1000; // files/second
        const avgTime = batchTime / scaleTestFiles.length;

        console.log(`  • Batch time: ${batchTime}ms`);
        console.log(`  • Average per file: ${avgTime.toFixed(2)}ms`);
        console.log(`  • Throughput: ${throughput.toFixed(1)} files/second`);

        // Scalability targets
        expect(avgTime).toBeLessThan(6); // Slight degradation acceptable at scale
        expect(throughput).toBeGreaterThan(15); // Should maintain good throughput

        // Test search performance at scale
        const largeScaleSearches = Array.from({ length: 50 }, (_, i) => `search${i % 10}`);
        const searchStart = Date.now();

        const searchPromises = largeScaleSearches.map(query =>
          engine.searchSymbols(query, { workspace_root: '/scale', language: 'typescript' })
        );

        await Promise.all(searchPromises);
        const totalSearchTime = Date.now() - searchStart;
        const avgSearchTime = totalSearchTime / largeScaleSearches.length;

        console.log(`  • ${largeScaleSearches.length} searches: ${totalSearchTime}ms`);
        console.log(`  • Average search: ${avgSearchTime.toFixed(2)}ms`);

        expect(avgSearchTime).toBeLessThan(8); // Allow some degradation for concurrent searches

      } finally {
        await engine.shutdown();
      }
    });
  });

  describe('Feature Flag Integration', () => {
    it('should enable/disable Phase B2 features correctly', async () => {
      console.log('🚦 Testing Feature Flag Integration');

      // Test with flags disabled
      featureFlags.setFlag({
        key: 'enhanced_ast_cache',
        name: 'Enhanced AST Cache',
        description: 'Test flag',
        enabled: false,
        rolloutPercentage: 0
      });

      const context = { userId: 'test-user', language: 'typescript' };
      
      expect(PhaseB2Flags.enhancedAstCache(context)).toBe(false);

      // Test with gradual rollout
      featureFlags.updateRollout('enhanced_ast_cache', 50);
      featureFlags.setFlag({
        key: 'enhanced_ast_cache',
        name: 'Enhanced AST Cache',
        description: 'Test flag',
        enabled: true,
        rolloutPercentage: 50
      });

      // Test multiple users - should get consistent results
      const user1Result = PhaseB2Flags.enhancedAstCache({ userId: 'user1', language: 'typescript' });
      const user1Again = PhaseB2Flags.enhancedAstCache({ userId: 'user1', language: 'typescript' });
      expect(user1Result).toBe(user1Again); // Consistent

      console.log('  • Feature flags working correctly');
    });

    it('should track A/B test performance', async () => {
      // Enable A/B testing for enhanced symbol search
      featureFlags.setFlag({
        key: 'test_optimization',
        name: 'Test Optimization',
        description: 'A/B test flag',
        enabled: true,
        rolloutPercentage: 50
      });

      const performanceData = [];

      // Simulate performance data collection
      for (let i = 0; i < 200; i++) {
        const userId = `user${i}`;
        const context = { userId, language: 'typescript' };
        const isInTreatment = featureFlags.isEnabled('test_optimization', context);
        
        // Simulate different performance for control vs treatment
        const latency = isInTreatment ? 3 : 7; // Treatment is faster
        const success = Math.random() > 0.05; // 95% success rate
        
        featureFlags.recordPerformance('test_optimization', context, latency, success);
        performanceData.push({ isInTreatment, latency, success });
      }

      // Analyze A/B test results
      const abResult = featureFlags.analyzeABTest('test_optimization');
      
      expect(abResult).not.toBeNull();
      if (abResult) {
        console.log('  • A/B Test Results:');
        console.log(`    - Control avg latency: ${abResult.controlGroup.avgLatency.toFixed(2)}ms`);
        console.log(`    - Treatment avg latency: ${abResult.treatmentGroup.avgLatency.toFixed(2)}ms`);
        console.log(`    - Statistical significance: ${(abResult.statisticalSignificance * 100).toFixed(1)}%`);
        console.log(`    - Recommendation: ${abResult.recommendation}`);

        expect(abResult.treatmentGroup.avgLatency).toBeLessThan(abResult.controlGroup.avgLatency);
        expect(abResult.recommendation).toBe('rollout');
      }
    });
  });

  describe('Memory and Resource Management', () => {
    it('should maintain reasonable memory usage at scale', async () => {
      console.log('💾 Testing Memory Management at Scale');

      const engine = new EnhancedSymbolSearchEngine(segmentStorage, {
        cacheConfig: PERFORMANCE_PRESETS.balanced, // Use balanced preset
        enableStructuralPatterns: true,
        enableCoverageTracking: true,
        batchProcessingEnabled: true,
      });

      await engine.initialize();

      try {
        // Simulate large-scale usage
        const memoryTestFiles = [];
        for (let batch = 0; batch < 5; batch++) {
          for (const [filename, content] of Object.entries(realWorldFiles)) {
            memoryTestFiles.push({
              filePath: `/memory/batch${batch}/${filename}`,
              content: content.repeat(2), // Double the size
              language: 'typescript' as const
            });
          }
        }

        await engine.batchIndexFiles(memoryTestFiles);

        const metrics = engine.getEnhancedMetrics();
        
        console.log(`  • Cache size: ${metrics.cache.cacheSize} files`);
        console.log(`  • Memory usage: ${metrics.cache.memoryUsage}MB`);
        console.log(`  • Symbols processed: ${metrics.performance.symbolsProcessed}`);

        // Memory should be bounded
        expect(metrics.cache.memoryUsage).toBeLessThan(100); // <100MB
        expect(metrics.cache.cacheSize).toBeLessThanOrEqual(200); // Respects limits

        // Perform many searches to test memory stability
        for (let i = 0; i < 50; i++) {
          await engine.searchSymbols(`search${i}`, { workspace_root: '/memory', language: 'typescript' });
        }

        const afterMetrics = engine.getEnhancedMetrics();
        
        // Memory shouldn't have grown significantly
        expect(afterMetrics.cache.memoryUsage).toBeLessThan(metrics.cache.memoryUsage * 1.2);

      } finally {
        await engine.shutdown();
      }
    });

    it('should properly cleanup resources on shutdown', async () => {
      const cache = new OptimizedASTCache(PERFORMANCE_PRESETS.balanced);
      const patternEngine = new StructuralPatternEngine();
      const coverageTracker = new CoverageTracker();

      // Use the components
      await cache.getAST('/test/file.ts', realWorldFiles['data-utils.ts'], 'typescript');
      await patternEngine.executePattern('ts-function-declarations', realWorldFiles['api-service.ts'], 'typescript');
      coverageTracker.recordFileIndexing('/test/file.ts', 'typescript', [], 10);

      // Verify they're working
      expect(cache.getMetrics().cacheSize).toBeGreaterThan(0);
      expect(patternEngine.getPatternCount()).toBeGreaterThan(0);
      expect(coverageTracker.getCurrentMetrics().indexedFiles).toBeGreaterThan(0);

      // Shutdown and verify cleanup
      await cache.shutdown();
      patternEngine.clear();
      coverageTracker.clear();

      expect(cache.getMetrics().cacheSize).toBe(0);
      expect(coverageTracker.getCurrentMetrics().indexedFiles).toBe(0);

      console.log('  • Resource cleanup verified');
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('should gracefully handle component failures', async () => {
      console.log('🛡️ Testing Error Recovery and Resilience');

      const engine = new EnhancedSymbolSearchEngine(segmentStorage, {
        enableStructuralPatterns: true,
        enableCoverageTracking: true,
        batchProcessingEnabled: true,
      });

      await engine.initialize();

      try {
        // Test with problematic content
        const problematicFiles = [
          { filePath: '/error/valid.ts', content: realWorldFiles['data-utils.ts'], language: 'typescript' as const },
          { filePath: '/error/malformed.ts', content: '{ invalid syntax }[', language: 'typescript' as const },
          { filePath: '/error/empty.ts', content: '', language: 'typescript' as const },
          { filePath: '/error/large.ts', content: 'x'.repeat(100000), language: 'typescript' as const },
        ];

        // Should not throw errors
        await expect(engine.batchIndexFiles(problematicFiles)).resolves.not.toThrow();

        // Should still be able to search
        const results = await engine.searchSymbols('DataValidator', { workspace_root: '/error', language: 'typescript' });
        expect(results.length).toBeGreaterThan(0); // Should find results from valid files

        const coverageReport = engine.getCoverageReport();
        expect(coverageReport.metrics.indexedFiles).toBeGreaterThan(0);

        // Check for coverage gaps
        if (coverageReport.gaps.length > 0) {
          console.log(`  • Found ${coverageReport.gaps.length} coverage gaps (expected with malformed files)`);
        }

      } finally {
        await engine.shutdown();
      }
    });

    it('should maintain performance under error conditions', async () => {
      const engine = new EnhancedSymbolSearchEngine(segmentStorage);
      await engine.initialize();

      try {
        // Mix of valid and invalid files
        const mixedFiles = [
          { filePath: '/mixed/good1.ts', content: realWorldFiles['api-service.ts'], language: 'typescript' as const },
          { filePath: '/mixed/bad1.ts', content: '{ broken', language: 'typescript' as const },
          { filePath: '/mixed/good2.ts', content: realWorldFiles['react-component.tsx'], language: 'typescript' as const },
          { filePath: '/mixed/bad2.ts', content: '', language: 'typescript' as const },
        ];

        const start = Date.now();
        await engine.batchIndexFiles(mixedFiles);
        const time = Date.now() - start;

        // Should complete in reasonable time despite errors
        expect(time).toBeLessThan(100);

        // Should still provide useful metrics
        const metrics = engine.getEnhancedMetrics();
        expect(metrics.performance.symbolsProcessed).toBeGreaterThan(0);

      } finally {
        await engine.shutdown();
      }
    });
  });

  describe('Real-World Usage Simulation', () => {
    it('should handle typical development workflow', async () => {
      console.log('👨‍💻 Simulating Real-World Development Workflow');

      const engine = new EnhancedSymbolSearchEngine(segmentStorage, {
        cacheConfig: PERFORMANCE_PRESETS.balanced,
        enableStructuralPatterns: true,
        enableCoverageTracking: true,
        batchProcessingEnabled: true,
        stageBTargetMs: 4,
      });

      await engine.initialize();

      try {
        // Step 1: Initial project indexing
        console.log('  • Step 1: Initial project indexing');
        const projectFiles = Object.entries(realWorldFiles).map(([filename, content]) => ({
          filePath: `/project/${filename}`,
          content,
          language: 'typescript' as const
        }));

        const indexingStart = Date.now();
        await engine.batchIndexFiles(projectFiles);
        const indexingTime = Date.now() - indexingStart;
        console.log(`    - Indexed ${projectFiles.length} files in ${indexingTime}ms`);

        // Step 2: Developer searches (typical queries)
        console.log('  • Step 2: Developer searches');
        const developerQueries = [
          'UserList',        // Finding React component
          'ApiService',      // Finding service class
          'useState',        // Finding React hook usage
          'interface User',  // Finding type definition
          'async',          // Finding async functions
          'EventEmitter',   // Finding base classes
        ];

        const searchResults = [];
        for (const query of developerQueries) {
          const searchStart = Date.now();
          const results = await engine.searchSymbols(query, { 
            workspace_root: '/project', 
            language: 'typescript' 
          });
          const searchTime = Date.now() - searchStart;
          
          searchResults.push({ query, time: searchTime, count: results.length });
          console.log(`    - "${query}": ${searchTime}ms (${results.length} results)`);
        }

        // Step 3: File modification and re-indexing
        console.log('  • Step 3: File modification simulation');
        const modifiedContent = realWorldFiles['data-utils.ts'] + '\n\n// New function added\nexport const newUtility = () => {};';
        
        const reindexStart = Date.now();
        await engine.indexFile('/project/data-utils.ts', modifiedContent, 'typescript');
        const reindexTime = Date.now() - reindexStart;
        console.log(`    - Re-indexed modified file in ${reindexTime}ms`);

        // Step 4: Performance summary
        console.log('  • Step 4: Performance Summary');
        const avgSearchTime = searchResults.reduce((sum, r) => sum + r.time, 0) / searchResults.length;
        const totalResults = searchResults.reduce((sum, r) => sum + r.count, 0);
        
        console.log(`    - Average search time: ${avgSearchTime.toFixed(2)}ms`);
        console.log(`    - Total search results: ${totalResults}`);
        console.log(`    - Re-index performance: ${reindexTime}ms`);

        // Validate performance targets
        expect(avgSearchTime).toBeLessThan(4);
        expect(reindexTime).toBeLessThan(8); // Allow more time for single file re-index
        expect(totalResults).toBeGreaterThan(0);

        // Step 5: Coverage and metrics
        const coverageReport = engine.getCoverageReport();
        const metrics = engine.getEnhancedMetrics();
        
        console.log(`    - Index coverage: ${coverageReport.metrics.coveragePercentage}%`);
        console.log(`    - Cache hit rate: ${metrics.cache.hitRate}%`);
        console.log(`    - Symbols processed: ${metrics.performance.symbolsProcessed}`);

        expect(coverageReport.metrics.coveragePercentage).toBeGreaterThan(95);

      } finally {
        await engine.shutdown();
      }
    });
  });
});