/**
 * RAPTOR API Endpoints
 * 
 * HTTP endpoints for managing RAPTOR system: policy management, snapshot
 * building, reclustering, and feature debugging.
 */

import { Request, Response } from 'express';
import { RaptorPolicyManager, RaptorPolicy, RollbackOptions } from './policy.js';
import { RaptorSnapshotManager, RaptorSnapshot } from './snapshot.js';
import { SemanticCardExtractor } from './extractor.js';
import { RaptorBuilder } from './builder.js';
import { ReclusterDaemon, DaemonConfig } from './recluster-daemon.js';
import { RaptorRuntimeFeatures } from './runtime-features.js';
import { RaptorEmbeddingService } from './embeddings.js';

export interface RaptorApiContext {
  policyManager: RaptorPolicyManager;
  snapshotManager: RaptorSnapshotManager;
  extractor: SemanticCardExtractor;
  builder: RaptorBuilder;
  daemon: ReclusterDaemon;
  runtimeFeatures: RaptorRuntimeFeatures;
  embeddingService: RaptorEmbeddingService;
}

/**
 * RAPTOR API endpoints
 */
export class RaptorApiEndpoints {
  private context: RaptorApiContext;

  constructor(context: RaptorApiContext) {
    this.context = context;
  }

  // Policy Management Endpoints

  async getPolicyStatus(req: Request, res: Response): Promise<void> {
    try {
      const policy = this.context.policyManager.getCurrentPolicy();
      const health = this.context.policyManager.checkPolicyHealth();
      const daemonStatus = this.context.daemon.getStatus();

      res.json({
        policy,
        health,
        daemon_status: daemonStatus,
        timestamp: Date.now()
      });
    } catch (error) {
      res.status(500).json({ 
        error: 'Failed to get policy status',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async updatePolicy(req: Request, res: Response): Promise<void> {
    try {
      const changes = req.body as Partial<RaptorPolicy>;
      const user = req.headers['x-user'] as string || 'anonymous';
      const reason = req.body._reason as string;

      if (!changes || Object.keys(changes).length === 0) {
        res.status(400).json({ error: 'No policy changes provided' });
        return;
      }

      // Remove metadata fields
      delete (changes as any)._reason;

      const validation = this.context.policyManager.updatePolicy(changes, user, reason);

      if (validation.valid) {
        res.json({
          success: true,
          policy: this.context.policyManager.getCurrentPolicy(),
          validation
        });
      } else {
        res.status(400).json({
          success: false,
          validation
        });
      }
    } catch (error) {
      res.status(500).json({
        error: 'Failed to update policy',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async rollbackPolicy(req: Request, res: Response): Promise<void> {
    try {
      const options = req.body as RollbackOptions;
      const user = req.headers['x-user'] as string || 'anonymous';

      const validation = this.context.policyManager.rollbackPolicy(options, user);

      if (validation.valid) {
        res.json({
          success: true,
          policy: this.context.policyManager.getCurrentPolicy(),
          validation
        });
      } else {
        res.status(400).json({
          success: false,
          validation
        });
      }
    } catch (error) {
      res.status(500).json({
        error: 'Failed to rollback policy',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async getPolicyHistory(req: Request, res: Response): Promise<void> {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : undefined;
      const history = this.context.policyManager.getHistory(limit);
      
      res.json({ history });
    } catch (error) {
      res.status(500).json({
        error: 'Failed to get policy history',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async getPolicySnapshots(req: Request, res: Response): Promise<void> {
    try {
      const snapshots = this.context.policyManager.getSnapshots();
      res.json({ snapshots });
    } catch (error) {
      res.status(500).json({
        error: 'Failed to get policy snapshots',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async emergencyDisable(req: Request, res: Response): Promise<void> {
    try {
      const reason = req.body.reason as string || 'Emergency disable requested';
      const user = req.headers['x-user'] as string || 'anonymous';

      const validation = this.context.policyManager.emergencyDisable(user, reason);

      res.json({
        success: true,
        message: 'RAPTOR system disabled',
        policy: this.context.policyManager.getCurrentPolicy(),
        validation
      });
    } catch (error) {
      res.status(500).json({
        error: 'Failed to emergency disable',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Snapshot Management Endpoints

  async getSnapshotInfo(req: Request, res: Response): Promise<void> {
    try {
      const repoSha = req.query.repo_sha as string;
      
      if (!repoSha) {
        res.status(400).json({ error: 'repo_sha parameter required' });
        return;
      }

      // In a real implementation, this would query the storage layer
      // For now, return mock metadata
      const metadata = {
        repo_sha: repoSha,
        version: 'latest',
        created_ts: Date.now(),
        file_count: 0,
        node_count: 0,
        levels: 0,
        staleness_hours: 0
      };

      res.json({ metadata });
    } catch (error) {
      res.status(500).json({
        error: 'Failed to get snapshot info',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async buildSnapshot(req: Request, res: Response): Promise<void> {
    try {
      const { repo_sha } = req.body;
      
      if (!repo_sha) {
        res.status(400).json({ error: 'repo_sha required' });
        return;
      }

      // Start background build process
      // In a real implementation, this would queue a build job
      const buildId = `build-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      
      res.json({
        success: true,
        build_id: buildId,
        message: 'Snapshot build started',
        estimated_duration: '5-15 minutes'
      });

      // TODO: Start actual build process asynchronously
    } catch (error) {
      res.status(500).json({
        error: 'Failed to start snapshot build',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async triggerRecluster(req: Request, res: Response): Promise<void> {
    try {
      const { repo_sha, node_id } = req.body;
      
      if (!repo_sha) {
        res.status(400).json({ error: 'repo_sha required' });
        return;
      }

      const reclusterId = `recluster-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

      res.json({
        success: true,
        recluster_id: reclusterId,
        message: node_id 
          ? `Reclustering node ${node_id}` 
          : 'Reclustering entire repository',
        estimated_duration: '1-5 minutes'
      });

      // TODO: Trigger actual reclustering
    } catch (error) {
      res.status(500).json({
        error: 'Failed to trigger recluster',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Feature Debugging Endpoints

  async getFeatureDebugInfo(req: Request, res: Response): Promise<void> {
    try {
      const fileId = req.query.file_id as string;
      const repoSha = req.query.repo_sha as string;
      
      if (!fileId) {
        res.status(400).json({ error: 'file_id parameter required' });
        return;
      }

      const debugInfo = this.context.runtimeFeatures.getFeatureDebugInfo(
        fileId, 
        repoSha || 'latest'
      );

      res.json({ debug_info: debugInfo });
    } catch (error) {
      res.status(500).json({
        error: 'Failed to get feature debug info',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async computeQueryFeatures(req: Request, res: Response): Promise<void> {
    try {
      const { query, repo_sha, candidate_files } = req.body;
      
      if (!query) {
        res.status(400).json({ error: 'query required' });
        return;
      }

      const queryEmbedding = await this.context.runtimeFeatures.prepareQuery(query);
      
      let features = new Map();
      if (candidate_files && candidate_files.length > 0) {
        features = await this.context.runtimeFeatures.computeFeatures(
          repo_sha || 'latest',
          candidate_files,
          queryEmbedding
        );
      }

      res.json({
        query_embedding: {
          features: {
            routes: Array.from(queryEmbedding.features.routes),
            tables: Array.from(queryEmbedding.features.tables),
            types: Array.from(queryEmbedding.features.types),
            tokens: Array.from(queryEmbedding.features.tokens),
            effects: Array.from(queryEmbedding.features.effects)
          }
        },
        file_features: Object.fromEntries(features)
      });
    } catch (error) {
      res.status(500).json({
        error: 'Failed to compute query features',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // System Health Endpoints

  async getSystemHealth(req: Request, res: Response): Promise<void> {
    try {
      const policy = this.context.policyManager.getCurrentPolicy();
      const policyHealth = this.context.policyManager.checkPolicyHealth();
      const daemonStatus = this.context.daemon.getStatus();

      // Compute overall health score
      let healthScore = 100;
      
      if (!policyHealth.healthy) {
        healthScore -= policyHealth.issues.length * 10;
      }
      
      if (daemonStatus.warnings.length > 0) {
        healthScore -= daemonStatus.warnings.length * 5;
      }

      if (daemonStatus.backlogSize > policy.pressure_alert_threshold) {
        healthScore -= 20;
      }

      const status = healthScore >= 80 ? 'healthy' : healthScore >= 60 ? 'degraded' : 'unhealthy';

      res.json({
        status,
        health_score: Math.max(0, healthScore),
        policy_health: policyHealth,
        daemon_status: daemonStatus,
        feature_flags: {
          raptor_enabled: policy.enabled,
          semantic_cards_enabled: policy.semantic_cards_enabled,
          prior_boost_enabled: policy.prior_boost_enabled,
          daemon_enabled: policy.recluster_daemon_enabled
        },
        timestamp: Date.now()
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        error: 'Failed to get system health',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  async getSystemMetrics(req: Request, res: Response): Promise<void> {
    try {
      const daemonStatus = this.context.daemon.getStatus();
      const policy = this.context.policyManager.getCurrentPolicy();

      const metrics = {
        // Performance metrics
        embedding_cache_size: this.context.embeddingService.getCacheSize(),
        extractor_cache_size: this.context.extractor.getCacheSize(),
        
        // Resource usage
        current_budget_usage: {
          summaries: daemonStatus.currentBudget.current_summaries_used / daemonStatus.currentBudget.max_summaries_per_hour,
          cpu_seconds: daemonStatus.currentBudget.current_cpu_used / daemonStatus.currentBudget.max_cpu_seconds_per_hour
        },

        // Pressure statistics
        pressure_stats: daemonStatus.pressureStats,
        
        // System limits
        policy_limits: {
          max_snapshots: policy.max_snapshots,
          max_cards_per_file: policy.max_cards_per_file,
          max_embedding_batch: policy.max_embedding_batch,
          ttl_days: policy.ttl_days
        },

        timestamp: Date.now()
      };

      res.json({ metrics });
    } catch (error) {
      res.status(500).json({
        error: 'Failed to get system metrics',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Utility methods for route setup
  setupRoutes(app: any): void {
    // Policy management routes
    app.get('/raptor/policy', this.getPolicyStatus.bind(this));
    app.patch('/raptor/policy', this.updatePolicy.bind(this));
    app.post('/raptor/policy/rollback', this.rollbackPolicy.bind(this));
    app.get('/raptor/policy/history', this.getPolicyHistory.bind(this));
    app.get('/raptor/policy/snapshots', this.getPolicySnapshots.bind(this));
    app.post('/raptor/policy/emergency-disable', this.emergencyDisable.bind(this));

    // Snapshot management routes
    app.get('/raptor/snapshot', this.getSnapshotInfo.bind(this));
    app.post('/raptor/build', this.buildSnapshot.bind(this));
    app.post('/raptor/recluster', this.triggerRecluster.bind(this));

    // Feature debugging routes
    app.get('/raptor/features', this.getFeatureDebugInfo.bind(this));
    app.post('/raptor/features/compute', this.computeQueryFeatures.bind(this));

    // System health routes
    app.get('/raptor/health', this.getSystemHealth.bind(this));
    app.get('/raptor/metrics', this.getSystemMetrics.bind(this));
  }

  // Helper method for creating middleware
  createAuthMiddleware(requiredRole: string = 'user') {
    return (req: Request, res: Response, next: any) => {
      const userRole = req.headers['x-user-role'] as string || 'anonymous';
      
      if (requiredRole === 'admin' && userRole !== 'admin') {
        res.status(403).json({ error: 'Admin access required' });
        return;
      }
      
      if (requiredRole === 'user' && !userRole) {
        res.status(401).json({ error: 'Authentication required' });
        return;
      }
      
      next();
    };
  }

  // Helper method for request validation
  validateRequest(schema: any) {
    return (req: Request, res: Response, next: any) => {
      try {
        // Simple validation - in production, use a proper schema validator
        const required = schema.required || [];
        for (const field of required) {
          if (!(field in req.body)) {
            res.status(400).json({ error: `Missing required field: ${field}` });
            return;
          }
        }
        next();
      } catch (error) {
        res.status(400).json({ error: 'Request validation failed' });
      }
    };
  }

  // Rate limiting helper
  createRateLimit(maxRequests: number, windowMs: number) {
    const requests = new Map<string, { count: number; resetTime: number }>();

    return (req: Request, res: Response, next: any) => {
      const clientId = req.ip || req.headers['x-forwarded-for'] as string || 'unknown';
      const now = Date.now();
      
      const clientData = requests.get(clientId);
      
      if (!clientData || now > clientData.resetTime) {
        requests.set(clientId, { count: 1, resetTime: now + windowMs });
        next();
        return;
      }
      
      if (clientData.count >= maxRequests) {
        res.status(429).json({ 
          error: 'Rate limit exceeded',
          retry_after: Math.ceil((clientData.resetTime - now) / 1000)
        });
        return;
      }
      
      clientData.count++;
      next();
    };
  }
}

export default RaptorApiEndpoints;