/**
 * RAPTOR Metrics HTTP Endpoint
 * 
 * Exposes /metrics/raptor endpoint for telemetry collection.
 * Provides both Prometheus format and JSON format responses.
 */

import { Request, Response } from 'express';
import { raptorTelemetry, type TopicExplanation } from '../telemetry/raptor-metrics.js';

/**
 * GET /metrics/raptor
 * 
 * Returns RAPTOR metrics in Prometheus format by default,
 * or JSON format with ?format=json
 */
export async function getRaptorMetrics(req: Request, res: Response): Promise<void> {
  try {
    const format = req.query.format as string;
    
    if (format === 'json') {
      // JSON format for debugging and dashboards
      const metrics = raptorTelemetry.getMetrics();
      const stats = raptorTelemetry.getAggregatedStats(60); // Last hour
      
      res.json({
        timestamp: new Date().toISOString(),
        metrics,
        aggregated_stats_1h: stats,
        system_info: {
          uptime_ms: process.uptime() * 1000,
          memory_usage: process.memoryUsage(),
          version: process.env.LENS_VERSION || 'unknown'
        }
      });
      
    } else {
      // Prometheus format (default)
      const prometheusMetrics = raptorTelemetry.getPrometheusMetrics();
      
      res.setHeader('Content-Type', 'text/plain; version=0.0.4; charset=utf-8');
      res.send(prometheusMetrics);
    }
    
  } catch (error) {
    console.error('Error serving RAPTOR metrics:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve metrics',
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * GET /metrics/raptor/explanations
 * 
 * Returns recent topic path explanations for NL queries
 */
export async function getTopicExplanations(req: Request, res: Response): Promise<void> {
  try {
    const limit = parseInt(req.query.limit as string) || 10;
    const explanations = raptorTelemetry.getTopicExplanations(limit);
    
    res.json({
      timestamp: new Date().toISOString(),
      explanations,
      total_count: explanations.length
    });
    
  } catch (error) {
    console.error('Error serving topic explanations:', error);
    res.status(500).json({
      error: 'Failed to retrieve topic explanations',
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * GET /metrics/raptor/debug
 * 
 * Returns detailed debugging information
 */
export async function getRaptorDebugInfo(req: Request, res: Response): Promise<void> {
  try {
    const exportData = raptorTelemetry.exportMetrics();
    const stats = raptorTelemetry.getAggregatedStats(60);
    
    res.json({
      timestamp: new Date().toISOString(),
      debug_info: {
        total_queries_tracked: exportData.query_history.length,
        recent_stats: stats,
        system_health: {
          staleness_seconds: exportData.summary.staleness_seconds,
          pressure: exportData.summary.pressure,
          reclusters: exportData.summary.reclusters
        },
        mix_distribution: exportData.summary.why_mix,
        topic_paths_sample: exportData.summary.recent_topic_paths.slice(-5)
      }
    });
    
  } catch (error) {
    console.error('Error serving RAPTOR debug info:', error);
    res.status(500).json({
      error: 'Failed to retrieve debug information',
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * POST /metrics/raptor/reset
 * 
 * Reset metrics (for testing/development only)
 */
export async function resetRaptorMetrics(req: Request, res: Response): Promise<void> {
  try {
    // Only allow in development
    if (process.env.NODE_ENV === 'production') {
      res.status(403).json({
        error: 'Metrics reset not allowed in production',
        timestamp: new Date().toISOString()
      });
      return;
    }
    
    raptorTelemetry.reset();
    
    res.json({
      message: 'RAPTOR metrics reset successfully',
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Error resetting RAPTOR metrics:', error);
    res.status(500).json({
      error: 'Failed to reset metrics',
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * Express router setup for RAPTOR metrics endpoints
 */
export function setupRaptorMetricsRoutes(app: any): void {
  // Main metrics endpoint
  app.get('/metrics/raptor', getRaptorMetrics);
  
  // Topic explanations endpoint
  app.get('/metrics/raptor/explanations', getTopicExplanations);
  
  // Debug information endpoint
  app.get('/metrics/raptor/debug', getRaptorDebugInfo);
  
  // Reset endpoint (dev only)
  app.post('/metrics/raptor/reset', resetRaptorMetrics);
  
  console.log('🔍 RAPTOR metrics endpoints registered:');
  console.log('   GET  /metrics/raptor - Prometheus metrics');
  console.log('   GET  /metrics/raptor?format=json - JSON metrics');
  console.log('   GET  /metrics/raptor/explanations - Topic path explanations');
  console.log('   GET  /metrics/raptor/debug - Debug information');
  if (process.env.NODE_ENV !== 'production') {
    console.log('   POST /metrics/raptor/reset - Reset metrics (dev only)');
  }
}

/**
 * Middleware to automatically record query events
 */
export function createQueryTrackingMiddleware() {
  return (req: Request, res: Response, next: any) => {
    // Capture query start time
    const startTime = Date.now();
    
    // Override res.json to capture response
    const originalJson = res.json;
    res.json = function(body: any) {
      const latency = Date.now() - startTime;
      
      // Extract query information from request/response
      if (req.body?.query && body?.results) {
        try {
          raptorTelemetry.recordQuery({
            query: req.body.query,
            intent: req.body.intent || 'unknown',
            language: req.body.language,
            results: body.results.map((result: any) => ({
              file_path: result.file_path || result.path || 'unknown',
              score: result.score || 0,
              mix_breakdown: result.mix_breakdown || {
                exact: 0, fuzzy: 0, symbol: 0, struct: 0, semantic: 0, topic_hit: 0
              },
              topic_ids: result.topic_ids
            })),
            latency_ms: latency,
            timestamp: new Date()
          });
        } catch (error) {
          console.error('Error recording query telemetry:', error);
        }
      }
      
      return originalJson.call(this, body);
    };
    
    next();
  };
}

/**
 * Health check for RAPTOR metrics system
 */
export function checkRaptorMetricsHealth(): {
  status: 'healthy' | 'degraded' | 'unhealthy';
  issues: string[];
  metrics_summary: any;
} {
  const issues: string[] = [];
  const metrics = raptorTelemetry.getMetrics();
  
  // Check staleness
  if (metrics.staleness_seconds > 300) { // 5 minutes
    issues.push(`High staleness: ${metrics.staleness_seconds}s`);
  }
  
  // Check system pressure
  if (metrics.pressure > 0.8) {
    issues.push(`High system pressure: ${(metrics.pressure * 100).toFixed(1)}%`);
  }
  
  // Check SLA pass rate
  if (metrics.sla_pass_rate_150ms < 0.9) {
    issues.push(`Low SLA pass rate: ${(metrics.sla_pass_rate_150ms * 100).toFixed(1)}%`);
  }
  
  // Determine overall status
  let status: 'healthy' | 'degraded' | 'unhealthy';
  if (issues.length === 0) {
    status = 'healthy';
  } else if (issues.length <= 2 && metrics.sla_pass_rate_150ms > 0.5) {
    status = 'degraded';
  } else {
    status = 'unhealthy';
  }
  
  return {
    status,
    issues,
    metrics_summary: {
      topic_hit_rate: metrics.topic_hit_rate,
      sla_pass_rate: metrics.sla_pass_rate_150ms,
      staleness_seconds: metrics.staleness_seconds,
      pressure: metrics.pressure,
      recent_queries: raptorTelemetry.getAggregatedStats(5).total_queries
    }
  };
}