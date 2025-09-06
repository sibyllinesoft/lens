/**
 * RAPTOR Metrics API Endpoints
 * 
 * Provides HTTP endpoints for accessing topic staleness, pressure telemetry,
 * and performance metrics from the RAPTOR system.
 */

import { FastifyInstance } from 'fastify';
import { MetricsTelemetry } from './metrics-telemetry.js';

// Response schemas for API documentation and validation
const TopicStalenessResponseSchema = {
  type: 'object',
  properties: {
    repo_sha: { type: 'string' },
    timestamp: { type: 'number' },
    overall_staleness: {
      type: 'object',
      properties: {
        avg_staleness: { type: 'number' },
        max_staleness: { type: 'number' },
        stale_topics_count: { type: 'number' },
        total_topics: { type: 'number' }
      }
    },
    by_level: {
      type: 'object',
      additionalProperties: {
        type: 'object',
        properties: {
          avg_staleness: { type: 'number' },
          stale_count: { type: 'number' },
          total_count: { type: 'number' }
        }
      }
    },
    stalest_topics: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          topic_id: { type: 'string' },
          staleness_score: { type: 'number' },
          age_days: { type: 'number' },
          card_count: { type: 'number' },
          last_updated: { type: 'number' }
        }
      }
    },
    freshness_distribution: {
      type: 'object',
      properties: {
        fresh: { type: 'number' },
        moderate: { type: 'number' },
        stale: { type: 'number' }
      }
    }
  }
} as const;

const TopicPressureResponseSchema = {
  type: 'object',
  properties: {
    repo_sha: { type: 'string' },
    timestamp: { type: 'number' },
    pressure_summary: {
      type: 'object',
      properties: {
        avg_pressure: { type: 'number' },
        max_pressure: { type: 'number' },
        high_pressure_topics: { type: 'number' },
        total_topics: { type: 'number' }
      }
    },
    pressure_components: {
      type: 'object',
      properties: {
        size_pressure: {
          type: 'object',
          properties: {
            avg: { type: 'number' },
            topics_near_split: { type: 'number' }
          }
        },
        stability_pressure: {
          type: 'object',
          properties: {
            avg: { type: 'number' },
            unstable_topics: { type: 'number' }
          }
        },
        staleness_pressure: {
          type: 'object',
          properties: {
            avg: { type: 'number' },
            pressure_from_age: { type: 'number' }
          }
        }
      }
    },
    high_pressure_topics: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          topic_id: { type: 'string' },
          total_pressure: { type: 'number' },
          size_pressure: { type: 'number' },
          stability_pressure: { type: 'number' },
          staleness_pressure: { type: 'number' },
          recommended_action: { type: 'string' }
        }
      }
    }
  }
} as const;

const SystemTelemetryResponseSchema = {
  type: 'object',
  properties: {
    timestamp: { type: 'number' },
    staleness_metrics: TopicStalenessResponseSchema,
    pressure_metrics: TopicPressureResponseSchema,
    performance_metrics: {
      type: 'object',
      properties: {
        repo_sha: { type: 'string' },
        timestamp: { type: 'number' },
        query_performance: {
          type: 'object',
          properties: {
            avg_topic_search_ms: { type: 'number' },
            avg_card_retrieval_ms: { type: 'number' },
            avg_symbol_resolution_ms: { type: 'number' },
            cache_hit_rates: {
              type: 'object',
              properties: {
                topic_cache: { type: 'number' },
                embedding_cache: { type: 'number' },
                symbol_cache: { type: 'number' }
              }
            }
          }
        },
        system_health: {
          type: 'object',
          properties: {
            memory_usage_mb: { type: 'number' },
            index_load_time_ms: { type: 'number' },
            active_snapshots: { type: 'number' }
          }
        },
        feature_utilization: {
          type: 'object',
          properties: {
            raptor_queries_pct: { type: 'number' },
            topic_hits_per_query: { type: 'number' },
            avg_businessness_boost: { type: 'number' },
            semantic_fallback_rate: { type: 'number' }
          }
        }
      }
    },
    alerts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severity: { type: 'string' },
          type: { type: 'string' },
          message: { type: 'string' },
          topic_id: { type: 'string', nullable: true },
          metric_value: { type: 'number' },
          threshold: { type: 'number' },
          timestamp: { type: 'number' },
          auto_remediation: { type: 'string', nullable: true }
        }
      }
    },
    health_status: { type: 'string' }
  }
} as const;

/**
 * Register all RAPTOR metrics endpoints with the Fastify server
 */
export async function registerRaptorMetricsEndpoints(
  fastify: FastifyInstance,
  metricsTelemetry: MetricsTelemetry
): Promise<void> {
  
  // Topic staleness metrics endpoint
  fastify.get('/metrics/topic_staleness', {
    schema: {
      description: 'Get topic staleness metrics showing how outdated topics are',
      tags: ['raptor-metrics'],
      response: {
        200: TopicStalenessResponseSchema
      }
    }
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const metrics = metricsTelemetry.getTopicStaleness();
      const latency = Date.now() - startTime;
      
      reply.header('X-Computation-Time-Ms', latency.toString());
      reply.header('Content-Type', 'application/json; charset=utf-8');
      
      return metrics;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(500);
      return {
        error: 'Failed to get topic staleness metrics',
        message: errorMsg,
        timestamp: Date.now()
      };
    }
  });

  // Topic pressure metrics endpoint
  fastify.get('/metrics/topic_pressure', {
    schema: {
      description: 'Get topic pressure metrics indicating topics that need attention',
      tags: ['raptor-metrics'],
      response: {
        200: TopicPressureResponseSchema
      }
    }
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const metrics = metricsTelemetry.getTopicPressure();
      const latency = Date.now() - startTime;
      
      reply.header('X-Computation-Time-Ms', latency.toString());
      reply.header('Content-Type', 'application/json; charset=utf-8');
      
      return metrics;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(500);
      return {
        error: 'Failed to get topic pressure metrics',
        message: errorMsg,
        timestamp: Date.now()
      };
    }
  });

  // Performance metrics endpoint
  fastify.get('/metrics/performance', {
    schema: {
      description: 'Get RAPTOR system performance metrics',
      tags: ['raptor-metrics'],
      response: {
        200: {
          type: 'object',
          properties: {
            repo_sha: { type: 'string' },
            timestamp: { type: 'number' },
            query_performance: {
              type: 'object',
              properties: {
                avg_topic_search_ms: { type: 'number' },
                avg_card_retrieval_ms: { type: 'number' },
                avg_symbol_resolution_ms: { type: 'number' },
                cache_hit_rates: {
                  type: 'object',
                  properties: {
                    topic_cache: { type: 'number' },
                    embedding_cache: { type: 'number' },
                    symbol_cache: { type: 'number' }
                  }
                }
              }
            },
            system_health: {
              type: 'object',
              properties: {
                memory_usage_mb: { type: 'number' },
                index_load_time_ms: { type: 'number' },
                active_snapshots: { type: 'number' }
              }
            },
            feature_utilization: {
              type: 'object',
              properties: {
                raptor_queries_pct: { type: 'number' },
                topic_hits_per_query: { type: 'number' },
                avg_businessness_boost: { type: 'number' },
                semantic_fallback_rate: { type: 'number' }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const metrics = metricsTelemetry.getPerformanceMetrics();
      const latency = Date.now() - startTime;
      
      reply.header('X-Computation-Time-Ms', latency.toString());
      reply.header('Content-Type', 'application/json; charset=utf-8');
      
      return metrics;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(500);
      return {
        error: 'Failed to get performance metrics',
        message: errorMsg,
        timestamp: Date.now()
      };
    }
  });

  // Comprehensive system telemetry endpoint
  fastify.get('/metrics/system_telemetry', {
    schema: {
      description: 'Get comprehensive RAPTOR system telemetry including all metrics and alerts',
      tags: ['raptor-metrics'],
      querystring: {
        type: 'object',
        properties: {
          hours: { 
            type: 'number', 
            description: 'Number of hours of history to include',
            minimum: 1,
            maximum: 168 // 1 week max
          }
        }
      },
      response: {
        200: SystemTelemetryResponseSchema
      }
    }
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const telemetry = await metricsTelemetry.getCurrentTelemetry();
      const latency = Date.now() - startTime;
      
      reply.header('X-Computation-Time-Ms', latency.toString());
      reply.header('Content-Type', 'application/json; charset=utf-8');
      
      return telemetry;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(500);
      return {
        error: 'Failed to get system telemetry',
        message: errorMsg,
        timestamp: Date.now()
      };
    }
  });

  // Historical metrics endpoint
  fastify.get('/metrics/history', {
    schema: {
      description: 'Get historical metrics data for trend analysis',
      tags: ['raptor-metrics'],
      querystring: {
        type: 'object',
        properties: {
          hours: { 
            type: 'number', 
            description: 'Number of hours of history to retrieve',
            minimum: 1,
            maximum: 168 // 1 week max
          },
          metric_type: {
            type: 'string',
            enum: ['staleness', 'pressure', 'performance', 'all'],
            description: 'Type of metrics to retrieve'
          }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            history: {
              type: 'array',
              items: SystemTelemetryResponseSchema
            },
            summary: {
              type: 'object',
              properties: {
                time_range_hours: { type: 'number' },
                data_points: { type: 'number' },
                avg_collection_interval_ms: { type: 'number' }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const query = request.query as { hours?: number; metric_type?: string };
      const hours = query.hours || 24; // Default to last 24 hours
      
      const startTime = Date.now();
      const history = metricsTelemetry.getMetricsHistory(hours);
      const latency = Date.now() - startTime;
      
      // Calculate summary statistics
      const intervals = history.length > 1 ? 
        history.slice(1).map((curr, i) => curr.timestamp - history[i].timestamp) : [];
      const avgInterval = intervals.length > 0 ? 
        intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length : 0;
      
      const response = {
        history,
        summary: {
          time_range_hours: hours,
          data_points: history.length,
          avg_collection_interval_ms: avgInterval
        }
      };
      
      reply.header('X-Computation-Time-Ms', latency.toString());
      reply.header('Content-Type', 'application/json; charset=utf-8');
      
      return response;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(500);
      return {
        error: 'Failed to get metrics history',
        message: errorMsg,
        timestamp: Date.now()
      };
    }
  });

  // Active alerts endpoint
  fastify.get('/metrics/alerts', {
    schema: {
      description: 'Get current active alerts from the RAPTOR system',
      tags: ['raptor-metrics'],
      querystring: {
        type: 'object',
        properties: {
          severity: {
            type: 'string',
            enum: ['low', 'medium', 'high', 'critical'],
            description: 'Filter alerts by severity level'
          },
          type: {
            type: 'string',
            enum: ['staleness', 'pressure', 'performance', 'error'],
            description: 'Filter alerts by type'
          }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            alerts: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  id: { type: 'string' },
                  severity: { type: 'string' },
                  type: { type: 'string' },
                  message: { type: 'string' },
                  topic_id: { type: 'string', nullable: true },
                  metric_value: { type: 'number' },
                  threshold: { type: 'number' },
                  timestamp: { type: 'number' },
                  auto_remediation: { type: 'string', nullable: true }
                }
              }
            },
            summary: {
              type: 'object',
              properties: {
                total_alerts: { type: 'number' },
                by_severity: {
                  type: 'object',
                  properties: {
                    critical: { type: 'number' },
                    high: { type: 'number' },
                    medium: { type: 'number' },
                    low: { type: 'number' }
                  }
                },
                by_type: {
                  type: 'object',
                  properties: {
                    staleness: { type: 'number' },
                    pressure: { type: 'number' },
                    performance: { type: 'number' },
                    error: { type: 'number' }
                  }
                }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const query = request.query as { severity?: string; type?: string };
      
      const startTime = Date.now();
      let alerts = metricsTelemetry.getActiveAlerts();
      
      // Apply filters
      if (query.severity) {
        alerts = alerts.filter(alert => alert.severity === query.severity);
      }
      
      if (query.type) {
        alerts = alerts.filter(alert => alert.type === query.type);
      }
      
      // Compute summary statistics
      const bySeverity = {
        critical: alerts.filter(a => a.severity === 'critical').length,
        high: alerts.filter(a => a.severity === 'high').length,
        medium: alerts.filter(a => a.severity === 'medium').length,
        low: alerts.filter(a => a.severity === 'low').length
      };
      
      const byType = {
        staleness: alerts.filter(a => a.type === 'staleness').length,
        pressure: alerts.filter(a => a.type === 'pressure').length,
        performance: alerts.filter(a => a.type === 'performance').length,
        error: alerts.filter(a => a.type === 'error').length
      };
      
      const latency = Date.now() - startTime;
      
      const response = {
        alerts,
        summary: {
          total_alerts: alerts.length,
          by_severity: bySeverity,
          by_type: byType
        }
      };
      
      reply.header('X-Computation-Time-Ms', latency.toString());
      reply.header('Content-Type', 'application/json; charset=utf-8');
      
      return response;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(500);
      return {
        error: 'Failed to get active alerts',
        message: errorMsg,
        timestamp: Date.now()
      };
    }
  });

  // Health check endpoint specifically for RAPTOR metrics
  fastify.get('/metrics/health', {
    schema: {
      description: 'Check health of the RAPTOR metrics collection system',
      tags: ['raptor-metrics'],
      response: {
        200: {
          type: 'object',
          properties: {
            status: { type: 'string' },
            timestamp: { type: 'number' },
            collection_active: { type: 'boolean' },
            last_collection: { type: 'number' },
            uptime_ms: { type: 'number' },
            version: { type: 'string' },
            config: {
              type: 'object',
              properties: {
                collection_interval_ms: { type: 'number' },
                retention_days: { type: 'number' },
                auto_remediation_enabled: { type: 'boolean' }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const telemetryData = metricsTelemetry.exportTelemetryData();
      const metricsHistory = telemetryData.metrics_history;
      const lastCollection = metricsHistory.length > 0 ? 
        Math.max(...metricsHistory.map(m => m.timestamp)) : 0;
      
      const health = {
        status: 'healthy',
        timestamp: Date.now(),
        collection_active: true,
        last_collection: lastCollection,
        uptime_ms: Date.now() - (lastCollection || Date.now()),
        version: '1.0.0',
        config: {
          collection_interval_ms: telemetryData.config.collection_interval_ms,
          retention_days: telemetryData.config.retention_days,
          auto_remediation_enabled: telemetryData.config.auto_remediation_enabled
        }
      };
      
      reply.header('Content-Type', 'application/json; charset=utf-8');
      return health;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(500);
      return {
        status: 'unhealthy',
        timestamp: Date.now(),
        error: errorMsg,
        collection_active: false,
        last_collection: 0,
        uptime_ms: 0,
        version: '1.0.0'
      };
    }
  });
  
  // Configuration update endpoint (for dynamic tuning)
  fastify.post('/metrics/config', {
    schema: {
      description: 'Update metrics collection configuration',
      tags: ['raptor-metrics'],
      body: {
        type: 'object',
        properties: {
          collection_interval_ms: { type: 'number', minimum: 10000 }, // Min 10 seconds
          retention_days: { type: 'number', minimum: 1, maximum: 30 },
          staleness_alert_threshold: { type: 'number', minimum: 0, maximum: 1 },
          pressure_alert_threshold: { type: 'number', minimum: 0, maximum: 1 },
          auto_remediation_enabled: { type: 'boolean' }
        },
        additionalProperties: false
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            updated_config: { type: 'object' },
            timestamp: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const updateConfig = request.body as any;
      
      // Update the telemetry configuration
      metricsTelemetry.updateConfig(updateConfig);
      
      // Return the updated configuration
      const telemetryData = metricsTelemetry.exportTelemetryData();
      
      reply.header('Content-Type', 'application/json; charset=utf-8');
      return {
        success: true,
        updated_config: telemetryData.config,
        timestamp: Date.now()
      };
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      
      reply.status(400);
      return {
        success: false,
        error: 'Failed to update configuration',
        message: errorMsg,
        timestamp: Date.now()
      };
    }
  });
}