#!/usr/bin/env node

/**
 * Mock Lens Server for Phase 1 Testing
 * Implements minimal endpoints needed for the 10-step protocol
 */

const http = require('http');
const url = require('url');

// Mock data generation
function generateMockBenchmarkResults(traceId) {
  return {
    success: true,
    trace_id: traceId,
    status: 'completed',
    benchmark_results: {
      system: 'SMOKE',
      status: 'completed',
      total_queries: 50,
      completed_queries: 48,
      failed_queries: 2,
      config_fingerprint: `smoke-baseline-${Date.now()}`,
      metrics: {
        recall_at_10: 0.752,
        recall_at_50: 0.856,
        ndcg_at_10: 0.743,
        mrr: 0.681,
        first_relevant_tokens: 8.3,
        stage_latencies: {
          stage_a_p50: 42,
          stage_a_p95: 78,
          stage_a_p99: 115,
          stage_b_p50: 68, 
          stage_b_p95: 103,
          stage_b_p99: 142,
          stage_c_p50: 85,
          stage_c_p95: 131,
          stage_c_p99: 189,
          e2e_p95: 312
        },
        fan_out_sizes: {
          positives_in_candidates: 16,
          stage_a_candidates: 847,
          stage_b_candidates: 213
        }
      },
      errors: [
        {
          query_id: 'q_23',
          error_type: 'timeout',
          stage: 'stage_c',
          message: 'Semantic search timeout after 300ms'
        },
        {
          query_id: 'q_41', 
          error_type: 'index_missing',
          stage: 'stage_a',
          message: 'Repository not found in lexical index'
        }
      ],
      timestamp: new Date().toISOString()
    },
    artifacts: {
      metrics_parquet: `/benchmark-results/baseline_${traceId}_metrics.parquet`,
      errors_ndjson: `/benchmark-results/baseline_${traceId}_errors.ndjson`, 
      traces_ndjson: `/benchmark-results/baseline_${traceId}_traces.ndjson`,
      report_pdf: `/benchmark-results/baseline_${traceId}_report.pdf`,
      config_fingerprint_json: `/benchmark-results/baseline_${traceId}_config.json`
    },
    duration_ms: 42847,
    timestamp: new Date().toISOString(),
    promotion_gate_status: {
      passed: true,
      criteria: {
        ndcg_improvement: { required: '≥ +2%', actual: '+2.3%', passed: true },
        recall_maintained: { required: '≥ 0.850', actual: '0.856', passed: true }, 
        latency_acceptable: { required: '≤ +10%', actual: '+3.1%', passed: true }
      },
      summary: 'All promotion gates PASSED'
    }
  };
}

function generateMockPolicyDump() {
  return {
    api_version: 'v1.0.0-rc.1',
    index_version: 'v1.0.0',
    policy_version: 'v1.0.0',
    stage_configurations: {
      stage_a: {
        rare_term_fuzzy: true,
        synonyms_when_identifier_density_below: 0.3,
        prefilter: { type: 'bigram', enabled: true },
        wand: { enabled: true, block_max: true },
        per_file_span_cap: 3,
        native_scanner: 'auto'
      },
      stage_b: {
        enabled: true,
        symbol_ranking_enabled: true,
        reranker_enabled: true,
        max_candidates: 200
      },
      stage_c: {
        enabled: true,
        semantic_gating: {
          nl_likelihood_threshold: 0.5,
          min_candidates: 10
        },
        ann_config: {
          efSearch: 64,
          k: 50
        },
        confidence_cutoff: 0.1
      }
    },
    kill_switches: {
      stage_b_enabled: true,
      stage_c_enabled: true,
      stage_a_native_scanner: true,
      kill_switch_active: false
    },
    telemetry: {
      trace_sample_rate: 0.15,
      metrics_enabled: true
    },
    quality_gates: {
      ndcg_improvement_threshold: 0.02,
      recall_maintenance_threshold: 0.85, 
      latency_increase_threshold: 0.10
    },
    timestamp: new Date().toISOString(),
    config_fingerprint: `baseline-policy-${Date.now()}`
  };
}

function generateMockHealthStatus() {
  return {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    shards_healthy: 3,
    stage_a_ready: true,
    loaded_repos: 5,
    memory_usage: 0.62,
    disk_usage: 0.34
  };
}

function generateMockCanaryStatus() {
  return {
    success: true,
    canary_deployment: {
      trafficPercentage: 25,
      killSwitchEnabled: false,
      stageFlags: {
        stageA_native_scanner: true,
        stageB_enabled: true,
        stageC_enabled: true
      }
    },
    timestamp: new Date().toISOString()
  };
}

// Create HTTP server
const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const path = parsedUrl.pathname;
  const method = req.method;

  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS, PATCH');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-trace-id');

  // Handle preflight requests
  if (method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // Route handling
  if (method === 'GET' && path === '/health') {
    res.setHeader('Content-Type', 'application/json');
    res.writeHead(200);
    res.end(JSON.stringify(generateMockHealthStatus()));

  } else if (method === 'GET' && path === '/policy/dump') {
    res.setHeader('Content-Type', 'application/json'); 
    res.writeHead(200);
    res.end(JSON.stringify(generateMockPolicyDump()));

  } else if (method === 'POST' && path === '/bench/run') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const requestData = JSON.parse(body);
        const traceId = requestData.trace_id || req.headers['x-trace-id'] || `mock-${Date.now()}`;
        
        res.setHeader('Content-Type', 'application/json');
        res.writeHead(200);
        res.end(JSON.stringify(generateMockBenchmarkResults(traceId)));
      } catch (e) {
        res.writeHead(400);
        res.end(JSON.stringify({ error: 'Invalid JSON', message: e.message }));
      }
    });

  } else if (method === 'GET' && path === '/canary/status') {
    res.setHeader('Content-Type', 'application/json');
    res.writeHead(200);
    res.end(JSON.stringify(generateMockCanaryStatus()));

  } else {
    res.writeHead(404);
    res.end(JSON.stringify({ error: 'Not Found', path, method }));
  }
});

const PORT = 3000;
const HOST = '0.0.0.0';

server.listen(PORT, HOST, () => {
  console.log(`🚀 Mock Lens server running on http://${HOST}:${PORT}`);
  console.log(`📊 Health check: http://${HOST}:${PORT}/health`);
  console.log(`🔧 Policy dump: http://${HOST}:${PORT}/policy/dump`);
  console.log(`🧪 Benchmark: POST http://${HOST}:${PORT}/bench/run`);
  console.log('');
  console.log('Ready for Phase 1 testing!');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down mock server...');
  server.close(() => {
    console.log('Mock server stopped.');
    process.exit(0);
  });
});