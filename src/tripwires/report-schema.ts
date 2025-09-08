// Benchmark Report Schema
// All benchmark runs must conform to this exact schema

export interface BenchmarkReport {
  // System Under Test information
  sut: {
    name: string;           // e.g., "lens-core"
    git_sha: string;        // Exact git commit
    build_sha?: string;     // Binary/artifact hash if applicable  
    rustc?: string;         // e.g., "1.80.1"
    target: string;         // e.g., "x86_64-unknown-linux-gnu"
    mode: 'real';           // MUST be 'real', never 'mock'
  };

  // Anti-fraud handshake
  handshake: {
    nonce: string;          // Random nonce sent to service
    response: string;       // SHA256(nonce || build_sha) from service
  };

  // Environment capture
  env: {
    cpu: string;            // CPU model
    cores: number;          // CPU core count
    kernel: string;         // Kernel version
    governor: string;       // CPU governor (should be 'performance')
    numa: string;           // NUMA topology
    memory_gb: number;      // Total system memory
    hostname: string;       // System hostname
  };

  // Dataset provenance
  dataset: {
    uri: string;            // Full URI to dataset (file://, s3://, etc.)
    sha256: string;         // SHA256 hash of dataset
    size_bytes: number;     // Dataset size in bytes
  };

  // Benchmark harness info
  harness: {
    git_sha: string;        // Harness git commit
    cmdline: string;        // Command line used to run benchmark
    version: string;        // Harness version
  };

  // Performance metrics
  metrics: {
    qps?: number;           // Queries per second
    p50_ms?: number;        // 50th percentile latency
    p95_ms?: number;        // 95th percentile latency
    p99_ms?: number;        // 99th percentile latency  
    rss_mb?: number;        // Memory usage
    cpu_percent?: number;   // CPU utilization
    [key: string]: any;     // Additional metrics allowed
  };

  // Binary/container attestation
  artifacts?: {
    binary_digest?: string; // SHA256 of binary
    image_digest?: string;  // SHA256 of container image
    attestation?: string;   // SLSA attestation URI
  };

  // Metadata
  metadata: {
    benchmark_id: string;   // Unique benchmark ID
    started_at: string;     // ISO timestamp when started
    completed_at: string;   // ISO timestamp when completed
    duration_ms: number;    // Total benchmark duration
  };
}

// Schema validation function
export function validateBenchmarkReport(report: any): report is BenchmarkReport {
  const required = [
    'sut', 'handshake', 'env', 'dataset', 'harness', 'metrics', 'metadata'
  ];
  
  for (const field of required) {
    if (!(field in report)) {
      throw new Error(`SCHEMA VIOLATION: Missing required field '${field}'`);
    }
  }
  
  // Validate SUT mode
  if (report.sut.mode !== 'real') {
    throw new Error(`TRIPWIRE VIOLATION: SUT mode must be 'real', got '${report.sut.mode}'`);
  }
  
  // Validate handshake fields
  if (!report.handshake.nonce || !report.handshake.response) {
    throw new Error('SCHEMA VIOLATION: Missing handshake nonce or response');
  }
  
  // Validate dataset provenance
  if (!report.dataset.uri || !report.dataset.sha256) {
    throw new Error('SCHEMA VIOLATION: Missing dataset URI or SHA256');
  }
  
  return true;
}

// Banned patterns that trigger tripwires
export const BANNED_PATTERNS = [
  /generateMock/i,
  /simulate/i,
  /MOCK_RESULT/i,
  /mock_file_/i,
  /\.rust$/, // .rust extensions in content paths
];

export function checkForBannedPatterns(text: string): string[] {
  const violations = [];
  for (const pattern of BANNED_PATTERNS) {
    if (pattern.test(text)) {
      violations.push(`Banned pattern detected: ${pattern.source}`);
    }
  }
  return violations;
}