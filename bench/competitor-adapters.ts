/**
 * Competitor Adapter Framework - Unified interface for all benchmark systems
 * 
 * This provides a standardized interface for running different search systems
 * under identical conditions with SLA enforcement and result normalization.
 */

import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';
import fetch from 'node-fetch';

const execAsync = promisify(exec);

export interface AdapterConfig {
  system_id: string;
  corpus_path: string;
  index_path?: string;
  server_port?: number;
  warmup_queries?: number;
  resource_limits?: {
    memory_mb: number;
    cpu_cores: number;
  };
  custom_params?: Record<string, any>;
}

export interface SearchHit {
  file: string;
  line: number;
  column: number;
  snippet: string;
  score: number;
  why_tag: 'exact' | 'struct' | 'semantic' | 'mixed';
  symbol_kind?: string;
  ast_path?: string;
  byte_offset?: number;
  span_length?: number;
}

export interface SearchResponse {
  hits: SearchHit[];
  latency_ms: number;
  within_sla: boolean;
  error?: string;
  system_info?: SystemInfo;
}

export interface SystemInfo {
  system_id: string;
  version: string;
  build_hash?: string;
  config_fingerprint: string;
  hardware_info: HardwareInfo;
}

export interface HardwareInfo {
  cpu_model: string;
  cpu_cores: number;
  memory_gb: number;
  cpu_flags: string[];
  governor: string;
  kernel_version: string;
}

/**
 * Base competitor adapter interface
 */
export abstract class CompetitorAdapter {
  protected config: AdapterConfig;
  protected isInitialized = false;
  protected systemInfo?: SystemInfo;

  constructor(config: AdapterConfig) {
    this.config = config;
  }

  /**
   * System preparation and warmup
   */
  abstract prepare(): Promise<void>;

  /**
   * Core search interface with SLA enforcement
   */
  abstract search(query: string, sla_ms: number): Promise<SearchResponse>;

  /**
   * Cleanup and resource release
   */
  abstract teardown(): Promise<void>;

  /**
   * Get system metadata for attestation
   */
  async getSystemInfo(): Promise<SystemInfo> {
    if (!this.systemInfo) {
      this.systemInfo = await this.collectSystemInfo();
    }
    return this.systemInfo;
  }

  /**
   * Collect hardware and system information
   */
  protected async collectSystemInfo(): Promise<SystemInfo> {
    const [cpuInfo, memInfo, kernelInfo] = await Promise.all([
      this.getCpuInfo(),
      this.getMemoryInfo(),
      this.getKernelInfo()
    ]);

    return {
      system_id: this.config.system_id,
      version: await this.getSystemVersion(),
      build_hash: await this.getBuildHash(),
      config_fingerprint: this.calculateConfigFingerprint(),
      hardware_info: {
        cpu_model: cpuInfo.model,
        cpu_cores: cpuInfo.cores,
        memory_gb: memInfo.total_gb,
        cpu_flags: cpuInfo.flags,
        governor: cpuInfo.governor,
        kernel_version: kernelInfo
      }
    };
  }

  protected abstract getSystemVersion(): Promise<string>;
  protected abstract getBuildHash(): Promise<string | undefined>;

  private calculateConfigFingerprint(): string {
    const configStr = JSON.stringify(this.config, null, 0);
    return require('crypto').createHash('sha256').update(configStr).digest('hex').substring(0, 16);
  }

  private async getCpuInfo() {
    const cpuinfo = await fs.readFile('/proc/cpuinfo', 'utf8');
    const lines = cpuinfo.split('\n');
    
    const modelLine = lines.find(line => line.startsWith('model name'));
    const model = modelLine ? modelLine.split(':')[1].trim() : 'unknown';
    
    const cores = lines.filter(line => line.startsWith('processor')).length;
    
    const flagsLine = lines.find(line => line.startsWith('flags'));
    const flags = flagsLine ? flagsLine.split(':')[1].trim().split(' ') : [];
    
    const governor = await this.getCpuGovernor();
    
    return { model, cores, flags, governor };
  }

  private async getCpuGovernor(): Promise<string> {
    try {
      const { stdout } = await execAsync('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor');
      return stdout.trim();
    } catch {
      return 'unknown';
    }
  }

  private async getMemoryInfo() {
    const meminfo = await fs.readFile('/proc/meminfo', 'utf8');
    const totalLine = meminfo.split('\n').find(line => line.startsWith('MemTotal'));
    const totalKb = totalLine ? parseInt(totalLine.split(/\s+/)[1]) : 0;
    const total_gb = Math.round(totalKb / 1024 / 1024);
    
    return { total_gb };
  }

  private async getKernelInfo(): Promise<string> {
    const { stdout } = await execAsync('uname -r');
    return stdout.trim();
  }

  /**
   * Warmup the system with sample queries
   */
  protected async runWarmup(): Promise<void> {
    if (!this.config.warmup_queries) return;

    console.log(`🔥 Running warmup with ${this.config.warmup_queries} queries for ${this.config.system_id}`);
    
    const warmupQueries = [
      'function definition',
      'class implementation', 
      'error handling',
      'import statement',
      'variable assignment'
    ];

    for (let i = 0; i < this.config.warmup_queries; i++) {
      const query = warmupQueries[i % warmupQueries.length];
      try {
        await this.search(query, 1000); // Generous SLA for warmup
      } catch (error) {
        console.warn(`⚠️  Warmup query ${i} failed: ${error}`);
      }
    }

    console.log(`✅ Warmup complete for ${this.config.system_id}`);
  }
}

/**
 * Lens system adapter
 */
export class LensAdapter extends CompetitorAdapter {
  private serverProcess?: any;

  async prepare(): Promise<void> {
    console.log(`🚀 Preparing Lens adapter`);
    
    // Start Lens server if needed
    if (this.config.server_port) {
      await this.startLensServer();
    }
    
    await this.runWarmup();
    this.isInitialized = true;
  }

  async search(query: string, sla_ms: number): Promise<SearchResponse> {
    if (!this.isInitialized) {
      throw new Error('Lens adapter not initialized');
    }

    const startTime = Date.now();
    
    try {
      const response = await fetch(`http://localhost:${this.config.server_port}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit: 50 }),
        timeout: sla_ms
      });

      const latency_ms = Date.now() - startTime;
      const within_sla = latency_ms <= sla_ms;

      if (!response.ok) {
        return {
          hits: [],
          latency_ms,
          within_sla,
          error: `HTTP ${response.status}: ${response.statusText}`
        };
      }

      const data = await response.json();
      const hits = this.normalizeLensResults(data.results || []);

      return {
        hits,
        latency_ms,
        within_sla,
        system_info: await this.getSystemInfo()
      };

    } catch (error) {
      const latency_ms = Date.now() - startTime;
      return {
        hits: [],
        latency_ms,
        within_sla: false,
        error: error.message
      };
    }
  }

  private normalizeLensResults(results: any[]): SearchHit[] {
    return results.map((result, index) => ({
      file: result.file || result.path || '',
      line: result.line || 1,
      column: result.column || 1, 
      snippet: result.snippet || result.content || '',
      score: result.score || (1.0 - index * 0.01),
      why_tag: this.mapLensWhyTag(result.why_tag || result.match_type),
      symbol_kind: result.symbol_kind,
      ast_path: result.ast_path,
      byte_offset: result.byte_offset,
      span_length: result.span_length
    }));
  }

  private mapLensWhyTag(lensTag: string): 'exact' | 'struct' | 'semantic' | 'mixed' {
    switch (lensTag?.toLowerCase()) {
      case 'exact': case 'lexical': return 'exact';
      case 'structural': case 'ast': return 'struct'; 
      case 'semantic': case 'embedding': return 'semantic';
      default: return 'mixed';
    }
  }

  private async startLensServer(): Promise<void> {
    console.log(`🖥️  Starting Lens server on port ${this.config.server_port}`);
    
    this.serverProcess = spawn('npm', ['run', 'start'], {
      env: { ...process.env, PORT: this.config.server_port?.toString() },
      stdio: 'pipe'
    });

    // Wait for server to be ready
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Server start timeout')), 30000);
      
      const checkServer = async () => {
        try {
          const response = await fetch(`http://localhost:${this.config.server_port}/health`);
          if (response.ok) {
            clearTimeout(timeout);
            resolve(void 0);
          }
        } catch {
          // Server not ready yet
        }
      };

      const interval = setInterval(checkServer, 1000);
      checkServer();
    });
  }

  async teardown(): Promise<void> {
    if (this.serverProcess) {
      console.log(`🛑 Stopping Lens server`);
      this.serverProcess.kill('SIGTERM');
      this.serverProcess = undefined;
    }
  }

  protected async getSystemVersion(): Promise<string> {
    const packageJson = await fs.readFile('package.json', 'utf8');
    const pkg = JSON.parse(packageJson);
    return pkg.version || '0.0.0';
  }

  protected async getBuildHash(): Promise<string | undefined> {
    try {
      const { stdout } = await execAsync('git rev-parse HEAD');
      return stdout.trim().substring(0, 8);
    } catch {
      return undefined;
    }
  }
}

/**
 * Elasticsearch BM25 adapter
 */
export class ElasticsearchAdapter extends CompetitorAdapter {
  async prepare(): Promise<void> {
    console.log(`🔍 Preparing Elasticsearch BM25 adapter`);
    
    // Ensure Elasticsearch is running
    await this.ensureElasticsearchRunning();
    
    // Create index if needed
    await this.createIndex();
    
    await this.runWarmup();
    this.isInitialized = true;
  }

  async search(query: string, sla_ms: number): Promise<SearchResponse> {
    const startTime = Date.now();
    
    try {
      const searchBody = {
        query: {
          multi_match: {
            query,
            fields: ['content^2', 'filename^1.5', 'symbols^1'],
            type: 'best_fields'
          }
        },
        size: 50,
        timeout: `${sla_ms}ms`
      };

      const response = await fetch(`http://localhost:9200/code/_search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(searchBody),
        timeout: sla_ms
      });

      const latency_ms = Date.now() - startTime;
      const within_sla = latency_ms <= sla_ms;

      if (!response.ok) {
        return {
          hits: [],
          latency_ms,
          within_sla,
          error: `Elasticsearch error: ${response.status}`
        };
      }

      const data = await response.json();
      const hits = this.normalizeElasticsearchResults(data.hits?.hits || []);

      return { hits, latency_ms, within_sla };

    } catch (error) {
      const latency_ms = Date.now() - startTime;
      return {
        hits: [],
        latency_ms, 
        within_sla: false,
        error: error.message
      };
    }
  }

  private normalizeElasticsearchResults(esHits: any[]): SearchHit[] {
    return esHits.map(hit => ({
      file: hit._source.file || '',
      line: hit._source.line || 1,
      column: hit._source.column || 1,
      snippet: hit._source.content || '',
      score: hit._score || 0,
      why_tag: 'exact' as const
    }));
  }

  private async ensureElasticsearchRunning(): Promise<void> {
    try {
      const response = await fetch('http://localhost:9200/_cluster/health');
      if (response.ok) return;
    } catch {}
    
    throw new Error('Elasticsearch not running on localhost:9200');
  }

  private async createIndex(): Promise<void> {
    const indexExists = await fetch('http://localhost:9200/code').then(r => r.ok);
    
    if (!indexExists) {
      const mapping = {
        mappings: {
          properties: {
            file: { type: 'keyword' },
            line: { type: 'integer' },
            column: { type: 'integer' },
            content: { type: 'text', analyzer: 'standard' },
            symbols: { type: 'text', analyzer: 'keyword' }
          }
        }
      };

      await fetch('http://localhost:9200/code', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mapping)
      });
    }
  }

  async teardown(): Promise<void> {
    // Elasticsearch cleanup if needed
  }

  protected async getSystemVersion(): Promise<string> {
    const response = await fetch('http://localhost:9200/');
    const info = await response.json();
    return info.version?.number || 'unknown';
  }

  protected async getBuildHash(): Promise<string | undefined> {
    return undefined; // No build hash for Elasticsearch
  }
}

/**
 * ripgrep adapter for lexical search
 */
export class RipgrepAdapter extends CompetitorAdapter {
  async prepare(): Promise<void> {
    console.log(`⚡ Preparing ripgrep adapter`);
    await this.runWarmup();
    this.isInitialized = true;
  }

  async search(query: string, sla_ms: number): Promise<SearchResponse> {
    const startTime = Date.now();
    
    try {
      // Escape regex special characters for literal search
      const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      
      const command = [
        'rg',
        '--json',
        '--max-count', '50',
        '--context', '2',
        escapedQuery,
        this.config.corpus_path
      ];

      const { stdout, stderr } = await execAsync(command.join(' '), {
        timeout: sla_ms
      });

      const latency_ms = Date.now() - startTime;
      const within_sla = latency_ms <= sla_ms;

      const hits = this.parseRipgrepOutput(stdout);

      return { hits, latency_ms, within_sla };

    } catch (error) {
      const latency_ms = Date.now() - startTime;
      return {
        hits: [],
        latency_ms,
        within_sla: false,
        error: error.message
      };
    }
  }

  private parseRipgrepOutput(output: string): SearchHit[] {
    const hits: SearchHit[] = [];
    const lines = output.trim().split('\n');
    
    for (const line of lines) {
      try {
        const result = JSON.parse(line);
        
        if (result.type === 'match') {
          hits.push({
            file: result.data.path.text,
            line: result.data.line_number,
            column: result.data.submatches[0]?.start || 1,
            snippet: result.data.lines.text,
            score: 1.0 - hits.length * 0.01, // Simple ranking
            why_tag: 'exact'
          });
        }
      } catch {
        // Skip invalid JSON lines
      }
    }
    
    return hits;
  }

  async teardown(): Promise<void> {
    // No cleanup needed for ripgrep
  }

  protected async getSystemVersion(): Promise<string> {
    const { stdout } = await execAsync('rg --version');
    return stdout.split('\n')[0].split(' ')[1] || 'unknown';
  }

  protected async getBuildHash(): Promise<string | undefined> {
    return undefined;
  }
}

/**
 * Sourcegraph-class LSP adapter
 */
export class SourcegraphAdapter extends CompetitorAdapter {
  async prepare(): Promise<void> {
    console.log(`🏗️  Preparing Sourcegraph adapter`);
    // Implementation would depend on specific Sourcegraph setup
    await this.runWarmup();
    this.isInitialized = true;
  }

  async search(query: string, sla_ms: number): Promise<SearchResponse> {
    // Mock implementation - would integrate with actual Sourcegraph API
    const startTime = Date.now();
    
    await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
    
    const latency_ms = Date.now() - startTime;
    
    return {
      hits: [],
      latency_ms,
      within_sla: latency_ms <= sla_ms
    };
  }

  async teardown(): Promise<void> {}

  protected async getSystemVersion(): Promise<string> {
    return 'sourcegraph-mock-1.0.0';
  }

  protected async getBuildHash(): Promise<string | undefined> {
    return undefined;
  }
}

/**
 * Factory function to create appropriate adapter
 */
export function createAdapter(systemId: string, config: AdapterConfig): CompetitorAdapter {
  switch (systemId.toLowerCase()) {
    case 'lens':
      return new LensAdapter(config);
    case 'elasticsearch':
    case 'bm25':
      return new ElasticsearchAdapter(config);
    case 'ripgrep':
    case 'rg':
      return new RipgrepAdapter(config);
    case 'sourcegraph':
      return new SourcegraphAdapter(config);
    default:
      throw new Error(`Unknown system: ${systemId}`);
  }
}

/**
 * Adapter registry for managing multiple systems
 */
export class AdapterRegistry {
  private adapters = new Map<string, CompetitorAdapter>();

  async registerAdapter(systemId: string, config: AdapterConfig): Promise<void> {
    const adapter = createAdapter(systemId, { ...config, system_id: systemId });
    await adapter.prepare();
    this.adapters.set(systemId, adapter);
  }

  getAdapter(systemId: string): CompetitorAdapter | undefined {
    return this.adapters.get(systemId);
  }

  async teardownAll(): Promise<void> {
    for (const [systemId, adapter] of this.adapters) {
      try {
        await adapter.teardown();
      } catch (error) {
        console.error(`Failed to teardown ${systemId}: ${error}`);
      }
    }
    this.adapters.clear();
  }

  getSystemIds(): string[] {
    return Array.from(this.adapters.keys());
  }
}