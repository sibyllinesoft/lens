/**
 * Configuration for data source switching between simulated and production
 * Implements Section 1 of TODO.md: Convert "simulated" → "real"
 */

export type DataSource = 'sim' | 'prod';

export interface DataSourceConfig {
  DATA_SOURCE: DataSource;
  LENS_ENDPOINTS: string[];
  AUTH_TOKEN?: string;
  SLA_MS: number;
  RETRY_COUNT: number;
  CANCEL_ON_FIRST: boolean;
}

export interface LensEndpointConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  headers: Record<string, string>;
}

// Environment variable parsing with defaults
export function getDataSourceConfig(): DataSourceConfig {
  const DATA_SOURCE = (process.env.DATA_SOURCE || 'sim') as DataSource;
  const LENS_ENDPOINTS = process.env.LENS_ENDPOINTS 
    ? JSON.parse(process.env.LENS_ENDPOINTS)
    : ['http://localhost:3000'];
  const AUTH_TOKEN = process.env.AUTH_TOKEN;
  const SLA_MS = parseInt(process.env.SLA_MS || '150', 10);
  const RETRY_COUNT = parseInt(process.env.RETRY_COUNT || '3', 10);
  const CANCEL_ON_FIRST = process.env.CANCEL_ON_FIRST === 'true';

  // Validation
  if (!['sim', 'prod'].includes(DATA_SOURCE)) {
    throw new Error(`Invalid DATA_SOURCE: ${DATA_SOURCE}. Must be 'sim' or 'prod'`);
  }

  if (DATA_SOURCE === 'prod' && (!LENS_ENDPOINTS || LENS_ENDPOINTS.length === 0)) {
    throw new Error('LENS_ENDPOINTS required when DATA_SOURCE=prod');
  }

  if (SLA_MS <= 0 || SLA_MS > 10000) {
    throw new Error(`Invalid SLA_MS: ${SLA_MS}. Must be between 1-10000ms`);
  }

  return {
    DATA_SOURCE,
    LENS_ENDPOINTS,
    AUTH_TOKEN,
    SLA_MS,
    RETRY_COUNT,
    CANCEL_ON_FIRST
  };
}

// Create lens endpoint configurations
export function createLensEndpointConfigs(config: DataSourceConfig): LensEndpointConfig[] {
  return config.LENS_ENDPOINTS.map(baseUrl => ({
    baseUrl,
    timeout: config.SLA_MS,
    retries: config.RETRY_COUNT,
    headers: {
      'Content-Type': 'application/json',
      ...(config.AUTH_TOKEN && { 'Authorization': `Bearer ${config.AUTH_TOKEN}` })
    }
  }));
}

// Configuration validation
export function validateDataSourceConfig(config: DataSourceConfig): void {
  if (config.DATA_SOURCE === 'prod') {
    if (!config.LENS_ENDPOINTS?.length) {
      throw new Error('Production mode requires at least one LENS_ENDPOINT');
    }
    
    // Validate endpoint URLs
    config.LENS_ENDPOINTS.forEach(endpoint => {
      try {
        new URL(endpoint);
      } catch (error) {
        throw new Error(`Invalid LENS_ENDPOINT URL: ${endpoint}`);
      }
    });
  }

  if (config.SLA_MS < 10 || config.SLA_MS > 30000) {
    throw new Error(`SLA_MS must be between 10-30000ms, got ${config.SLA_MS}`);
  }

  if (config.RETRY_COUNT < 0 || config.RETRY_COUNT > 10) {
    throw new Error(`RETRY_COUNT must be between 0-10, got ${config.RETRY_COUNT}`);
  }
}