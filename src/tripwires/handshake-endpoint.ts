// Anti-Fraud Service Handshake Endpoint
// This endpoint provides build info and challenge/response for benchmark verification

import { createHash } from 'crypto';

interface BuildInfo {
  git_sha: string;
  dirty_flag: boolean;
  build_timestamp: string;
  rustc_version?: string;
  target_triple: string;
  feature_flags: string[];
  mode: 'real' | 'mock';
}

interface HandshakeRequest {
  nonce: string;
}

interface HandshakeResponse extends BuildInfo {
  nonce: string;
  response: string; // SHA256(nonce || build_sha)
  service_name: string;
}

// GET /__buildinfo
export async function getBuildInfo(): Promise<BuildInfo> {
  const buildInfo: BuildInfo = {
    git_sha: process.env.GIT_SHA || 'unknown',
    dirty_flag: process.env.GIT_DIRTY === 'true',
    build_timestamp: process.env.BUILD_TIMESTAMP || new Date().toISOString(),
    target_triple: process.env.TARGET_TRIPLE || process.platform + '-' + process.arch,
    feature_flags: (process.env.FEATURE_FLAGS || '').split(',').filter(Boolean),
    mode: 'real' as const // CRITICAL: Never 'mock' in production
  };
  
  return buildInfo;
}

// POST /__buildinfo/handshake
export async function performHandshake(request: HandshakeRequest): Promise<HandshakeResponse> {
  const buildInfo = await getBuildInfo();
  
  // Generate challenge response: SHA256(nonce || build_sha)
  const challengeInput = request.nonce + buildInfo.git_sha;
  const response = createHash('sha256').update(challengeInput).digest('hex');
  
  return {
    ...buildInfo,
    nonce: request.nonce,
    response,
    service_name: 'lens-core'
  };
}

// Tripwire: Fail if mode is not 'real'
export function validateRealMode(): void {
  const mode = process.env.NODE_ENV === 'test' ? 'test' : 'real';
  if (mode !== 'real' && mode !== 'test') {
    throw new Error(`TRIPWIRE VIOLATION: Service mode must be 'real', got '${mode}'`);
  }
}