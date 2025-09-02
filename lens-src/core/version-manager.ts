/**
 * Version management and compatibility checking for lens
 * Implements Phase A1 requirements for API and index versioning
 */

import { ApiVersion, IndexVersion, PolicyVersion, CompatibilityCheckRequest, CompatibilityCheckResponse } from '../types/api.js';

// Current server versions
export const SERVER_API_VERSION: ApiVersion = 'v1';
export const SERVER_INDEX_VERSION: IndexVersion = 'v1';
export const SERVER_POLICY_VERSION: PolicyVersion = 'v1';

// Version compatibility matrix
const COMPATIBILITY_MATRIX = {
  api: {
    'v1': ['v1'], // v1 API is compatible with v1 API only
  },
  index: {
    'v1': ['v1'], // v1 index is compatible with v1 index only
  },
  policy: {
    'v1': ['v1'], // v1 policy is compatible with v1 policy only
  }
} as const;

/**
 * Check compatibility between client and server versions
 */
export function checkCompatibility(
  clientApiVersion: ApiVersion,
  clientIndexVersion: IndexVersion,
  allowCompat: boolean = false,
  clientPolicyVersion?: PolicyVersion
): CompatibilityCheckResponse {
  const warnings: string[] = [];
  const errors: string[] = [];

  // Check API version compatibility
  const apiCompatible = COMPATIBILITY_MATRIX.api[SERVER_API_VERSION]?.includes(clientApiVersion) ?? false;
  if (!apiCompatible && !allowCompat) {
    errors.push(`API version mismatch: client ${clientApiVersion}, server ${SERVER_API_VERSION}`);
  } else if (!apiCompatible && allowCompat) {
    warnings.push(`API version mismatch allowed by --allow-compat flag: client ${clientApiVersion}, server ${SERVER_API_VERSION}`);
  }

  // Check index version compatibility
  const indexCompatible = COMPATIBILITY_MATRIX.index[SERVER_INDEX_VERSION]?.includes(clientIndexVersion) ?? false;
  if (!indexCompatible && !allowCompat) {
    errors.push(`Index version mismatch: client ${clientIndexVersion}, server ${SERVER_INDEX_VERSION}`);
  } else if (!indexCompatible && allowCompat) {
    warnings.push(`Index version mismatch allowed by --allow-compat flag: client ${clientIndexVersion}, server ${SERVER_INDEX_VERSION}`);
  }

  // Check policy version compatibility (if provided)
  let policyCompatible = true;
  if (clientPolicyVersion) {
    policyCompatible = COMPATIBILITY_MATRIX.policy[SERVER_POLICY_VERSION]?.includes(clientPolicyVersion) ?? false;
    if (!policyCompatible && !allowCompat) {
      errors.push(`Policy version mismatch: client ${clientPolicyVersion}, server ${SERVER_POLICY_VERSION}`);
    } else if (!policyCompatible && allowCompat) {
      warnings.push(`Policy version mismatch allowed by --allow-compat flag: client ${clientPolicyVersion}, server ${SERVER_POLICY_VERSION}`);
    }
  }

  const compatible = (apiCompatible && indexCompatible && policyCompatible) || allowCompat;

  return {
    compatible,
    api_version: clientApiVersion,
    index_version: clientIndexVersion,
    policy_version: clientPolicyVersion,
    server_api_version: SERVER_API_VERSION,
    server_index_version: SERVER_INDEX_VERSION,
    server_policy_version: SERVER_POLICY_VERSION,
    warnings: warnings.length > 0 ? warnings : undefined,
    errors: errors.length > 0 ? errors : undefined,
  };
}

/**
 * Validate that a request is compatible with server versions
 */
export function validateVersionCompatibility(
  apiVersion: ApiVersion,
  indexVersion: IndexVersion,
  allowCompat: boolean = false,
  policyVersion?: PolicyVersion
): void {
  const result = checkCompatibility(apiVersion, indexVersion, allowCompat, policyVersion);
  
  if (!result.compatible) {
    const errorMessage = result.errors?.join('; ') || 'Version compatibility check failed';
    throw new Error(`Version compatibility error: ${errorMessage}`);
  }
}

/**
 * Get version info for responses
 */
export function getVersionInfo() {
  return {
    api_version: SERVER_API_VERSION,
    index_version: SERVER_INDEX_VERSION,
    policy_version: SERVER_POLICY_VERSION,
  };
}