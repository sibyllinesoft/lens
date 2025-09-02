/**
 * Compatibility checker for lens
 * Implements Phase A1.2 - compatibility checking against nightly bundles
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { LensTracer } from '../telemetry/tracer.js';
import { SERVER_API_VERSION, SERVER_INDEX_VERSION, SERVER_POLICY_VERSION } from './version-manager.js';

export interface NightlyBundle {
  bundle_id: string;
  created_at: string;
  api_version: string;
  index_version: string;
  policy_version: string;
  schema_hash: string;
  index_format_version: number;
}

export interface CompatibilityReport {
  compatible: boolean;
  current_version: {
    api_version: string;
    index_version: string;
    policy_version: string;
  };
  bundles_checked: NightlyBundle[];
  compatibility_matrix: {
    bundle_id: string;
    api_compatible: boolean;
    index_compatible: boolean;
    policy_compatible: boolean;
    issues: string[];
  }[];
  overall_status: 'compatible' | 'incompatible' | 'partial';
  warnings: string[];
  errors: string[];
}

/**
 * Check compatibility against the last two nightly bundles
 */
export async function checkBundleCompatibility(
  bundlesPath: string = './nightly-bundles',
  allowCompat: boolean = false
): Promise<CompatibilityReport> {
  const span = LensTracer.createChildSpan('compat_check_bundles');
  
  try {
    // Load the last two nightly bundles
    const bundles = await loadRecentBundles(bundlesPath, 2);
    
    if (bundles.length === 0) {
      throw new Error('No nightly bundles found for compatibility checking');
    }
    
    const report: CompatibilityReport = {
      compatible: true,
      current_version: {
        api_version: SERVER_API_VERSION,
        index_version: SERVER_INDEX_VERSION,
        policy_version: SERVER_POLICY_VERSION,
      },
      bundles_checked: bundles,
      compatibility_matrix: [],
      overall_status: 'compatible',
      warnings: [],
      errors: [],
    };
    
    // Check compatibility for each bundle
    for (const bundle of bundles) {
      const bundleCheck = checkSingleBundleCompatibility(bundle, allowCompat);
      report.compatibility_matrix.push(bundleCheck);
      
      // Aggregate issues
      if (!bundleCheck.api_compatible || !bundleCheck.index_compatible || !bundleCheck.policy_compatible) {
        report.compatible = false;
        report.errors.push(...bundleCheck.issues);
      }
    }
    
    // Determine overall status
    const compatibleCount = report.compatibility_matrix.filter(m => 
      m.api_compatible && m.index_compatible && m.policy_compatible
    ).length;
    
    if (compatibleCount === report.compatibility_matrix.length) {
      report.overall_status = 'compatible';
    } else if (compatibleCount > 0) {
      report.overall_status = 'partial';
    } else {
      report.overall_status = 'incompatible';
    }
    
    // Add warnings for partial compatibility
    if (report.overall_status === 'partial') {
      report.warnings.push('Some bundles are incompatible but compatibility can be maintained with --allow-compat flag');
    }
    
    span.setAttributes({
      success: true,
      bundles_checked: bundles.length,
      compatible: report.compatible,
      overall_status: report.overall_status,
    });
    
    return report;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    throw new Error(`Bundle compatibility check failed: ${errorMsg}`);
  } finally {
    span.end();
  }
}

/**
 * Load the most recent nightly bundles
 */
async function loadRecentBundles(bundlesPath: string, count: number): Promise<NightlyBundle[]> {
  const bundles: NightlyBundle[] = [];
  
  try {
    await fs.access(bundlesPath);
    const files = await fs.readdir(bundlesPath);
    const bundleFiles = files
      .filter(f => f.endsWith('.bundle.json'))
      .sort()
      .slice(-count); // Get the most recent ones
    
    for (const bundleFile of bundleFiles) {
      try {
        const bundlePath = path.join(bundlesPath, bundleFile);
        const bundleData = await fs.readFile(bundlePath, 'utf-8');
        const bundle = JSON.parse(bundleData) as NightlyBundle;
        
        // Validate bundle structure
        if (bundle.bundle_id && bundle.api_version && bundle.index_version && bundle.policy_version) {
          bundles.push(bundle);
        } else {
          console.warn(`Invalid bundle file: ${bundleFile}`);
        }
      } catch (error) {
        console.warn(`Failed to load bundle ${bundleFile}:`, error);
      }
    }
  } catch (error) {
    // Bundles directory doesn't exist or is inaccessible
    console.log(`Nightly bundles directory not found at ${bundlesPath}`);
  }
  
  return bundles;
}

/**
 * Check compatibility for a single bundle
 */
function checkSingleBundleCompatibility(
  bundle: NightlyBundle, 
  allowCompat: boolean
): {
  bundle_id: string;
  api_compatible: boolean;
  index_compatible: boolean;
  policy_compatible: boolean;
  issues: string[];
} {
  const issues: string[] = [];
  
  // Check API version compatibility
  const apiCompatible = bundle.api_version === SERVER_API_VERSION;
  if (!apiCompatible && !allowCompat) {
    issues.push(`API version mismatch: bundle ${bundle.api_version} vs server ${SERVER_API_VERSION}`);
  }
  
  // Check index version compatibility
  const indexCompatible = bundle.index_version === SERVER_INDEX_VERSION;
  if (!indexCompatible && !allowCompat) {
    issues.push(`Index version mismatch: bundle ${bundle.index_version} vs server ${SERVER_INDEX_VERSION}`);
  }
  
  // Check policy version compatibility
  const policyCompatible = bundle.policy_version === SERVER_POLICY_VERSION;
  if (!policyCompatible && !allowCompat) {
    issues.push(`Policy version mismatch: bundle ${bundle.policy_version} vs server ${SERVER_POLICY_VERSION}`);
  }
  
  return {
    bundle_id: bundle.bundle_id,
    api_compatible: apiCompatible || allowCompat,
    index_compatible: indexCompatible || allowCompat,
    policy_compatible: policyCompatible || allowCompat,
    issues,
  };
}

/**
 * Generate a nightly bundle for testing (development utility)
 */
export async function generateNightlyBundle(
  bundlesPath: string = './nightly-bundles',
  bundleId?: string
): Promise<NightlyBundle> {
  const span = LensTracer.createChildSpan('generate_nightly_bundle');
  
  try {
    // Ensure bundles directory exists
    await fs.mkdir(bundlesPath, { recursive: true });
    
    // Generate bundle metadata
    const bundle: NightlyBundle = {
      bundle_id: bundleId || `nightly-${Date.now()}`,
      created_at: new Date().toISOString(),
      api_version: SERVER_API_VERSION,
      index_version: SERVER_INDEX_VERSION,
      policy_version: SERVER_POLICY_VERSION,
      schema_hash: generateSchemaHash(),
      index_format_version: 1,
    };
    
    // Write bundle to disk
    const bundlePath = path.join(bundlesPath, `${bundle.bundle_id}.bundle.json`);
    await fs.writeFile(bundlePath, JSON.stringify(bundle, null, 2));
    
    span.setAttributes({
      success: true,
      bundle_id: bundle.bundle_id,
      bundle_path: bundlePath,
    });
    
    console.log(`Generated nightly bundle: ${bundle.bundle_id}`);
    return bundle;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    throw new Error(`Failed to generate nightly bundle: ${errorMsg}`);
  } finally {
    span.end();
  }
}

/**
 * Generate a simple schema hash for the current configuration
 */
function generateSchemaHash(): string {
  const schemaData = {
    api_version: SERVER_API_VERSION,
    index_version: SERVER_INDEX_VERSION,
    policy_version: SERVER_POLICY_VERSION,
    timestamp: Date.now(),
  };
  
  // Simple hash based on JSON string (in production, use a proper crypto hash)
  let hash = 0;
  const str = JSON.stringify(schemaData);
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(16);
}

/**
 * CLI integration - fail CI on mismatch unless --allow-compat
 */
export async function runCompatibilityCheck(
  bundlesPath?: string, 
  allowCompat: boolean = false
): Promise<void> {
  console.log('🔍 Running compatibility check against nightly bundles...');
  
  try {
    const report = await checkBundleCompatibility(bundlesPath, allowCompat);
    
    // Print results
    console.log(`📊 Compatibility Report:`);
    console.log(`   Status: ${report.overall_status.toUpperCase()}`);
    console.log(`   Bundles checked: ${report.bundles_checked.length}`);
    console.log(`   Compatible: ${report.compatible}`);
    
    if (report.warnings.length > 0) {
      console.log(`⚠️  Warnings:`);
      for (const warning of report.warnings) {
        console.log(`   - ${warning}`);
      }
    }
    
    if (report.errors.length > 0) {
      console.log(`❌ Errors:`);
      for (const error of report.errors) {
        console.log(`   - ${error}`);
      }
    }
    
    // Write report to disk
    const reportPath = './compat_report.json';
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`📝 Report saved to: ${reportPath}`);
    
    // Exit with error if incompatible (unless --allow-compat)
    if (!report.compatible && !allowCompat) {
      console.log('💥 Compatibility check failed! Use --allow-compat to override.');
      process.exit(1);
    } else if (!report.compatible && allowCompat) {
      console.log('⚠️  Compatibility issues found but allowed by --allow-compat flag');
    } else {
      console.log('✅ All compatibility checks passed!');
    }
    
  } catch (error) {
    console.error('💥 Compatibility check failed:', error);
    process.exit(1);
  }
}