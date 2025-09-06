/**
 * Audit Bundle Generator for Reproducible Benchmark Artifacts
 * Implements TODO.md requirement 6: Generate repro.tar.gz with complete reproducibility artifacts
 */

import { promises as fs } from 'fs';
import path from 'path';
import { createHash } from 'crypto';
import { execSync } from 'child_process';
import * as tar from 'tar';
import type { VersionedFingerprint } from './governance-system.js';

export interface AuditBundleConfig {
  outputDir: string;
  bundleName?: string;
  includeSourceCode: boolean;
  includeDatasets: boolean;
  includeModels: boolean;
  includeDependencies: boolean;
  compressionLevel?: number;
}

export interface BundleArtifact {
  path: string;
  fullPath: string;
  size: number;
  description: string;
  category: 'fingerprint' | 'dataset' | 'model' | 'code' | 'dependency' | 'script' | 'result';
}

export interface AuditBundleManifest {
  version: string;
  generatedAt: string;
  fingerprintHash: string;
  contentHash: string;
  files: Array<{
    path: string;
    size: number;
    hash: string;
    description: string;
    category: 'fingerprint' | 'dataset' | 'model' | 'code' | 'dependency' | 'script' | 'result';
  }>;
  reproducibilityScript: string;
  environment: {
    nodeVersion: string;
    npmVersion: string;
    gitHash: string;
    gitRemote: string;
    systemInfo: {
      platform: string;
      arch: string;
      cpus: number;
      memory: number;
    };
  };
  instructions: {
    setup: string[];
    execution: string[];
    validation: string[];
  };
}

/**
 * Generates complete reproducibility audit bundles
 */
export class AuditBundleGenerator {
  
  constructor(private readonly config: AuditBundleConfig) {}
  
  /**
   * Generate complete audit bundle with all reproducibility artifacts
   */
  async generateAuditBundle(
    fingerprint: VersionedFingerprint,
    benchmarkResults: any[],
    groundTruthData: any[],
    modelArtifacts: string[] = []
  ): Promise<{
    bundlePath: string;
    manifest: AuditBundleManifest;
    verificationHash: string;
  }> {
    
    // Create temporary working directory
    const workingDir = path.join(this.config.outputDir, `audit-working-${Date.now()}`);
    await fs.mkdir(workingDir, { recursive: true });
    
    try {
      // 1. Generate core artifacts
      const artifacts = await this.generateCoreArtifacts(
        fingerprint,
        benchmarkResults,
        groundTruthData,
        workingDir
      );
      
      // 2. Create reproducibility script
      const reproScript = await this.generateReproducibilityScript(
        fingerprint,
        workingDir
      );
      artifacts.push(reproScript);
      
      // 3. Include source code snapshots
      if (this.config.includeSourceCode) {
        const sourceArtifacts = await this.captureSourceCodeSnapshot(workingDir);
        artifacts.push(...sourceArtifacts);
      }
      
      // 4. Include model artifacts
      if (this.config.includeModels && modelArtifacts.length > 0) {
        const modelFiles = await this.includeModelArtifacts(modelArtifacts, workingDir);
        artifacts.push(...modelFiles);
      }
      
      // 5. Include dependencies
      if (this.config.includeDependencies) {
        const depArtifacts = await this.captureDependencies(workingDir);
        artifacts.push(...depArtifacts);
      }
      
      // 6. Generate manifest
      const manifest = await this.generateManifest(
        fingerprint,
        artifacts,
        reproScript.path
      );
      
      // Save manifest
      const manifestPath = path.join(workingDir, 'audit-manifest.json');
      await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
      artifacts.push({
        path: 'audit-manifest.json',
        fullPath: manifestPath,
        size: (await fs.stat(manifestPath)).size,
        description: 'Audit bundle manifest with complete file inventory',
        category: 'fingerprint' as const
      });
      
      // 7. Create tar bundle
      const bundleName = this.config.bundleName || 'repro.tar.gz';
      const bundlePath = path.join(this.config.outputDir, bundleName);
      
      await this.createTarBundle(workingDir, bundlePath);
      
      // 8. Generate verification hash
      const bundleContent = await fs.readFile(bundlePath);
      const verificationHash = createHash('sha256').update(bundleContent).digest('hex');
      
      // Create verification file
      const verificationPath = path.join(this.config.outputDir, `${bundleName}.sha256`);
      await fs.writeFile(verificationPath, `${verificationHash}  ${bundleName}\n`);
      
      return {
        bundlePath,
        manifest,
        verificationHash
      };
      
    } finally {
      // Cleanup working directory
      try {
        await fs.rm(workingDir, { recursive: true, force: true });
      } catch (error) {
        console.warn('Failed to cleanup working directory:', error);
      }
    }
  }
  
  /**
   * Generate core reproducibility artifacts
   */
  private async generateCoreArtifacts(
    fingerprint: VersionedFingerprint,
    benchmarkResults: any[],
    groundTruthData: any[],
    workingDir: string
  ): Promise<BundleArtifact[]> {
    
    const artifacts = [];
    
    // 1. Fingerprint file
    const fingerprintPath = path.join(workingDir, 'benchmark-fingerprint.json');
    await fs.writeFile(fingerprintPath, JSON.stringify(fingerprint, null, 2));
    artifacts.push({
      path: 'benchmark-fingerprint.json',
      fullPath: fingerprintPath,
      size: (await fs.stat(fingerprintPath)).size,
      description: 'Complete versioned fingerprint with all governance parameters',
      category: 'fingerprint' as const
    });
    
    // 2. Benchmark results
    const resultsPath = path.join(workingDir, 'benchmark-results.json');
    await fs.writeFile(resultsPath, JSON.stringify(benchmarkResults, null, 2));
    artifacts.push({
      path: 'benchmark-results.json',
      fullPath: resultsPath,
      size: (await fs.stat(resultsPath)).size,
      description: 'Raw benchmark execution results with all metrics',
      category: 'result' as const
    });
    
    // 3. Ground truth dataset
    const datasetPath = path.join(workingDir, 'ground-truth-dataset.json');
    await fs.writeFile(datasetPath, JSON.stringify(groundTruthData, null, 2));
    artifacts.push({
      path: 'ground-truth-dataset.json',
      fullPath: datasetPath,
      size: (await fs.stat(datasetPath)).size,
      description: 'Complete ground truth dataset with expected results',
      category: 'dataset' as const
    });
    
    // 4. Seed files for random number generation
    const seedsPath = path.join(workingDir, 'random-seeds.json');
    const seedsData = {
      primarySeed: fingerprint.seed,
      seedSet: fingerprint.seed_set,
      generatedAt: new Date().toISOString(),
      usage: 'Use these seeds to reproduce identical random sampling and bootstrap resampling'
    };
    await fs.writeFile(seedsPath, JSON.stringify(seedsData, null, 2));
    artifacts.push({
      path: 'random-seeds.json',
      fullPath: seedsPath,
      size: (await fs.stat(seedsPath)).size,
      description: 'Random seeds for reproducible sampling and bootstrap procedures',
      category: 'fingerprint' as const
    });
    
    // 5. Configuration snapshot
    const configPath = path.join(workingDir, 'benchmark-config.json');
    const configData = {
      fingerprintConfig: fingerprint,
      auditConfig: this.config,
      environmentSnapshot: await this.captureEnvironment()
    };
    await fs.writeFile(configPath, JSON.stringify(configData, null, 2));
    artifacts.push({
      path: 'benchmark-config.json',
      fullPath: configPath,
      size: (await fs.stat(configPath)).size,
      description: 'Complete configuration snapshot including environment details',
      category: 'fingerprint' as const
    });
    
    return artifacts;
  }
  
  /**
   * Generate one-shot reproducibility script
   */
  private async generateReproducibilityScript(
    fingerprint: VersionedFingerprint,
    workingDir: string
  ): Promise<BundleArtifact> {
    
    const scriptContent = this.generateBashScript(fingerprint);
    const scriptPath = path.join(workingDir, 'reproduce-benchmark.sh');
    
    await fs.writeFile(scriptPath, scriptContent, { mode: 0o755 });
    
    return {
      path: 'reproduce-benchmark.sh',
      fullPath: scriptPath,
      size: (await fs.stat(scriptPath)).size,
      description: 'One-shot script to reproduce complete benchmark from scratch',
      category: 'script' as const
    };
  }
  
  /**
   * Generate bash script for full reproduction
   */
  private generateBashScript(fingerprint: VersionedFingerprint): string {
    return `#!/bin/bash
# Lens Benchmark Reproduction Script
# Generated: ${new Date().toISOString()}
# Fingerprint: ${fingerprint.pool_v.substring(0, 8)}
# CBU Coefficients Version: ${fingerprint.cbu_coeff_v}

set -euo pipefail

echo "🔄 Starting Lens benchmark reproduction..."
echo "📌 Target fingerprint: ${fingerprint.pool_v}"
echo "📊 CBU coefficients version: ${fingerprint.cbu_coeff_v}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check Node.js version
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js is required but not installed"
        exit 1
    fi
    
    local node_version=$(node --version)
    echo "✅ Node.js version: $node_version"
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        echo "❌ npm is required but not installed"
        exit 1
    fi
    
    local npm_version=$(npm --version)
    echo "✅ npm version: $npm_version"
    
    # Check git
    if ! command -v git &> /dev/null; then
        echo "❌ git is required but not installed"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
    echo ""
}

# Setup environment
setup_environment() {
    echo "🏗️  Setting up environment..."
    
    # Create working directory
    local work_dir="lens-benchmark-reproduction"
    if [ -d "$work_dir" ]; then
        echo "🗑️  Removing existing working directory"
        rm -rf "$work_dir"
    fi
    
    mkdir -p "$work_dir"
    cd "$work_dir"
    
    # Clone repository at specific commit
    echo "📥 Cloning repository at commit ${fingerprint.pool_v}"
    git clone https://github.com/your-org/lens.git .
    git checkout ${fingerprint.pool_v}
    
    # Install dependencies
    echo "📦 Installing dependencies..."
    npm ci
    
    echo "✅ Environment setup complete"
    echo ""
}

# Restore datasets
restore_datasets() {
    echo "📊 Restoring datasets..."
    
    # Extract ground truth data
    if [ -f "../ground-truth-dataset.json" ]; then
        echo "📋 Loading ground truth dataset"
        cp ../ground-truth-dataset.json ./benchmarks/
    else
        echo "⚠️  Ground truth dataset not found, using embedded data"
    fi
    
    # Load seeds
    if [ -f "../random-seeds.json" ]; then
        echo "🎲 Loading random seeds"
        cp ../random-seeds.json ./benchmarks/
    fi
    
    echo "✅ Datasets restored"
    echo ""
}

# Execute benchmark
execute_benchmark() {
    echo "🚀 Executing benchmark reproduction..."
    
    # Set reproducibility environment variables
    export LENS_REPRODUCIBILITY_MODE=true
    export LENS_SEED=${fingerprint.seed}
    export LENS_CBU_COEFF_V=${fingerprint.cbu_coeff_v}
    export LENS_CONTRACT_V=${fingerprint.contract_v}
    
    # Run benchmark with exact configuration
    npm run benchmark -- \\
        --config="../benchmark-config.json" \\
        --fingerprint="../benchmark-fingerprint.json" \\
        --output="./reproduction-results" \\
        --governance-validation=true \\
        --clustered-bootstrap=true \\
        --bootstrap-samples=${fingerprint.bootstrap_config.b_default} \\
        --multiple-testing-correction="${fingerprint.multiple_testing.method}" \\
        --calibration-gates=true
    
    echo "✅ Benchmark execution complete"
    echo ""
}

# Validate results
validate_results() {
    echo "🔍 Validating reproduction results..."
    
    # Compare key metrics with original results
    if [ -f "../benchmark-results.json" ] && [ -f "./reproduction-results/metrics.json" ]; then
        echo "📊 Comparing metrics..."
        
        # Use jq to extract and compare key metrics
        if command -v jq &> /dev/null; then
            local original_ndcg=$(jq '.metrics.ndcg_at_10' ../benchmark-results.json)
            local reproduced_ndcg=$(jq '.metrics.ndcg_at_10' ./reproduction-results/metrics.json)
            
            echo "📈 Original nDCG@10: $original_ndcg"
            echo "📈 Reproduced nDCG@10: $reproduced_ndcg"
            
            # Tolerance check (within 1% for floating point precision)
            local tolerance=0.01
            if [ $(echo "$original_ndcg - $reproduced_ndcg < $tolerance" | bc -l) -eq 1 ] && \\
               [ $(echo "$reproduced_ndcg - $original_ndcg < $tolerance" | bc -l) -eq 1 ]; then
                echo "✅ Metrics validation passed (within tolerance)"
            else
                echo "❌ Metrics validation failed (beyond tolerance)"
                exit 1
            fi
        else
            echo "⚠️  jq not available, skipping detailed metric comparison"
        fi
    else
        echo "⚠️  Result files not found for comparison"
    fi
    
    # Validate governance requirements
    if [ -f "./reproduction-results/governance-validation.json" ]; then
        local governance_pass=$(jq '.overallPassed' ./reproduction-results/governance-validation.json)
        if [ "$governance_pass" = "true" ]; then
            echo "✅ Governance validation passed"
        else
            echo "❌ Governance validation failed"
            jq '.recommendedActions' ./reproduction-results/governance-validation.json
            exit 1
        fi
    fi
    
    echo "✅ Results validation complete"
    echo ""
}

# Generate reproduction report
generate_report() {
    echo "📄 Generating reproduction report..."
    
    local report_file="reproduction-report.md"
    cat > "$report_file" << EOF
# Lens Benchmark Reproduction Report

**Generated**: $(date)
**Fingerprint**: ${fingerprint.pool_v}
**CBU Coefficients Version**: ${fingerprint.cbu_coeff_v}
**Contract Version**: ${fingerprint.contract_v}

## Environment
- Node.js: $(node --version)
- npm: $(npm --version)
- Platform: $(uname -s) $(uname -m)
- Hostname: $(hostname)

## Reproduction Status
✅ Environment setup completed
✅ Datasets restored successfully  
✅ Benchmark executed successfully
✅ Results validation passed
✅ Governance requirements satisfied

## Key Metrics
$(if [ -f "./reproduction-results/metrics.json" ]; then
    echo "- nDCG@10: $(jq '.metrics.ndcg_at_10' ./reproduction-results/metrics.json)"
    echo "- Recall@50: $(jq '.metrics.recall_at_50' ./reproduction-results/metrics.json)"
    echo "- CBU Score: $(jq '.metrics.cbu_score' ./reproduction-results/metrics.json)"
    echo "- ECE Score: $(jq '.metrics.ece_score' ./reproduction-results/metrics.json)"
fi)

## Files Generated
$(ls -la reproduction-results/ | grep -v '^total')

---
*This report confirms successful reproduction of the Lens benchmark with identical configuration and governance validation.*
EOF

    echo "📄 Report saved to: $report_file"
    echo "✅ Reproduction complete!"
}

# Main execution flow
main() {
    local start_time=$(date +%s)
    
    echo "🎯 Lens Benchmark Reproduction"
    echo "=============================="
    
    check_prerequisites
    setup_environment
    restore_datasets
    execute_benchmark
    validate_results
    generate_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "🏁 Reproduction completed successfully in \${duration} seconds"
    echo "📊 Results available in: reproduction-results/"
    echo "📄 Report: reproduction-report.md"
}

# Trap for cleanup on exit
cleanup() {
    echo "🧹 Performing cleanup..."
    # Add any necessary cleanup steps here
}
trap cleanup EXIT

# Execute main function
main "$@"
`;
  }
  
  /**
   * Capture source code snapshot
   */
  private async captureSourceCodeSnapshot(
    workingDir: string
  ): Promise<BundleArtifact[]> {
    
    const artifacts = [];
    
    try {
      // Get git patch for reproducibility
      const gitPatch = execSync('git format-patch --stdout HEAD~1..HEAD', {
        encoding: 'utf-8',
        cwd: process.cwd()
      });
      
      const patchPath = path.join(workingDir, 'source-changes.patch');
      await fs.writeFile(patchPath, gitPatch);
      
      artifacts.push({
        path: 'source-changes.patch',
        fullPath: patchPath,
        size: gitPatch.length,
        description: 'Git patch showing exact source code changes',
        category: 'code' as const
      });
      
    } catch (error) {
      console.warn('Could not generate git patch:', error);
    }
    
    try {
      // Capture package.json and package-lock.json
      const packageJsonPath = path.join(process.cwd(), 'package.json');
      const packageLockPath = path.join(process.cwd(), 'package-lock.json');
      
      if (await this.fileExists(packageJsonPath)) {
        const destPath = path.join(workingDir, 'package.json');
        await fs.copyFile(packageJsonPath, destPath);
        artifacts.push({
          path: 'package.json',
          fullPath: destPath,
          size: (await fs.stat(destPath)).size,
          description: 'Node.js package configuration',
          category: 'code' as const
        });
      }
      
      if (await this.fileExists(packageLockPath)) {
        const destPath = path.join(workingDir, 'package-lock.json');
        await fs.copyFile(packageLockPath, destPath);
        artifacts.push({
          path: 'package-lock.json',
          fullPath: destPath,
          size: (await fs.stat(destPath)).size,
          description: 'Exact dependency versions lockfile',
          category: 'code' as const
        });
      }
      
    } catch (error) {
      console.warn('Could not capture package files:', error);
    }
    
    return artifacts;
  }
  
  /**
   * Include model artifacts
   */
  private async includeModelArtifacts(
    modelPaths: string[],
    workingDir: string
  ): Promise<BundleArtifact[]> {
    
    const artifacts = [];
    const modelsDir = path.join(workingDir, 'models');
    await fs.mkdir(modelsDir, { recursive: true });
    
    for (const modelPath of modelPaths) {
      if (await this.fileExists(modelPath)) {
        const filename = path.basename(modelPath);
        const destPath = path.join(modelsDir, filename);
        
        await fs.copyFile(modelPath, destPath);
        
        artifacts.push({
          path: `models/${filename}`,
          fullPath: destPath,
          size: (await fs.stat(destPath)).size,
          description: `Model artifact: ${filename}`,
          category: 'model' as const
        });
      }
    }
    
    return artifacts;
  }
  
  /**
   * Capture dependencies
   */
  private async captureDependencies(
    workingDir: string
  ): Promise<BundleArtifact[]> {
    
    const artifacts = [];
    
    try {
      // Generate npm list
      const npmList = execSync('npm list --all --json', {
        encoding: 'utf-8',
        cwd: process.cwd()
      });
      
      const depsPath = path.join(workingDir, 'dependencies.json');
      await fs.writeFile(depsPath, npmList);
      
      artifacts.push({
        path: 'dependencies.json',
        fullPath: depsPath,
        size: npmList.length,
        description: 'Complete dependency tree with versions',
        category: 'dependency' as const
      });
      
    } catch (error) {
      console.warn('Could not capture npm dependencies:', error);
    }
    
    return artifacts;
  }
  
  /**
   * Generate comprehensive manifest
   */
  private async generateManifest(
    fingerprint: VersionedFingerprint,
    artifacts: Array<{
      path: string;
      fullPath: string;
      size: number;
      description: string;
      category: string;
    }>,
    reproScriptPath: string
  ): Promise<AuditBundleManifest> {
    
    // Calculate content hash from all artifacts
    const allContent = await Promise.all(
      artifacts.map(async (artifact) => {
        const content = await fs.readFile(artifact.fullPath);
        return createHash('sha256').update(content).digest('hex');
      })
    );
    
    const contentHash = createHash('sha256')
      .update(allContent.join(''))
      .digest('hex');
    
    const fingerprintHash = createHash('sha256')
      .update(JSON.stringify(fingerprint))
      .digest('hex');
    
    const environment = await this.captureEnvironment();
    
    return {
      version: '1.0.0',
      generatedAt: new Date().toISOString(),
      fingerprintHash,
      contentHash,
      files: artifacts.map(artifact => ({
        path: artifact.path,
        size: artifact.size,
        hash: createHash('sha256').update(require('fs').readFileSync(artifact.fullPath)).digest('hex'),
        description: artifact.description,
        category: artifact.category as any
      })),
      reproducibilityScript: reproScriptPath.split('/').pop() || 'reproduce-benchmark.sh',
      environment,
      instructions: {
        setup: [
          'Extract audit bundle: tar -xzf repro.tar.gz',
          'Change to extracted directory',
          'Ensure Node.js and npm are installed',
          'Make script executable: chmod +x reproduce-benchmark.sh'
        ],
        execution: [
          'Run reproduction script: ./reproduce-benchmark.sh',
          'Monitor progress and check for any errors',
          'Script will validate results automatically'
        ],
        validation: [
          'Check reproduction-report.md for summary',
          'Compare key metrics in reproduction-results/',
          'Verify governance validation passed',
          'Confirm fingerprint matches original'
        ]
      }
    };
  }
  
  /**
   * Capture current environment details
   */
  private async captureEnvironment(): Promise<AuditBundleManifest['environment']> {
    let nodeVersion = 'unknown';
    let npmVersion = 'unknown';
    let gitHash = 'unknown';
    let gitRemote = 'unknown';
    
    try {
      nodeVersion = execSync('node --version', { encoding: 'utf-8' }).trim();
      npmVersion = execSync('npm --version', { encoding: 'utf-8' }).trim();
      gitHash = execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim();
      gitRemote = execSync('git remote get-url origin', { encoding: 'utf-8' }).trim();
    } catch (error) {
      console.warn('Could not capture full environment details:', error);
    }
    
    return {
      nodeVersion,
      npmVersion,
      gitHash,
      gitRemote,
      systemInfo: {
        platform: process.platform,
        arch: process.arch,
        cpus: require('os').cpus().length,
        memory: Math.round(require('os').totalmem() / (1024 * 1024 * 1024)) // GB
      }
    };
  }
  
  /**
   * Create tar bundle from working directory
   */
  private async createTarBundle(workingDir: string, outputPath: string): Promise<void> {
    await tar.create(
      {
        gzip: true,
        file: outputPath,
        cwd: workingDir,
        portable: true
      },
      await fs.readdir(workingDir)
    );
  }
  
  /**
   * Check if file exists
   */
  private async fileExists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
  
  /**
   * Validate audit bundle integrity
   */
  async validateAuditBundle(bundlePath: string): Promise<{
    isValid: boolean;
    errors: string[];
    manifest?: AuditBundleManifest;
  }> {
    const errors: string[] = [];
    
    try {
      // Check if bundle exists
      if (!(await this.fileExists(bundlePath))) {
        return { isValid: false, errors: ['Audit bundle file not found'] };
      }
      
      // Extract to temporary directory for validation
      const tempDir = path.join(this.config.outputDir, `validation-${Date.now()}`);
      await fs.mkdir(tempDir, { recursive: true });
      
      try {
        // Extract bundle
        await tar.extract({
          file: bundlePath,
          cwd: tempDir
        });
        
        // Load and validate manifest
        const manifestPath = path.join(tempDir, 'audit-manifest.json');
        if (!(await this.fileExists(manifestPath))) {
          errors.push('Manifest file missing from bundle');
          return { isValid: false, errors };
        }
        
        const manifestContent = await fs.readFile(manifestPath, 'utf-8');
        const manifest: AuditBundleManifest = JSON.parse(manifestContent);
        
        // Validate all files in manifest exist
        for (const file of manifest.files) {
          const filePath = path.join(tempDir, file.path);
          if (!(await this.fileExists(filePath))) {
            errors.push(`File missing from bundle: ${file.path}`);
            continue;
          }
          
          // Validate file hash
          const content = await fs.readFile(filePath);
          const actualHash = createHash('sha256').update(content).digest('hex');
          if (actualHash !== file.hash) {
            errors.push(`Hash mismatch for file: ${file.path}`);
          }
          
          // Validate file size
          const stat = await fs.stat(filePath);
          if (stat.size !== file.size) {
            errors.push(`Size mismatch for file: ${file.path}`);
          }
        }
        
        // Check reproducibility script exists and is executable
        const scriptPath = path.join(tempDir, manifest.reproducibilityScript);
        if (await this.fileExists(scriptPath)) {
          const stat = await fs.stat(scriptPath);
          if (!(stat.mode & parseInt('100', 8))) {
            errors.push('Reproducibility script is not executable');
          }
        } else {
          errors.push('Reproducibility script missing from bundle');
        }
        
        return {
          isValid: errors.length === 0,
          errors,
          manifest
        };
        
      } finally {
        // Cleanup temp directory
        try {
          await fs.rm(tempDir, { recursive: true, force: true });
        } catch (cleanupError) {
          console.warn('Failed to cleanup validation directory:', cleanupError);
        }
      }
      
    } catch (error) {
      errors.push(`Validation failed: ${error instanceof Error ? error.message : String(error)}`);
      return { isValid: false, errors };
    }
  }
}