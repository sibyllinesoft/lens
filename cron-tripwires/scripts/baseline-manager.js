#!/usr/bin/env node

/**
 * Baseline Configuration Manager
 * Manages baseline configurations for auto-revert system
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { createHash } from 'crypto';

class BaselineManager {
    constructor() {
        this.baselineDir = './cron-tripwires/baselines';
        this.timestamp = new Date().toISOString();
        
        // Ensure baseline directory exists
        if (!existsSync(this.baselineDir)) {
            mkdirSync(this.baselineDir, { recursive: true });
        }
    }

    async execute() {
        const command = process.argv[2];
        const fingerprint = process.argv[3];

        switch (command) {
            case 'capture':
                await this.captureBaseline(fingerprint);
                break;
            case 'list':
                await this.listBaselines();
                break;
            case 'validate':
                await this.validateBaseline(fingerprint);
                break;
            case 'restore':
                await this.restoreBaseline(fingerprint);
                break;
            default:
                this.showHelp();
        }
    }

    async captureBaseline(fingerprint) {
        if (!fingerprint) {
            throw new Error('Fingerprint required for baseline capture');
        }

        console.log(`📸 Capturing baseline: ${fingerprint}`);
        
        const baseline = {
            fingerprint: fingerprint,
            timestamp: this.timestamp,
            files: {}
        };
        
        // Capture configuration files
        const configFiles = [
            'config.json',
            'weights.json',
            'policy.json',
            'adapters.config',
            'embeddings.manifest'
        ];
        
        for (const file of configFiles) {
            if (existsSync(file)) {
                const content = readFileSync(file, 'utf8');
                const hash = createHash('sha256');
                hash.update(content);
                
                baseline.files[file] = {
                    content: content,
                    sha256: hash.digest('hex'),
                    size: content.length
                };
                
                console.log(`✅ ${file}: ${hash.digest('hex').substring(0, 12)}...`);
            } else {
                console.log(`⚠️  ${file}: not found (skipping)`);
            }
        }
        
        // Save individual configuration file for easy revert
        if (baseline.files['config.json']) {
            writeFileSync(
                `${this.baselineDir}/config-${fingerprint}.json`,
                baseline.files['config.json'].content
            );
        }
        
        // Save complete baseline manifest
        writeFileSync(
            `${this.baselineDir}/baseline-${fingerprint}.json`,
            JSON.stringify(baseline, null, 2)
        );
        
        console.log(`\n✅ Baseline captured: ${fingerprint}`);
        console.log(`📁 Baseline files: ${this.baselineDir}/`);
    }

    async listBaselines() {
        console.log('📋 Available baselines:');
        
        const { readdirSync } = await import('fs');
        
        try {
            const files = readdirSync(this.baselineDir)
                .filter(f => f.startsWith('baseline-') && f.endsWith('.json'))
                .sort();
                
            if (files.length === 0) {
                console.log('  No baselines found');
                return;
            }
            
            for (const file of files) {
                const baseline = JSON.parse(
                    readFileSync(`${this.baselineDir}/${file}`, 'utf8')
                );
                
                console.log(`  ${baseline.fingerprint} (${baseline.timestamp.split('T')[0]})`);
            }
        } catch (error) {
            console.log('  Error reading baseline directory');
        }
    }

    async validateBaseline(fingerprint) {
        if (!fingerprint) {
            throw new Error('Fingerprint required for validation');
        }

        console.log(`🔍 Validating baseline: ${fingerprint}`);
        
        const baselinePath = `${this.baselineDir}/baseline-${fingerprint}.json`;
        if (!existsSync(baselinePath)) {
            throw new Error(`Baseline not found: ${fingerprint}`);
        }
        
        const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
        
        // Validate each file
        for (const [filename, fileInfo] of Object.entries(baseline.files)) {
            const hash = createHash('sha256');
            hash.update(fileInfo.content);
            const currentHash = hash.digest('hex');
            
            if (currentHash === fileInfo.sha256) {
                console.log(`✅ ${filename}: integrity verified`);
            } else {
                console.log(`❌ ${filename}: integrity check failed`);
            }
        }
        
        console.log(`\n✅ Baseline validation complete: ${fingerprint}`);
    }

    showHelp() {
        console.log(`Lens v2.2 Baseline Manager

Usage:
  node baseline-manager.js <command> [options]

Commands:
  capture <fingerprint>   Capture current configuration as baseline
  list                    List available baselines
  validate <fingerprint>  Validate baseline integrity
  restore <fingerprint>   Restore configuration from baseline

Examples:
  node baseline-manager.js capture v22_1f3db391_1757345166574
  node baseline-manager.js list
  node baseline-manager.js validate v22_1f3db391_1757345166574
`);
    }
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
    try {
        const manager = new BaselineManager();
        await manager.execute();
    } catch (error) {
        console.error('❌ Baseline management failed:', error.message);
        process.exit(1);
    }
}
