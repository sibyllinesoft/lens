#!/usr/bin/env node

/**
 * Forensics Manifest Generator
 * 
 * Creates a comprehensive inventory of all files with content hashes
 * for damage control and provenance tracking.
 */

import { createHash } from 'crypto';
import { readFileSync, writeFileSync, statSync } from 'fs';
import { join, relative } from 'path';
import { glob } from 'glob';

const REPO_ROOT = process.cwd();

async function generateManifest() {
    console.log('🔍 FORENSICS: Generating content manifest...');
    
    const patterns = [
        'benchmark-results/**/*',
        'baseline-results/**/*', 
        'pinned-datasets/**/*',
        'validation-data/**/*',
        'indexed-content/**/*',
        'src/**/*',
        '*.js',
        '*.json',
        '*.md',
        '*.ts',
        'reports/**/*',
        'artifacts/**/*'
    ];
    
    const manifest = [];
    const timestamp = new Date().toISOString();
    
    for (const pattern of patterns) {
        try {
            const files = await glob(pattern, { 
                cwd: REPO_ROOT,
                nodir: true,
                ignore: ['node_modules/**', '.git/**', 'forensics/**']
            });
            
            for (const file of files) {
                try {
                    const fullPath = join(REPO_ROOT, file);
                    const stats = statSync(fullPath);
                    const content = readFileSync(fullPath);
                    const sha256 = createHash('sha256').update(content).digest('hex');
                    
                    manifest.push({
                        path: relative(REPO_ROOT, fullPath),
                        sha256,
                        bytes: stats.size,
                        mtime: stats.mtime.toISOString(),
                        scanned_at: timestamp
                    });
                } catch (err) {
                    console.warn(`⚠️  Failed to process ${file}: ${err.message}`);
                }
            }
        } catch (err) {
            console.warn(`⚠️  Failed to glob ${pattern}: ${err.message}`);
        }
    }
    
    // Write JSONL format as specified
    const manifestPath = join(REPO_ROOT, 'forensics', 'manifest.jsonl');
    const jsonlContent = manifest.map(entry => JSON.stringify(entry)).join('\n');
    writeFileSync(manifestPath, jsonlContent);
    
    console.log(`✅ FORENSICS: Manifest generated with ${manifest.length} files`);
    console.log(`📁 Saved to: ${manifestPath}`);
    
    // Generate summary
    const summary = {
        total_files: manifest.length,
        total_bytes: manifest.reduce((sum, entry) => sum + entry.bytes, 0),
        generated_at: timestamp,
        file_types: {}
    };
    
    manifest.forEach(entry => {
        const ext = entry.path.split('.').pop() || 'no-extension';
        summary.file_types[ext] = (summary.file_types[ext] || 0) + 1;
    });
    
    writeFileSync(
        join(REPO_ROOT, 'forensics', 'manifest-summary.json'),
        JSON.stringify(summary, null, 2)
    );
    
    console.log(`📊 Summary: ${summary.total_files} files, ${(summary.total_bytes / 1024 / 1024).toFixed(2)} MB`);
    
    return manifest;
}

if (import.meta.url === `file://${process.argv[1]}`) {
    generateManifest().catch(console.error);
}

export { generateManifest };