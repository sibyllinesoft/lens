#!/usr/bin/env node

/**
 * Simple Forensics Manifest Generator
 * Creates inventory with content hashes for damage control
 */

import { createHash } from 'crypto';
import { readFileSync, writeFileSync, statSync, readdirSync } from 'fs';
import { join, relative } from 'path';
import { existsSync } from 'fs';

function walkDirectory(dir, baseDir = dir) {
    const files = [];
    try {
        const items = readdirSync(dir, { withFileTypes: true });
        for (const item of items) {
            if (item.name.startsWith('.') || item.name === 'node_modules' || item.name === 'forensics') {
                continue;
            }
            
            const fullPath = join(dir, item.name);
            if (item.isDirectory()) {
                files.push(...walkDirectory(fullPath, baseDir));
            } else {
                files.push(fullPath);
            }
        }
    } catch (err) {
        console.warn(`⚠️  Cannot read directory ${dir}: ${err.message}`);
    }
    return files;
}

function generateManifest() {
    console.log('🔍 FORENSICS: Generating content manifest...');
    
    const REPO_ROOT = process.cwd();
    const timestamp = new Date().toISOString();
    
    // Walk all directories
    const allFiles = walkDirectory(REPO_ROOT);
    const manifest = [];
    
    for (const fullPath of allFiles) {
        try {
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
            console.warn(`⚠️  Failed to process ${relative(REPO_ROOT, fullPath)}: ${err.message}`);
        }
    }
    
    // Write JSONL format
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

generateManifest();