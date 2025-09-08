#!/usr/bin/env node

/**
 * Synthetic Output Scanner
 * 
 * Scans all files for synthetic/mock markers as specified in TODO.md:
 * - Filenames like `mock_file_*.rust`
 * - Fields like `span_error_reason: "MOCK_RESULT"`, `generateMock*`, `"Simulate"`, `"mock"`
 * - Missing service handshake markers
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';

function loadManifest() {
    const manifestPath = join(process.cwd(), 'forensics', 'manifest.jsonl');
    if (!existsSync(manifestPath)) {
        throw new Error('Run generate-forensics-manifest-simple.js first');
    }
    
    const content = readFileSync(manifestPath, 'utf-8');
    return content.split('\n').filter(line => line.trim()).map(line => JSON.parse(line));
}

function scanForSyntheticMarkers() {
    console.log('🔍 FORENSICS: Scanning for synthetic/mock markers...');
    
    const manifest = loadManifest();
    const contaminated = [];
    
    // Synthetic markers to detect
    const fileNamePatterns = [
        /mock_file_.*\.rust$/i,
        /mock.*\.rust$/i,
        /simulate/i,
        /fake/i,
        /synthetic/i
    ];
    
    const contentPatterns = [
        /span_error_reason.*["']MOCK_RESULT["']/i,
        /generateMock/i,
        /"Simulate"/i,
        /"mock"/i,
        /MOCK_RESULT/i,
        /synthetic.*data/i,
        /fake.*result/i,
        /mock.*generator/i,
        /anchor.*smoke/i // Specifically mentioned in TODO
    ];
    
    for (const entry of manifest) {
        const reasons = [];
        
        // Check filename patterns
        for (const pattern of fileNamePatterns) {
            if (pattern.test(entry.path)) {
                reasons.push(`Suspicious filename pattern: ${pattern.source}`);
            }
        }
        
        // Check file content for text files
        if (entry.path.match(/\.(js|json|ts|md|txt|csv|log|yaml|yml)$/i)) {
            try {
                const content = readFileSync(entry.path, 'utf-8');
                
                for (const pattern of contentPatterns) {
                    if (pattern.test(content)) {
                        reasons.push(`Content contains suspicious pattern: ${pattern.source}`);
                    }
                }
                
                // Check for missing service handshake in benchmark results
                if (entry.path.includes('benchmark') || entry.path.includes('results')) {
                    if (entry.path.endsWith('.json')) {
                        try {
                            const jsonData = JSON.parse(content);
                            
                            // Check for missing handshake markers
                            const hasHandshake = jsonData.handshake || 
                                               jsonData.sut || 
                                               jsonData.build_sha || 
                                               jsonData.__buildinfo;
                            
                            if (!hasHandshake) {
                                reasons.push('Missing service handshake in benchmark result');
                            }
                            
                            // Check for explicit mock markers in JSON
                            const jsonStr = JSON.stringify(jsonData).toLowerCase();
                            if (jsonStr.includes('mock') || jsonStr.includes('simulate') || jsonStr.includes('fake')) {
                                reasons.push('JSON content contains mock/simulate/fake references');
                            }
                            
                        } catch (err) {
                            // Not valid JSON, skip
                        }
                    }
                }
                
            } catch (err) {
                // Cannot read file, skip content checks
            }
        }
        
        if (reasons.length > 0) {
            contaminated.push({
                path: entry.path,
                sha256: entry.sha256,
                reasons: reasons,
                first_seen_commit: 'TBD', // Will be filled by git bisect
                scanned_at: new Date().toISOString()
            });
        }
    }
    
    // Write contaminated.csv
    const csvPath = join(process.cwd(), 'forensics', 'contaminated.csv');
    const csvHeader = 'path,sha256,reason,first_seen_commit,scanned_at\n';
    const csvRows = contaminated.flatMap(item => 
        item.reasons.map(reason => 
            `"${item.path}","${item.sha256}","${reason}","${item.first_seen_commit}","${item.scanned_at}"`
        )
    );
    
    writeFileSync(csvPath, csvHeader + csvRows.join('\n'));
    
    // Write detailed JSON report
    const reportPath = join(process.cwd(), 'forensics', 'contaminated-detailed.json');
    writeFileSync(reportPath, JSON.stringify(contaminated, null, 2));
    
    console.log(`🔍 FORENSICS: Found ${contaminated.length} potentially contaminated files`);
    console.log(`📁 CSV report: ${csvPath}`);
    console.log(`📁 Detailed report: ${reportPath}`);
    
    if (contaminated.length > 0) {
        console.log('\n⚠️  CONTAMINATED FILES DETECTED:');
        contaminated.slice(0, 10).forEach(item => {
            console.log(`   ${item.path} - ${item.reasons[0]}`);
        });
        if (contaminated.length > 10) {
            console.log(`   ... and ${contaminated.length - 10} more`);
        }
    }
    
    return contaminated;
}

scanForSyntheticMarkers();