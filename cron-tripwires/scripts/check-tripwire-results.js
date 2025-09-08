#!/usr/bin/env node

/**
 * Tripwire Results Checker
 * Parses validation results and exits with appropriate code
 */

import { readFileSync, existsSync } from 'fs';

const resultsFile = process.argv[2];

if (!resultsFile || !existsSync(resultsFile)) {
    console.error('❌ Validation results file not found');
    process.exit(2);
}

try {
    const results = JSON.parse(readFileSync(resultsFile, 'utf8'));
    
    console.log(`📊 Checking tripwire results from ${results.timestamp}`);
    
    if (results.overall_status === 'PASS') {
        console.log('✅ All tripwires PASSED');
        process.exit(0);
    } else if (results.overall_status === 'FAIL') {
        console.log(`❌ ${results.alerts.length} tripwire(s) FAILED`);
        results.alerts.forEach(alert => {
            console.log(`   [${alert.severity}] ${alert.tripwire}: ${alert.message}`);
        });
        process.exit(1);
    } else {
        console.log('❌ Unknown validation status');
        process.exit(2);
    }
} catch (error) {
    console.error('❌ Error parsing results:', error.message);
    process.exit(2);
}
