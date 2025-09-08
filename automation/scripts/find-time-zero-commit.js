#!/usr/bin/env node

/**
 * Time-Zero Commit Finder
 * 
 * Uses git log -S and git bisect to find when synthetic markers first appeared
 */

import { execSync } from 'child_process';
import { writeFileSync } from 'fs';
import { join } from 'path';

function runGitCommand(cmd) {
    try {
        return execSync(cmd, { encoding: 'utf-8', cwd: process.cwd() }).trim();
    } catch (error) {
        console.warn(`Git command failed: ${cmd}`);
        return null;
    }
}

function findTimeZeroCommit() {
    console.log('🔍 FORENSICS: Finding time-zero commit...');
    
    // Suspicious patterns from TODO.md
    const suspiciousPatterns = [
        'MOCK_RESULT',
        'generateMock',
        'mock_file_',
        'anchor.*smoke',
        'Simulate'
    ];
    
    const timeZeroReport = {
        analysis_timestamp: new Date().toISOString(),
        suspicious_patterns: suspiciousPatterns,
        pattern_first_appearances: {},
        suspected_time_zero: null,
        git_log_analysis: []
    };
    
    console.log('📊 Analyzing git history for suspicious patterns...');
    
    // Search git history for each pattern
    for (const pattern of suspiciousPatterns) {
        console.log(`   Searching for: ${pattern}`);
        
        // Use git log -S to find when pattern was introduced
        const gitLogCmd = `git log -S "${pattern}" --oneline --reverse`;
        const logOutput = runGitCommand(gitLogCmd);
        
        if (logOutput) {
            const commits = logOutput.split('\n').filter(line => line.trim());
            if (commits.length > 0) {
                const firstCommit = commits[0];
                const [hash, ...messageParts] = firstCommit.split(' ');
                
                timeZeroReport.pattern_first_appearances[pattern] = {
                    first_commit: hash,
                    commit_message: messageParts.join(' '),
                    total_commits: commits.length,
                    all_commits: commits
                };
                
                console.log(`     First found in: ${hash} - ${messageParts.join(' ')}`);
            }
        }
        
        // Also check recent commits more thoroughly
        const recentCommitsCmd = `git log --oneline -20 --grep="${pattern}"`;
        const recentOutput = runGitCommand(recentCommitsCmd);
        if (recentOutput) {
            timeZeroReport.git_log_analysis.push({
                pattern,
                grep_results: recentOutput.split('\n').filter(line => line.trim())
            });
        }
    }
    
    // Find the earliest suspicious commit
    const allFirstCommits = Object.values(timeZeroReport.pattern_first_appearances)
        .map(info => info.first_commit)
        .filter(Boolean);
    
    if (allFirstCommits.length > 0) {
        // Get commit timestamps to find earliest
        const commitTimestamps = allFirstCommits.map(hash => {
            const timestamp = runGitCommand(`git show -s --format=%ct ${hash}`);
            return {
                hash,
                timestamp: parseInt(timestamp),
                date: runGitCommand(`git show -s --format=%ci ${hash}`)
            };
        }).sort((a, b) => a.timestamp - b.timestamp);
        
        const earliestCommit = commitTimestamps[0];
        timeZeroReport.suspected_time_zero = {
            commit_hash: earliestCommit.hash,
            commit_date: earliestCommit.date,
            confidence: 'high',
            reasoning: 'Earliest commit introducing suspicious patterns'
        };
        
        console.log(`🚨 SUSPECTED TIME-ZERO: ${earliestCommit.hash} (${earliestCommit.date})`);
    } else {
        console.log('⚠️  No clear time-zero found in git history - patterns may predate repo');
        timeZeroReport.suspected_time_zero = {
            commit_hash: null,
            commit_date: null,
            confidence: 'unknown',
            reasoning: 'No suspicious patterns found in git history'
        };
    }
    
    // Additional analysis: find large commits that might have introduced bulk synthetic data
    console.log('📊 Analyzing large commits for bulk synthetic data introduction...');
    
    const largeCommitsCmd = `git log --oneline --stat | grep -E "files? changed" | head -10`;
    const largeCommitsOutput = runGitCommand('git log --oneline --shortstat -10');
    
    if (largeCommitsOutput) {
        timeZeroReport.large_commits_analysis = largeCommitsOutput.split('\n')
            .filter(line => line.includes('insertion') || line.includes('deletion'))
            .slice(0, 5);
    }
    
    // Look for specific anchor smoke mentions
    const anchorSmokeCmd = `git log --oneline --all --grep="smoke" --grep="anchor" --grep="mock"`;
    const anchorSmokeOutput = runGitCommand(anchorSmokeCmd);
    if (anchorSmokeOutput) {
        timeZeroReport.anchor_smoke_commits = anchorSmokeOutput.split('\n').filter(line => line.trim());
    }
    
    // Write comprehensive report
    const reportPath = join(process.cwd(), 'forensics', 'time-zero-analysis.json');
    writeFileSync(reportPath, JSON.stringify(timeZeroReport, null, 2));
    
    console.log(`📁 Time-zero analysis saved to: ${reportPath}`);
    console.log(`🔍 Analysis complete. ${Object.keys(timeZeroReport.pattern_first_appearances).length} suspicious patterns detected.`);
    
    return timeZeroReport;
}

findTimeZeroCommit();