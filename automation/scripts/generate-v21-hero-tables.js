#!/usr/bin/env node
/**
 * Generate hero tables from Protocol v2.1 results
 * Follows TODO.md specification for "hero tables/plots"
 */

import fs from 'fs';
import path from 'path';

const PROTOCOL_V21_AGG = './canonical/v21/agg.json';
const OUTPUT_DIR = './tables';

async function generateHeroTables() {
    console.log('📊 GENERATING PROTOCOL V2.1 HERO TABLES');
    console.log('=====================================');

    // Load Protocol v2.1 aggregated results
    const aggData = JSON.parse(fs.readFileSync(PROTOCOL_V21_AGG, 'utf8'));
    console.log(`📥 Loaded ${aggData.length} aggregated results`);

    // Separate span-only and hierarchical results
    const spanOnly = aggData.filter(row => row.credit_mode_used === 'span_only');
    const hierarchical = aggData.filter(row => row.credit_mode_used === 'hierarchical');

    console.log(`📋 Found ${spanOnly.length} span-only and ${hierarchical.length} hierarchical results`);

    // Generate span-only hero table
    const spanHeroData = generateHeroData(spanOnly);
    const spanCsv = convertToCSV(spanHeroData, ['system', 'capability_slice', 'mean_ndcg_at_10', 'mean_success_at_10', 'sla_compliance_rate', 'total_queries']);
    
    fs.writeFileSync(path.join(OUTPUT_DIR, 'hero_span_v21.csv'), spanCsv);
    console.log(`✅ Generated ${OUTPUT_DIR}/hero_span_v21.csv`);

    // Generate hierarchical hero table
    const hierarchicalHeroData = generateHeroData(hierarchical);
    const hierarchicalCsv = convertToCSV(hierarchicalHeroData, ['system', 'capability_slice', 'mean_ndcg_at_10', 'mean_success_at_10', 'sla_compliance_rate', 'total_queries']);
    
    fs.writeFileSync(path.join(OUTPUT_DIR, 'hero_hierarchical_v21.csv'), hierarchicalCsv);
    console.log(`✅ Generated ${OUTPUT_DIR}/hero_hierarchical_v21.csv`);

    // Generate combined capability slice summary
    const sliceSummary = generateSliceSummary(spanOnly);
    const sliceCsv = convertToCSV(sliceSummary, ['capability_slice', 'leader_system', 'leader_ndcg', 'systems_count', 'avg_ndcg']);
    
    fs.writeFileSync(path.join(OUTPUT_DIR, 'capability_slice_leaders.csv'), sliceCsv);
    console.log(`✅ Generated ${OUTPUT_DIR}/capability_slice_leaders.csv`);

    console.log('\n🏆 PROTOCOL V2.1 HERO TABLES COMPLETE');
    console.log('====================================');
}

function generateHeroData(results) {
    // Group by system and aggregate across all datasets/scenarios
    const systemStats = {};
    
    for (const row of results) {
        const system = row.system;
        if (!systemStats[system]) {
            systemStats[system] = {
                system: system,
                capability_slice: row.system_slice,
                ndcg_scores: [],
                success_scores: [],
                within_sla_count: 0,
                total_queries: 0
            };
        }
        
        systemStats[system].ndcg_scores.push(row.ndcg10);
        systemStats[system].success_scores.push(row.success10);
        systemStats[system].within_sla_count += row.within_sla ? 1 : 0;
        systemStats[system].total_queries += 1;
    }
    
    // Convert to hero table format
    const heroData = [];
    for (const [system, stats] of Object.entries(systemStats)) {
        heroData.push({
            system: system,
            capability_slice: stats.capability_slice,
            mean_ndcg_at_10: (stats.ndcg_scores.reduce((a, b) => a + b, 0) / stats.ndcg_scores.length).toFixed(4),
            mean_success_at_10: (stats.success_scores.reduce((a, b) => a + b, 0) / stats.success_scores.length).toFixed(4),
            sla_compliance_rate: (stats.within_sla_count / stats.total_queries).toFixed(4),
            total_queries: stats.total_queries
        });
    }
    
    // Sort by nDCG descending
    heroData.sort((a, b) => parseFloat(b.mean_ndcg_at_10) - parseFloat(a.mean_ndcg_at_10));
    
    return heroData;
}

function generateSliceSummary(results) {
    // Group by capability slice to find leaders
    const sliceStats = {};
    
    for (const row of results) {
        const slice = row.system_slice;
        if (!sliceStats[slice]) {
            sliceStats[slice] = {
                capability_slice: slice,
                systems: {},
                best_ndcg: 0,
                leader_system: null
            };
        }
        
        if (!sliceStats[slice].systems[row.system]) {
            sliceStats[slice].systems[row.system] = {
                ndcg_scores: [],
                system: row.system
            };
        }
        
        sliceStats[slice].systems[row.system].ndcg_scores.push(row.ndcg10);
    }
    
    // Calculate slice leaders
    const sliceSummary = [];
    for (const [slice, stats] of Object.entries(sliceStats)) {
        let bestSystem = null;
        let bestNdcg = 0;
        const systemNdcgs = [];
        
        for (const [system, systemStats] of Object.entries(stats.systems)) {
            const avgNdcg = systemStats.ndcg_scores.reduce((a, b) => a + b, 0) / systemStats.ndcg_scores.length;
            systemNdcgs.push(avgNdcg);
            
            if (avgNdcg > bestNdcg) {
                bestNdcg = avgNdcg;
                bestSystem = system;
            }
        }
        
        const avgSliceNdcg = systemNdcgs.reduce((a, b) => a + b, 0) / systemNdcgs.length;
        
        sliceSummary.push({
            capability_slice: slice,
            leader_system: bestSystem,
            leader_ndcg: bestNdcg.toFixed(4),
            systems_count: Object.keys(stats.systems).length,
            avg_ndcg: avgSliceNdcg.toFixed(4)
        });
    }
    
    // Sort by leader nDCG descending
    sliceSummary.sort((a, b) => parseFloat(b.leader_ndcg) - parseFloat(a.leader_ndcg));
    
    return sliceSummary;
}

function convertToCSV(data, columns) {
    const header = columns.join(',');
    const rows = data.map(row => columns.map(col => row[col]).join(','));
    return [header, ...rows].join('\n');
}

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

generateHeroTables().catch(console.error);