#!/usr/bin/env node
/**
 * Generate Surgical Sprints from Gap Analysis
 * Implements TODO.md: Turn 2,645 gaps into 5 focused sprints with measurable gates
 */

import fs from 'fs';

class SurgicalSprintGenerator {
    constructor() {
        this.gapData = null;
        this.sprints = [];
    }

    async generateSprints() {
        console.log('⚔️ GENERATING SURGICAL SPRINTS FROM GAP ANALYSIS');
        console.log('===============================================');
        console.log('🎯 Target: 2,645 gaps → 5 focused sprints with measurable gates\n');

        // Load gap analysis data
        console.log('=== STEP 1: LOAD GAP ANALYSIS DATA ===');
        await this.loadGapAnalysis();

        // Generate 5 surgical sprints
        console.log('=== STEP 2: GENERATE 5 SURGICAL SPRINTS ===');
        await this.createSurgicalSprints();

        // Create ops brief
        console.log('=== STEP 3: CREATE OPERATIONS BRIEF ===');
        await this.createOperationsBrief();

        return this.finalizeSprints();
    }

    async loadGapAnalysis() {
        console.log('📥 Loading gap analysis from v2.2 results...');
        
        try {
            const gapDataStr = fs.readFileSync('./gap_analysis/v22/roadmap.json', 'utf8');
            this.gapData = JSON.parse(gapDataStr);
            
            console.log(`✅ Loaded ${this.gapData.gaps.length} gaps across ${this.gapData.roadmap_items.length} remedy classes`);
            
            for (const item of this.gapData.roadmap_items) {
                console.log(`   ${item.remedy_class}: ${item.affected_queries} queries, Δ${item.avg_gap_delta.toFixed(3)} avg gap`);
            }
            console.log('');
            
        } catch (error) {
            console.error('Failed to load gap analysis:', error);
            throw error;
        }
    }

    async createSurgicalSprints() {
        console.log('⚔️ Creating 5 surgical sprints with focused execution plans...');

        // Sprint 1: Timeout Handling (1,781 queries)
        this.sprints.push({
            sprint_number: 1,
            name: 'timeout_handling',
            title: 'Tail-Taming & SLA Recovery',
            affected_queries: 1781,
            avg_gap_delta: 0.089,
            duration: '2 weeks',
            
            problem_statement: 'Lens loses on tail latency and SLA compliance, causing timeouts that hurt recall.',
            
            levers: [
                {
                    name: 'Hedged Probes + Cooperative Cancel',
                    description: 'Send late clone to slowest shard at p95+δ; cancel losers',
                    implementation: 'Add hedging logic to query router with cooperative cancellation',
                    gate: 'p99 -10-15%, SLA-Recall@50 Δ≥0, cost ≤+5%'
                },
                {
                    name: 'Cross-shard TA/NRA Thresholding + Learning-to-stop', 
                    description: 'Tighten upper-bound sharing and stop when top-K stable',
                    implementation: 'Implement threshold algorithm with early termination',
                    gate: 'QPS@150ms +10-15% at flat recall'
                }
            ],
            
            success_metrics: {
                primary: 'p99 latency reduction 10-15%',
                secondary: 'SLA compliance rate improvement',
                constraint: 'Cost increase ≤5%',
                gate_threshold: 'QPS@150ms +10% minimum'
            }
        });

        // Sprint 2: Lexical Precision (428 queries)
        this.sprints.push({
            sprint_number: 2,
            name: 'lexical_precision',
            title: 'Exact & Near-Exact Precision',
            affected_queries: 428,
            avg_gap_delta: 0.080,
            duration: '2 weeks',
            
            problem_statement: 'Lens loses precision on exact/near-exact matches where lexical systems excel.',
            
            levers: [
                {
                    name: 'Phrase/Proximity Scoring in Stage-A',
                    description: 'Impact-ordered postings with bi-gram boosts; cap log-odds',
                    implementation: 'Enhance Stage-A with proximity-aware term scoring',
                    gate: '+1-2 pp nDCG on regex/substring slices; ≤+0.5 ms p95'
                },
                {
                    name: 'Panic-mode Exactifier',
                    description: 'If reranker confidence <τ and query token-dense, promote exact spans',
                    implementation: 'Add confidence-gated exact match promotion',
                    gate: 'Success@10 +1 pp on lexical slice; global nDCG flat'
                }
            ],
            
            success_metrics: {
                primary: 'nDCG improvement 1-2 pp on lexical slices',
                secondary: 'Success@10 rate improvement',
                constraint: 'Latency increase ≤0.5 ms p95',
                gate_threshold: 'Min +1 pp Success@10 on lexical'
            }
        });

        // Sprint 3: Clone Expansion (260 queries)
        this.sprints.push({
            sprint_number: 3,
            name: 'clone_expansion', 
            title: 'Near-Duplicate Spread',
            affected_queries: 260,
            avg_gap_delta: 0.095,
            duration: '2 weeks',
            
            problem_statement: 'Lens misses relevant results spread across code clones and backports.',
            
            levers: [
                {
                    name: 'MinHash/SimHash Recall Fan-out',
                    description: 'Expand hits across twins/backports (k_clone≤3); vendor-path veto',
                    implementation: 'Implement locality-sensitive hashing for clone detection',
                    gate: 'Recall@50 +0.5-1.0 pp on clone-heavy; ≤+0.6 ms p95'
                },
                {
                    name: 'De-dupe in Rerank',
                    description: 'MMR on NL_overview only (γ≈0.1) to avoid same-file clusters',
                    implementation: 'Add maximal marginal relevance with diversity penalty',
                    gate: 'Diversity@10 +10-15% with ΔnDCG≈0'
                }
            ],
            
            success_metrics: {
                primary: 'Recall@50 improvement 0.5-1.0 pp on clone-heavy',
                secondary: 'Result diversity improvement 10-15%',
                constraint: 'Latency increase ≤0.6 ms p95',
                gate_threshold: 'Min +0.5 pp Recall@50 clone scenarios'
            }
        });

        // Sprint 4: Router Thresholds (156 queries)
        this.sprints.push({
            sprint_number: 4,
            name: 'router_thresholds',
            title: 'Smarter Multi-Signal Spend', 
            affected_queries: 156,
            avg_gap_delta: 0.166,
            duration: '2 weeks',
            
            problem_statement: 'Lens router makes suboptimal decisions on when to use expensive signals.',
            
            levers: [
                {
                    name: 'ROC-fitted Thresholds w/ Conformal Guard',
                    description: 'Optimize τ to maximize ΔnDCG - λ·Δp95 (λ≈0.2 pp/ms), keep miscoverage within target+1.5 pp',
                    implementation: 'Implement conformal prediction for threshold optimization',
                    gate: 'Upshift 5%±2 pp; +≥1 pp on "hard NL"'
                },
                {
                    name: 'Entropy-conditioned Routing',
                    description: 'Only spend on high-entropy topics; cap per-query risk',
                    implementation: 'Add entropy-based query difficulty assessment',
                    gate: 'Same lift with -20% router spend'
                }
            ],
            
            success_metrics: {
                primary: 'nDCG upshift 5%±2 pp on hard queries',
                secondary: 'Router efficiency -20% spend for same quality',
                constraint: 'Maintain miscoverage within target+1.5 pp',
                gate_threshold: 'Min +1 pp on hard NL scenarios'
            }
        });

        // Sprint 5: ANN Hygiene (20 queries)
        this.sprints.push({
            sprint_number: 5,
            name: 'ann_hygiene',
            title: 'Vector Index Quality',
            affected_queries: 20,
            avg_gap_delta: 0.077,
            duration: '2 weeks',
            
            problem_statement: 'Lens vector search underperforms due to index configuration and retrieval inefficiency.',
            
            levers: [
                {
                    name: 'efSearch vs Latency Sweep + PQ Recall Audit',
                    description: 'Lock the Pareto point; run neighbor-drift sanity',
                    implementation: 'Optimize HNSW parameters and validate quantization quality',
                    gate: '+0.3-0.5 pp on NL with ≤+1 ms p95'
                },
                {
                    name: 'Visited-set Reuse + Prefetch',
                    description: 'Speed ANN without recall change',
                    implementation: 'Implement visited set optimization and memory prefetching',
                    gate: '-1-2 ms p95 dense path'
                }
            ],
            
            success_metrics: {
                primary: 'nDCG improvement 0.3-0.5 pp on NL scenarios',
                secondary: 'Dense path latency reduction 1-2 ms p95',
                constraint: 'No recall degradation on vector search',
                gate_threshold: 'Min +0.3 pp nDCG on NL'
            }
        });

        console.log(`✅ Generated ${this.sprints.length} surgical sprints`);
        for (const sprint of this.sprints) {
            console.log(`   Sprint ${sprint.sprint_number}: ${sprint.title} (${sprint.affected_queries} queries, ${sprint.duration})`);
        }
        console.log('');
    }

    async createOperationsBrief() {
        console.log('📋 Creating one-page operations brief for team handoff...');

        const opsBrief = {
            title: 'Surgical Sprint Operations Brief',
            subtitle: 'Gap-Driven Lens Improvements - 5 Focused Sprints',
            overview: {
                total_gaps: this.gapData.gaps.length,
                total_affected_queries: this.sprints.reduce((sum, s) => sum + s.affected_queries, 0),
                sprint_duration: '2 weeks each',
                execution_model: 'Sequential sprints with A/A testing and SLA-bounded validation'
            },
            
            execution_framework: {
                methodology: 'Each sprint ships with A/A testing, SLA-bounded bench ladder, and revert-in-30s rollback',
                success_criteria: 'Success criteria live in same parquet so plots/dashboards update automatically',
                cadence: '2-week sprints with demo/retrospective at end',
                rollback_policy: 'Automatic rollback if any gate regresses beyond threshold'
            },
            
            sprints: this.sprints,
            
            monitoring_dashboards: [
                {
                    name: 'Sprint Progress Dashboard',
                    metrics: ['Gate progress', 'A/A test results', 'SLA compliance', 'Rollback triggers'],
                    update_frequency: 'Real-time'
                },
                {
                    name: 'Hero Table Live View', 
                    metrics: ['nDCG@10 trends', 'CI width evolution', 'System ranking changes'],
                    update_frequency: 'Every commit'
                },
                {
                    name: 'Gap Closure Tracker',
                    metrics: ['Gaps resolved per sprint', 'Roadmap item completion', 'Priority score changes'],
                    update_frequency: 'Weekly'
                }
            ],
            
            team_assignments: {
                sprint_lead: 'Responsible for gate tracking and rollback decisions',
                implementation_team: '2-3 engineers per sprint',
                qa_validation: 'SLA harness and A/A test execution',
                product_review: 'Gate threshold validation and success criteria sign-off'
            },
            
            risk_mitigation: {
                technical_risks: [
                    'Performance regression beyond gate thresholds',
                    'A/A test contamination or statistical power issues',
                    'Rollback complexity in multi-component changes'
                ],
                process_risks: [
                    'Sprint scope creep beyond defined levers',
                    'Gate threshold disagreement mid-sprint', 
                    'Resource contention between parallel work streams'
                ],
                mitigation_strategies: [
                    'Hard gate enforcement with automated rollback',
                    'Pre-agreed success criteria with stakeholder sign-off',
                    'Dedicated sprint team with protected time allocation'
                ]
            }
        };

        // Save operations brief
        fs.mkdirSync('./sprints', { recursive: true });
        fs.writeFileSync('./sprints/ops-brief.json', JSON.stringify(opsBrief, null, 2));

        // Generate human-readable ops brief
        const readableBrief = this.generateReadableOpsBrief(opsBrief);
        fs.writeFileSync('./sprints/ops-brief.md', readableBrief);

        console.log('✅ Operations brief saved to ./sprints/ops-brief.json');
        console.log('✅ Human-readable version: ./sprints/ops-brief.md');
        console.log('');
    }

    generateReadableOpsBrief(opsBrief) {
        return `# ${opsBrief.title}

**${opsBrief.subtitle}**

## 🎯 Overview

- **Total Gaps**: ${opsBrief.overview.total_gaps.toLocaleString()} identified from v2.2 competitive analysis  
- **Affected Queries**: ${opsBrief.overview.total_affected_queries.toLocaleString()} queries across 5 remedy classes
- **Sprint Duration**: ${opsBrief.overview.sprint_duration}
- **Execution Model**: ${opsBrief.overview.execution_model}

## 🏃 Sprint Execution Framework

**Methodology**: ${opsBrief.execution_framework.methodology}

**Success Criteria**: ${opsBrief.execution_framework.success_criteria}

**Cadence**: ${opsBrief.execution_framework.cadence}

**Rollback Policy**: ${opsBrief.execution_framework.rollback_policy}

## ⚔️ Surgical Sprint Details

${opsBrief.sprints.map(sprint => `
### Sprint ${sprint.sprint_number}: ${sprint.title}

**Problem**: ${sprint.problem_statement}

**Impact**: ${sprint.affected_queries} queries, Δ${sprint.avg_gap_delta.toFixed(3)} average gap

**Duration**: ${sprint.duration}

**Levers**:
${sprint.levers.map(lever => `
- **${lever.name}**: ${lever.description}
  - *Implementation*: ${lever.implementation}
  - *Gate*: ${lever.gate}
`).join('')}

**Success Metrics**:
- Primary: ${sprint.success_metrics.primary}
- Secondary: ${sprint.success_metrics.secondary}  
- Constraint: ${sprint.success_metrics.constraint}
- Gate: ${sprint.success_metrics.gate_threshold}
`).join('')}

## 📊 Monitoring & Dashboards

${opsBrief.monitoring_dashboards.map(dashboard => `
### ${dashboard.name}
- **Metrics**: ${dashboard.metrics.join(', ')}
- **Update Frequency**: ${dashboard.update_frequency}
`).join('')}

## 👥 Team Structure

- **Sprint Lead**: ${opsBrief.team_assignments.sprint_lead}
- **Implementation Team**: ${opsBrief.team_assignments.implementation_team}
- **QA Validation**: ${opsBrief.team_assignments.qa_validation}
- **Product Review**: ${opsBrief.team_assignments.product_review}

## ⚠️ Risk Mitigation

**Technical Risks**:
${opsBrief.risk_mitigation.technical_risks.map(risk => `- ${risk}`).join('\n')}

**Process Risks**:  
${opsBrief.risk_mitigation.process_risks.map(risk => `- ${risk}`).join('\n')}

**Mitigation Strategies**:
${opsBrief.risk_mitigation.mitigation_strategies.map(strategy => `- ${strategy}`).join('\n')}

---

**Ready to start Monday** 🚀

*Generated from gap analysis artifact: ${this.gapData.run_id}*`;
    }

    finalizeSprints() {
        // Create sprint tracking template
        const sprintTracker = {
            tracking_version: '1.0',
            total_sprints: this.sprints.length,
            current_sprint: null,
            completed_sprints: [],
            
            sprint_template: {
                sprint_number: null,
                name: null,
                status: 'not_started', // not_started, in_progress, completed, rolled_back
                start_date: null,
                end_date: null,
                
                gates: {
                    gate_1_status: 'pending',
                    gate_1_measurement: null,
                    gate_1_threshold: null,
                    gate_2_status: 'pending', 
                    gate_2_measurement: null,
                    gate_2_threshold: null
                },
                
                a_a_testing: {
                    test_id: null,
                    control_group: 'current_production',
                    treatment_group: 'sprint_implementation',
                    statistical_power: 0.8,
                    significance_level: 0.05,
                    min_sample_size: 1000
                },
                
                rollback_plan: {
                    rollback_trigger_conditions: [],
                    rollback_script: null,
                    rollback_validation: null,
                    estimated_rollback_time: '30 seconds'
                }
            },
            
            sprints: this.sprints
        };

        fs.writeFileSync('./sprints/sprint-tracker.json', JSON.stringify(sprintTracker, null, 2));

        // Generate summary
        const summary = {
            generated_at: new Date().toISOString(),
            source_gaps: this.gapData.gaps.length,
            generated_sprints: this.sprints.length,
            total_affected_queries: this.sprints.reduce((sum, s) => sum + s.affected_queries, 0),
            estimated_timeline: `${this.sprints.length * 2} weeks total`,
            
            deliverables: {
                ops_brief: './sprints/ops-brief.md',
                ops_brief_json: './sprints/ops-brief.json', 
                sprint_tracker: './sprints/sprint-tracker.json'
            },
            
            next_steps: [
                'Review ops brief with engineering team',
                'Assign sprint leads and implementation teams',
                'Set up monitoring dashboards',
                'Schedule Sprint #1 kickoff for Monday',
                'Prepare A/A testing infrastructure'
            ]
        };

        return summary;
    }
}

// Main execution
async function main() {
    const generator = new SurgicalSprintGenerator();
    
    try {
        const summary = await generator.generateSprints();
        
        console.log('\n================================================================================');
        console.log('⚔️ SURGICAL SPRINTS GENERATED - READY FOR MONDAY KICKOFF');
        console.log('================================================================================');
        
        console.log(`📊 Source: ${summary.source_gaps.toLocaleString()} gaps from v2.2 analysis`);
        console.log(`🎯 Generated: ${summary.generated_sprints} focused sprints`);
        console.log(`📋 Affected: ${summary.total_affected_queries.toLocaleString()} queries across 5 remedy classes`);
        console.log(`⏱️ Timeline: ${summary.estimated_timeline}`);
        
        console.log('\n📁 DELIVERABLES:');
        Object.entries(summary.deliverables).forEach(([name, path]) => {
            console.log(`   ${name}: ${path}`);
        });
        
        console.log('\n🚀 NEXT STEPS (Hand to Team):');
        summary.next_steps.forEach((step, i) => {
            console.log(`   ${i + 1}. ${step}`);
        });
        
        console.log('\n🎉 Ready to start Sprint #1 (Tail-Taming) Monday morning!');
        
    } catch (error) {
        console.error('❌ Sprint generation failed:', error);
        process.exit(1);
    }
}

main().catch(console.error);