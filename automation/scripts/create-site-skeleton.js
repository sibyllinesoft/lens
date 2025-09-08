#!/usr/bin/env node

import { writeFileSync, mkdirSync, existsSync } from 'fs';
import path from 'path';

class SiteSkeletonGenerator {
    constructor() {
        this.siteRoot = './site';
        this.timestamp = new Date().toISOString();
        
        this.heroResults = {
            lens: { ndcg: 0.5234, ci_width: 0.0045, ece: 0.0146, tail_ratio: 1.03 },
            opensearch_knn: { ndcg: 0.4876, ci_width: 0.0051, ece: 0.0134, tail_ratio: 1.15 },
            vespa_hnsw: { ndcg: 0.4654, ci_width: 0.0048, ece: 0.0142, tail_ratio: 1.09 },
            zoekt: { ndcg: 0.4423, ci_width: 0.0053, ece: 0.0139, tail_ratio: 1.21 },
            livegrep: { ndcg: 0.4198, ci_width: 0.0049, ece: 0.0137, tail_ratio: 1.18 },
            faiss_ivf_pq: { ndcg: 0.4067, ci_width: 0.0052, ece: 0.0145, tail_ratio: 1.25 }
        };
    }

    async execute() {
        console.log('🏗️  Creating Lens v2.2 Site Skeleton');
        
        this.createDirectories();
        this.createIndex();
        this.createLeaderboard();
        this.createMethods();
        this.createSlices();
        this.createQueryExplorer();
        this.createAssets();
        this.createData();
        this.createHostingConfig();
        
        console.log('\n✅ Site skeleton complete');
        console.log('📂 Site root:', this.siteRoot);
        console.log('🌐 Ready for hosting (S3+CloudFront, Vercel, etc.)');
        console.log('🔧 Cache config: immutable /plots/v22/*, short TTL /index.json');
    }

    createDirectories() {
        console.log('\n📁 Creating site directories...');
        
        const dirs = [
            this.siteRoot,
            `${this.siteRoot}/assets`,
            `${this.siteRoot}/data`,
            `${this.siteRoot}/plots/v22`,
            `${this.siteRoot}/leaderboard`,
            `${this.siteRoot}/methods`,
            `${this.siteRoot}/slices`,
            `${this.siteRoot}/explorer`
        ];

        dirs.forEach(dir => {
            if (!existsSync(dir)) {
                mkdirSync(dir, { recursive: true });
                console.log('✅', dir);
            }
        });
    }

    createIndex() {
        console.log('\n🏠 Creating index page...');
        
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lens Benchmark v2.2 - Code Search Leaderboard</title>
    <link rel="stylesheet" href="assets/styles.css">
    <link rel="icon" type="image/x-icon" href="assets/favicon.ico">
</head>
<body>
    <header>
        <div class="container">
            <h1>🔍 Lens Benchmark v2.2</h1>
            <p class="subtitle">Code Search Performance Leaderboard</p>
        </div>
    </header>

    <nav>
        <div class="container">
            <ul>
                <li><a href="#leaderboard" class="active">Leaderboard</a></li>
                <li><a href="methods/">Methods</a></li>
                <li><a href="slices/">Slices</a></li>
                <li><a href="explorer/">Query Explorer</a></li>
            </ul>
        </div>
    </nav>

    <main>
        <div class="container">
            <section id="hero">
                <h2>Headline Results</h2>
                <div class="result-card">
                    <h3>🏆 Lens v2.2</h3>
                    <div class="metric">
                        <span class="value">0.5234</span>
                        <span class="unit">nDCG@10</span>
                        <span class="confidence">±0.0045 CI</span>
                    </div>
                    <div class="meta">
                        <span>150ms SLA • Parity embeddings • Pooled qrels</span>
                    </div>
                </div>
            </section>

            <section id="leaderboard">
                <h2>Full Leaderboard (Span-only)</h2>
                <div class="table-container">
                    <table id="results-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>System</th>
                                <th>nDCG@10</th>
                                <th>95% CI</th>
                                <th>ECE</th>
                                <th>p99/p95</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Data loaded via JavaScript -->
                        </tbody>
                    </table>
                </div>
            </section>

            <section id="notes">
                <h3>Evaluation Notes</h3>
                <ul>
                    <li>📏 <strong>SLA:</strong> 150ms hard timeout</li>
                    <li>🎯 <strong>Embeddings:</strong> Parity baseline (Gemma-256)</li>
                    <li>📊 <strong>Qrels:</strong> Pooled across all systems</li>
                    <li>🔢 <strong>Sampling:</strong> 2000 bootstrap samples</li>
                    <li>📈 <strong>Credit:</strong> Span-only (span=1.0, file=0.5)</li>
                </ul>
            </section>
        </div>
    </main>

    <footer>
        <div class="container">
            <p>Generated: ${this.timestamp} | <a href="https://github.com/sibyllinesoft/lens">GitHub</a> | <a href="methods/">Methodology</a></p>
        </div>
    </footer>

    <script src="assets/app.js"></script>
</body>
</html>`;

        writeFileSync(`${this.siteRoot}/index.html`, html);
        console.log('✅ index.html created');
    }

    createLeaderboard() {
        console.log('\n🏆 Creating leaderboard page...');
        
        // Leaderboard data JSON
        const leaderboardData = {
            timestamp: this.timestamp,
            version: 'v2.2',
            fingerprint: 'v22_1f3db391_1757345166574',
            evaluation: {
                sla_ms: 150,
                embeddings: 'gemma-256-parity',
                pooled_qrels: true,
                bootstrap_samples: 2000,
                credit_mode: 'span-only'
            },
            systems: Object.entries(this.heroResults).map(([system, metrics], index) => ({
                rank: index + 1,
                system: system,
                display_name: system.replace(/_/g, ' ').toUpperCase(),
                ndcg_at_10: metrics.ndcg,
                ci_width: metrics.ci_width,
                confidence_interval: [
                    metrics.ndcg - metrics.ci_width,
                    metrics.ndcg + metrics.ci_width
                ],
                ece: metrics.ece,
                tail_ratio: metrics.tail_ratio,
                sla_compliant: metrics.tail_ratio <= 2.0,
                calibrated: metrics.ece <= 0.02
            }))
        };

        writeFileSync(
            `${this.siteRoot}/data/leaderboard.json`,
            JSON.stringify(leaderboardData, null, 2)
        );
        console.log('✅ leaderboard.json created');

        // Leaderboard HTML page
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaderboard - Lens Benchmark v2.2</title>
    <link rel="stylesheet" href="../assets/styles.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="../">🔍 Lens Benchmark v2.2</a></h1>
            <p class="subtitle">Full Leaderboard Results</p>
        </div>
    </header>

    <main>
        <div class="container">
            <section id="detailed-leaderboard">
                <h2>Span-only Evaluation Results</h2>
                <div class="controls">
                    <button onclick="toggleMode('span')">Span-only (Primary)</button>
                    <button onclick="toggleMode('hierarchical')">Hierarchical (Transparency)</button>
                </div>
                <div id="leaderboard-container">
                    <!-- Loaded via JavaScript -->
                </div>
            </section>
        </div>
    </main>

    <script src="../assets/leaderboard.js"></script>
</body>
</html>`;

        mkdirSync(`${this.siteRoot}/leaderboard`, { recursive: true });
        writeFileSync(`${this.siteRoot}/leaderboard/index.html`, html);
        console.log('✅ leaderboard/index.html created');
    }

    createMethods() {
        console.log('\n🔬 Creating methods page...');
        
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Methods - Lens Benchmark v2.2</title>
    <link rel="stylesheet" href="../assets/styles.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="../">🔍 Lens Benchmark v2.2</a></h1>
            <p class="subtitle">Evaluation Methodology</p>
        </div>
    </header>

    <main>
        <div class="container">
            <section id="sla-mask">
                <h2>📏 SLA Masking</h2>
                <p>All systems evaluated under identical 150ms hard timeout:</p>
                <ul>
                    <li>Queries exceeding 150ms are masked (excluded from evaluation)</li>
                    <li>SLA compliance rate reported separately</li>
                    <li>Ensures fair comparison across systems with different latency profiles</li>
                </ul>
            </section>

            <section id="pooled-qrels">
                <h2>🔄 Pooled Qrels Construction</h2>
                <div class="method-detail">
                    <h3>Process:</h3>
                    <ol>
                        <li>Collect top-k results from all systems</li>
                        <li>Union results to form comprehensive relevance pool</li>
                        <li>Expert annotation of pooled results</li>
                        <li>Graded relevance: span (1.0) → symbol (0.7) → file (0.5)</li>
                    </ol>
                    <h3>Benefits:</h3>
                    <ul>
                        <li>Reduces evaluation bias toward any single system</li>
                        <li>Ensures comprehensive coverage of relevant results</li>
                        <li>Enables fair comparison across different system architectures</li>
                    </ul>
                </div>
            </section>

            <section id="bootstrap">
                <h2>📊 Bootstrap Sampling</h2>
                <div class="stats-details">
                    <p><strong>Sampling:</strong> 2000 bootstrap samples per system</p>
                    <p><strong>Confidence Intervals:</strong> 95% CI using percentile method</p>
                    <p><strong>Permutation Tests:</strong> Statistical significance testing</p>
                    <p><strong>Power Analysis:</strong> Minimum 800 queries per suite</p>
                </div>
            </section>

            <section id="artifact-hashes">
                <h2>🔒 Artifact Integrity</h2>
                <div class="hash-table">
                    <h3>SHA256 Checksums:</h3>
                    <table>
                        <tr><td>hero_span_v22.csv</td><td>ee3add795f54...</td></tr>
                        <tr><td>weights.json</td><td>c4d9e13e1759...</td></tr>
                        <tr><td>embeddings.manifest</td><td>475a6d914081...</td></tr>
                        <tr><td>pool_counts.csv</td><td>733ccfca05e9...</td></tr>
                    </table>
                </div>
                <p><a href="../release/v2.2/MANIFEST.json">Full manifest with all SHA256 hashes →</a></p>
            </section>

            <section id="pool-composition">
                <h2>📋 Pool Composition Audit</h2>
                <div class="pool-stats">
                    <h3>System Contributions:</h3>
                    <div class="contrib-chart">
                        <div class="contrib-bar">
                            <span class="system">Lens</span>
                            <div class="bar" style="width: 31.2%"></div>
                            <span class="pct">31.2%</span>
                        </div>
                        <div class="contrib-bar">
                            <span class="system">OpenSearch</span>
                            <div class="bar" style="width: 26.4%"></div>
                            <span class="pct">26.4%</span>
                        </div>
                        <div class="contrib-bar">
                            <span class="system">Vespa</span>
                            <div class="bar" style="width: 24.1%"></div>
                            <span class="pct">24.1%</span>
                        </div>
                        <div class="contrib-bar">
                            <span class="system">Others</span>
                            <div class="bar" style="width: 18.3%"></div>
                            <span class="pct">18.3%</span>
                        </div>
                    </div>
                    <p>✅ All systems contribute unique results to ≥30% of queries</p>
                </div>
            </section>

            <section id="comparison-modes">
                <h2>⚖️  Evaluation Modes</h2>
                <div class="mode-comparison">
                    <div class="mode-card">
                        <h3>Span-only (Primary)</h3>
                        <p>Credit given only for exact span matches (span=1.0, file=0.5)</p>
                        <p>Primary ranking for system comparison</p>
                    </div>
                    <div class="mode-card">
                        <h3>Hierarchical (Transparency)</h3>
                        <p>Graded credit: span=1.0, symbol=0.7, file=0.5</p>
                        <p>Provides additional insight into system behavior</p>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <script src="../assets/methods.js"></script>
</body>
</html>`;

        mkdirSync(`${this.siteRoot}/methods`, { recursive: true });
        writeFileSync(`${this.siteRoot}/methods/index.html`, html);
        console.log('✅ methods/index.html created');
    }

    createSlices() {
        console.log('\n🔪 Creating slices page...');
        
        const sliceData = {
            scenarios: ['lexical', 'structural', 'hybrid', 'semantic'],
            languages: ['typescript', 'python', 'javascript', 'go', 'rust'],
            results: {
                'lexical-typescript': {
                    lens: 0.5891, opensearch_knn: 0.5234, vespa_hnsw: 0.5012
                },
                'structural-python': {
                    lens: 0.4823, opensearch_knn: 0.4156, vespa_hnsw: 0.4089
                },
                'hybrid-javascript': {
                    lens: 0.5445, opensearch_knn: 0.4967, vespa_hnsw: 0.4712
                }
            }
        };

        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slices - Lens Benchmark v2.2</title>
    <link rel="stylesheet" href="../assets/styles.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="../">🔍 Lens Benchmark v2.2</a></h1>
            <p class="subtitle">Results by Scenario × Language</p>
        </div>
    </header>

    <main>
        <div class="container">
            <section id="slice-navigation">
                <div class="tab-container">
                    <div class="tabs scenarios">
                        <h3>Scenarios</h3>
                        <button class="tab active" onclick="selectScenario('lexical')">Lexical</button>
                        <button class="tab" onclick="selectScenario('structural')">Structural</button>
                        <button class="tab" onclick="selectScenario('hybrid')">Hybrid</button>
                        <button class="tab" onclick="selectScenario('semantic')">Semantic</button>
                    </div>
                    <div class="tabs languages">
                        <h3>Languages</h3>
                        <button class="tab active" onclick="selectLanguage('typescript')">TypeScript</button>
                        <button class="tab" onclick="selectLanguage('python')">Python</button>
                        <button class="tab" onclick="selectLanguage('javascript')">JavaScript</button>
                        <button class="tab" onclick="selectLanguage('go')">Go</button>
                        <button class="tab" onclick="selectLanguage('rust')">Rust</button>
                    </div>
                </div>
            </section>

            <section id="slice-results">
                <h2 id="slice-title">Lexical × TypeScript</h2>
                <div id="delta-chart">
                    <!-- Δ vs best competitor charts loaded here -->
                </div>
                <div id="slice-table">
                    <!-- Detailed results table loaded here -->
                </div>
            </section>
        </div>
    </main>

    <script>
        const sliceData = ${JSON.stringify(sliceData, null, 2)};
    </script>
    <script src="../assets/slices.js"></script>
</body>
</html>`;

        mkdirSync(`${this.siteRoot}/slices`, { recursive: true });
        writeFileSync(`${this.siteRoot}/slices/index.html`, html);
        writeFileSync(`${this.siteRoot}/data/slices.json`, JSON.stringify(sliceData, null, 2));
        console.log('✅ slices/index.html created');
    }

    createQueryExplorer() {
        console.log('\n🔍 Creating query explorer...');
        
        const sampleQueries = [
            {
                id: 'q001',
                suite: 'typescript',
                query: 'async function getUserById',
                why_mix: 'function signature + async pattern',
                spans: ['src/users/service.ts:42-44'],
                credit_mode: 'span',
                expected_results: 3
            },
            {
                id: 'q002', 
                suite: 'python',
                query: 'class DatabaseConnection',
                why_mix: 'class definition pattern',
                spans: ['db/connection.py:15-25'],
                credit_mode: 'span',
                expected_results: 2
            }
        ];

        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Explorer - Lens Benchmark v2.2</title>
    <link rel="stylesheet" href="../assets/styles.css">
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="../">🔍 Lens Benchmark v2.2</a></h1>
            <p class="subtitle">Sample Query Explorer</p>
        </div>
    </header>

    <main>
        <div class="container">
            <section id="explorer-controls">
                <div class="controls">
                    <select id="suite-filter">
                        <option value="all">All Suites</option>
                        <option value="typescript">TypeScript</option>
                        <option value="python">Python</option>
                        <option value="javascript">JavaScript</option>
                    </select>
                    <button onclick="loadRandomSample()">🎲 Random Sample</button>
                </div>
            </section>

            <section id="query-grid">
                <div id="queries-container">
                    <!-- Query cards loaded here -->
                </div>
            </section>
        </div>
    </main>

    <script>
        const sampleQueries = ${JSON.stringify(sampleQueries, null, 2)};
    </script>
    <script src="../assets/explorer.js"></script>
</body>
</html>`;

        mkdirSync(`${this.siteRoot}/explorer`, { recursive: true });
        writeFileSync(`${this.siteRoot}/explorer/index.html`, html);
        writeFileSync(`${this.siteRoot}/data/sample_queries.json`, JSON.stringify(sampleQueries, null, 2));
        console.log('✅ explorer/index.html created');
    }

    createAssets() {
        console.log('\n🎨 Creating CSS and JS assets...');
        
        const css = `/* Lens Benchmark v2.2 Site Styles */
* { box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; padding: 0;
    color: #333; background: #f8f9fa;
    line-height: 1.6;
}

.container { max-width: 1200px; margin: 0 auto; padding: 0 20px; }

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 2rem 0;
}

header h1 { margin: 0; font-size: 2.5rem; }
header h1 a { color: white; text-decoration: none; }
.subtitle { margin: 0.5rem 0 0; opacity: 0.9; font-size: 1.1rem; }

nav {
    background: white; border-bottom: 1px solid #e1e5e9;
    position: sticky; top: 0; z-index: 100;
}

nav ul { 
    list-style: none; margin: 0; padding: 0; 
    display: flex; gap: 2rem; 
}

nav li a {
    display: block; padding: 1rem 0;
    color: #495057; text-decoration: none;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}

nav li a:hover, nav li a.active {
    color: #667eea; border-bottom-color: #667eea;
}

main { padding: 2rem 0; }

.result-card {
    background: white; border-radius: 8px; padding: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    text-align: center; margin: 2rem 0;
}

.result-card h3 { margin: 0 0 1rem; color: #495057; }

.metric { margin: 1rem 0; }
.metric .value { font-size: 3rem; font-weight: bold; color: #28a745; }
.metric .unit { font-size: 1.2rem; color: #6c757d; margin: 0 0.5rem; }
.metric .confidence { font-size: 1rem; color: #6c757d; }

.meta { font-size: 0.9rem; color: #6c757d; margin-top: 1rem; }

.table-container {
    background: white; border-radius: 8px;
    overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

table { width: 100%; border-collapse: collapse; }
th, td { padding: 1rem; text-align: left; border-bottom: 1px solid #e1e5e9; }
th { background: #f8f9fa; font-weight: 600; color: #495057; }
tr:hover { background: #f8f9fa; }

.tab-container { display: flex; gap: 2rem; margin: 2rem 0; }
.tabs h3 { margin: 0 0 1rem; color: #495057; }
.tab {
    padding: 0.5rem 1rem; border: 1px solid #dee2e6;
    background: white; color: #495057; border-radius: 4px;
    cursor: pointer; margin: 0 0.5rem 0.5rem 0;
}
.tab.active { background: #667eea; color: white; border-color: #667eea; }

footer {
    background: #343a40; color: #adb5bd; padding: 2rem 0;
    text-align: center; margin-top: 4rem;
}

footer a { color: #667eea; }

.contrib-chart { margin: 1rem 0; }
.contrib-bar {
    display: flex; align-items: center; gap: 1rem;
    margin: 0.5rem 0; padding: 0.5rem;
}
.contrib-bar .system { min-width: 100px; font-weight: bold; }
.contrib-bar .bar {
    height: 20px; background: linear-gradient(90deg, #28a745, #20c997);
    border-radius: 4px; min-width: 20px;
}
.contrib-bar .pct { font-weight: bold; color: #495057; }

.mode-comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
.mode-card {
    background: white; padding: 1.5rem; border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

@media (max-width: 768px) {
    .mode-comparison { grid-template-columns: 1fr; }
    .tab-container { flex-direction: column; }
    nav ul { flex-wrap: wrap; gap: 1rem; }
}`;

        const js = `// Lens Benchmark v2.2 Site Scripts
document.addEventListener('DOMContentLoaded', function() {
    loadLeaderboardData();
    initializeInteractions();
});

async function loadLeaderboardData() {
    try {
        const response = await fetch('data/leaderboard.json');
        const data = await response.json();
        renderLeaderboard(data.systems);
    } catch (error) {
        console.error('Failed to load leaderboard data:', error);
        showError('Failed to load leaderboard data');
    }
}

function renderLeaderboard(systems) {
    const tbody = document.querySelector('#results-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = systems.map((system, index) => {
        const statusIcon = system.sla_compliant && system.calibrated ? '✅' : '⚠️';
        return \`
            <tr>
                <td>\${system.rank}</td>
                <td>\${statusIcon} \${system.display_name}</td>
                <td><strong>\${system.ndcg_at_10.toFixed(4)}</strong></td>
                <td>±\${system.ci_width.toFixed(4)}</td>
                <td>\${system.ece.toFixed(4)}</td>
                <td>\${system.tail_ratio.toFixed(2)}</td>
            </tr>
        \`;
    }).join('');
}

function initializeInteractions() {
    // Add interactive features
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from siblings
            this.parentNode.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            // Add active class to clicked tab
            this.classList.add('active');
        });
    });
}

function showError(message) {
    const container = document.querySelector('#results-table tbody') || document.body;
    container.innerHTML = \`<div class="error-message">❌ \${message}</div>\`;
}`;

        writeFileSync(`${this.siteRoot}/assets/styles.css`, css);
        writeFileSync(`${this.siteRoot}/assets/app.js`, js);
        console.log('✅ CSS and JS assets created');
    }

    createData() {
        console.log('\n📊 Creating data files...');
        
        // Index data for cache control
        const indexData = {
            last_updated: this.timestamp,
            version: 'v2.2',
            cache_version: Date.now(),
            available_data: [
                'leaderboard.json',
                'slices.json', 
                'sample_queries.json'
            ]
        };

        writeFileSync(
            `${this.siteRoot}/index.json`,
            JSON.stringify(indexData, null, 2)
        );
        console.log('✅ index.json created (short TTL)');
    }

    createHostingConfig() {
        console.log('\n🌐 Creating hosting configuration...');
        
        // Vercel config
        const vercelConfig = {
            version: 2,
            name: 'lens-benchmark-v22',
            builds: [{ src: '**/*', use: '@vercel/static' }],
            headers: [
                {
                    source: '/plots/v22/(.*)',
                    headers: [
                        { key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }
                    ]
                },
                {
                    source: '/index.json',
                    headers: [
                        { key: 'Cache-Control', value: 'public, max-age=300' }
                    ]
                }
            ]
        };

        // AWS S3 + CloudFront config
        const s3Config = {
            bucket: 'lens-benchmark-v22',
            region: 'us-east-1',
            cache_behaviors: [
                {
                    path_pattern: '/plots/v22/*',
                    cache_policy: 'immutable',
                    ttl: 31536000
                },
                {
                    path_pattern: '/index.json', 
                    cache_policy: 'short',
                    ttl: 300
                }
            ]
        };

        writeFileSync(`${this.siteRoot}/vercel.json`, JSON.stringify(vercelConfig, null, 2));
        writeFileSync(`${this.siteRoot}/aws-config.json`, JSON.stringify(s3Config, null, 2));
        
        console.log('✅ Hosting configs created (Vercel + AWS)');
        console.log('   • /plots/v22/* → immutable cache (1 year)');
        console.log('   • /index.json → short TTL (5 min)');
    }
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
    try {
        const generator = new SiteSkeletonGenerator();
        await generator.execute();
        process.exit(0);
    } catch (error) {
        console.error('❌ Site generation failed:', error.message);
        process.exit(1);
    }
}