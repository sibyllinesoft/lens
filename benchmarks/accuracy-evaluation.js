#!/usr/bin/env node

const http = require('http');
const fs = require('fs').promises;
const path = require('path');

/**
 * Comprehensive Accuracy/ML Benchmark
 * Evaluates search relevance, ranking quality, and ML model performance
 */
class AccuracyEvaluationBenchmark {
  constructor() {
    this.baseUrl = 'http://localhost:3001';
    this.goldStandard = this.createGoldStandardDataset();
    this.results = [];
  }

  createGoldStandardDataset() {
    return [
      // Exact match queries - should have perfect precision
      {
        query: 'UserService',
        expected_results: [
          { file: 'src/example.ts', relevance: 1.0, reason: 'Class definition' },
          { file: 'sample-code/user-service.ts', relevance: 1.0, reason: 'Class definition' }
        ],
        query_type: 'exact_match',
        expected_precision: 1.0,
        expected_recall: 0.8,
        min_results: 1
      },
      
      // Function search queries
      {
        query: 'findUser',
        expected_results: [
          { file: 'src/example.ts', relevance: 1.0, reason: 'Method definition' },
          { file: 'sample-code/user-service.ts', relevance: 1.0, reason: 'Method definition' }
        ],
        query_type: 'function_search',
        expected_precision: 0.8,
        expected_recall: 0.7,
        min_results: 1
      },
      
      // Semantic queries - test ML understanding
      {
        query: 'user management',
        expected_results: [
          { file: 'src/example.ts', relevance: 0.9, reason: 'UserService class handles user operations' },
          { file: 'sample-code/user-service.ts', relevance: 0.9, reason: 'User management functionality' }
        ],
        query_type: 'semantic',
        expected_precision: 0.6,
        expected_recall: 0.5,
        min_results: 1
      },
      
      // Pattern matching queries
      {
        query: 'async function',
        expected_results: [
          { file: 'src/example.ts', relevance: 1.0, reason: 'Contains async methods' }
        ],
        query_type: 'pattern_match',
        expected_precision: 0.9,
        expected_recall: 0.7,
        min_results: 1
      },
      
      // Fuzzy search queries
      {
        query: 'usrSrvce', // Intentional typos
        expected_results: [
          { file: 'src/example.ts', relevance: 0.8, reason: 'UserService fuzzy match' },
          { file: 'sample-code/user-service.ts', relevance: 0.8, reason: 'UserService fuzzy match' }
        ],
        query_type: 'fuzzy_search',
        expected_precision: 0.7,
        expected_recall: 0.6,
        min_results: 1
      },
      
      // Type-based queries
      {
        query: 'interface User',
        expected_results: [
          { file: 'src/example.ts', relevance: 1.0, reason: 'User interface definition' }
        ],
        query_type: 'type_search',
        expected_precision: 1.0,
        expected_recall: 0.9,
        min_results: 1
      },
      
      // Complex semantic queries
      {
        query: 'email validation logic',
        expected_results: [
          { file: 'src/example.ts', relevance: 0.9, reason: 'validateEmail function' },
          { file: 'sample-code/utils.js', relevance: 0.9, reason: 'validateEmail function' }
        ],
        query_type: 'complex_semantic',
        expected_precision: 0.5,
        expected_recall: 0.4,
        min_results: 1
      },
      
      // Cross-file relationship queries
      {
        query: 'user persistence',
        expected_results: [
          { file: 'src/example.ts', relevance: 0.8, reason: 'saveToFile/loadFromFile methods' }
        ],
        query_type: 'relationship',
        expected_precision: 0.4,
        expected_recall: 0.3,
        min_results: 1
      }
    ];
  }

  async makeSearchRequest(query, mode = 'hybrid', fuzzy = 0.7) {
    return new Promise((resolve, reject) => {
      const data = JSON.stringify({ q: query, mode, fuzzy });
      const options = {
        hostname: 'localhost',
        port: 3001,
        path: '/search',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data)
        }
      };

      const req = http.request(options, (res) => {
        let responseData = '';
        res.on('data', (chunk) => responseData += chunk);
        res.on('end', () => {
          try {
            const result = JSON.parse(responseData);
            resolve({
              statusCode: res.statusCode,
              data: result,
              query: { q: query, mode, fuzzy }
            });
          } catch (e) {
            resolve({
              statusCode: res.statusCode,
              error: responseData,
              query: { q: query, mode, fuzzy }
            });
          }
        });
      });

      req.on('error', reject);
      req.write(data);
      req.end();
    });
  }

  calculatePrecision(actualResults, expectedResults) {
    if (actualResults.length === 0) return 0;
    
    let relevantRetrieved = 0;
    for (const actual of actualResults) {
      const expectedMatch = expectedResults.find(exp => 
        actual.file && actual.file.includes(path.basename(exp.file))
      );
      if (expectedMatch && expectedMatch.relevance >= 0.5) {
        relevantRetrieved++;
      }
    }
    
    return relevantRetrieved / actualResults.length;
  }

  calculateRecall(actualResults, expectedResults) {
    if (expectedResults.length === 0) return 1;
    
    let relevantRetrieved = 0;
    for (const expected of expectedResults) {
      const actualMatch = actualResults.find(actual => 
        actual.file && actual.file.includes(path.basename(expected.file))
      );
      if (actualMatch) {
        relevantRetrieved++;
      }
    }
    
    return relevantRetrieved / expectedResults.length;
  }

  calculateF1Score(precision, recall) {
    if (precision + recall === 0) return 0;
    return (2 * precision * recall) / (precision + recall);
  }

  calculateMRR(actualResults, expectedResults) {
    // Mean Reciprocal Rank - position of first relevant result
    for (let i = 0; i < actualResults.length; i++) {
      const result = actualResults[i];
      const expectedMatch = expectedResults.find(exp => 
        result.file && result.file.includes(path.basename(exp.file))
      );
      if (expectedMatch && expectedMatch.relevance >= 0.5) {
        return 1 / (i + 1);
      }
    }
    return 0;
  }

  calculateNDCG(actualResults, expectedResults, k = 10) {
    // Normalized Discounted Cumulative Gain
    const dcg = actualResults.slice(0, k).reduce((sum, result, index) => {
      const expectedMatch = expectedResults.find(exp => 
        result.file && result.file.includes(path.basename(exp.file))
      );
      const relevance = expectedMatch ? expectedMatch.relevance : 0;
      return sum + relevance / Math.log2(index + 2);
    }, 0);
    
    // Ideal DCG (if results were perfectly ordered)
    const sortedExpected = [...expectedResults].sort((a, b) => b.relevance - a.relevance);
    const idcg = sortedExpected.slice(0, k).reduce((sum, exp, index) => {
      return sum + exp.relevance / Math.log2(index + 2);
    }, 0);
    
    return idcg === 0 ? 0 : dcg / idcg;
  }

  async evaluateQuery(goldStandardItem) {
    const { query, expected_results, query_type, expected_precision, expected_recall, min_results } = goldStandardItem;
    
    console.log(`\nEvaluating ${query_type}: "${query}"`);
    
    // Test different search modes
    const modes = ['lex', 'struct', 'hybrid'];
    const results = {};
    
    for (const mode of modes) {
      try {
        const response = await this.makeSearchRequest(query, mode);
        
        if (response.statusCode === 200 && response.data && response.data.results) {
          const actualResults = response.data.results;
          
          // Calculate metrics
          const precision = this.calculatePrecision(actualResults, expected_results);
          const recall = this.calculateRecall(actualResults, expected_results);
          const f1_score = this.calculateF1Score(precision, recall);
          const mrr = this.calculateMRR(actualResults, expected_results);
          const ndcg = this.calculateNDCG(actualResults, expected_results);
          
          results[mode] = {
            total_results: actualResults.length,
            metrics: { precision, recall, f1_score, mrr, ndcg },
            meets_min_results: actualResults.length >= min_results,
            meets_precision_target: precision >= expected_precision,
            meets_recall_target: recall >= expected_recall,
            performance: response.data.performance || {},
            sample_results: actualResults.slice(0, 3).map(r => ({
              file: r.file,
              score: r.score,
              line: r.line
            }))
          };
          
          console.log(`  ${mode}: P=${precision.toFixed(3)}, R=${recall.toFixed(3)}, F1=${f1_score.toFixed(3)}, MRR=${mrr.toFixed(3)}, NDCG=${ndcg.toFixed(3)} (${actualResults.length} results)`);
        } else {
          results[mode] = {
            error: response.error || 'No results',
            statusCode: response.statusCode,
            total_results: 0,
            metrics: { precision: 0, recall: 0, f1_score: 0, mrr: 0, ndcg: 0 }
          };
          console.log(`  ${mode}: ERROR - ${results[mode].error}`);
        }
      } catch (error) {
        results[mode] = {
          error: error.message,
          total_results: 0,
          metrics: { precision: 0, recall: 0, f1_score: 0, mrr: 0, ndcg: 0 }
        };
        console.log(`  ${mode}: ERROR - ${error.message}`);
      }
      
      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Determine best mode for this query type
    const bestMode = Object.keys(results).reduce((best, mode) => {
      const currentF1 = results[mode].metrics?.f1_score || 0;
      const bestF1 = results[best]?.metrics?.f1_score || 0;
      return currentF1 > bestF1 ? mode : best;
    }, modes[0]);
    
    return {
      query,
      query_type,
      expected: { precision: expected_precision, recall: expected_recall, min_results },
      results,
      best_mode: bestMode,
      best_performance: results[bestMode]
    };
  }

  async runFullAccuracyEvaluation() {
    console.log('🎯 Starting Comprehensive Accuracy/ML Evaluation Benchmark');
    console.log('=' .repeat(70));
    
    const report = {
      timestamp: new Date().toISOString(),
      server: this.baseUrl,
      total_queries: this.goldStandard.length,
      query_evaluations: [],
      summary: {
        by_query_type: {},
        by_search_mode: {},
        overall: {}
      }
    };
    
    // Evaluate each query in the gold standard
    for (const goldStandardItem of this.goldStandard) {
      const evaluation = await this.evaluateQuery(goldStandardItem);
      report.query_evaluations.push(evaluation);
    }
    
    // Calculate summary statistics
    this.calculateSummaryStatistics(report);
    
    return report;
  }

  calculateSummaryStatistics(report) {
    const { query_evaluations } = report;
    
    // Group by query type
    const queryTypeGroups = {};
    const modeGroups = { lex: [], struct: [], hybrid: [] };
    
    for (const evaluation of query_evaluations) {
      const { query_type, results } = evaluation;
      
      if (!queryTypeGroups[query_type]) {
        queryTypeGroups[query_type] = [];
      }
      queryTypeGroups[query_type].push(evaluation);
      
      // Group by mode
      for (const [mode, result] of Object.entries(results)) {
        if (result.metrics) {
          modeGroups[mode].push(result.metrics);
        }
      }
    }
    
    // Calculate averages by query type
    for (const [queryType, evaluations] of Object.entries(queryTypeGroups)) {
      const bestResults = evaluations.map(e => e.best_performance?.metrics || {});
      report.summary.by_query_type[queryType] = this.calculateAverageMetrics(bestResults);
    }
    
    // Calculate averages by search mode
    for (const [mode, metricsArray] of Object.entries(modeGroups)) {
      report.summary.by_search_mode[mode] = this.calculateAverageMetrics(metricsArray);
    }
    
    // Overall statistics (using best mode for each query)
    const allBestMetrics = query_evaluations.map(e => e.best_performance?.metrics || {});
    report.summary.overall = this.calculateAverageMetrics(allBestMetrics);
    
    // Success rates
    report.summary.success_rates = {
      queries_with_results: query_evaluations.filter(e => 
        e.best_performance && e.best_performance.total_results > 0
      ).length / query_evaluations.length,
      
      queries_meeting_precision_target: query_evaluations.filter(e => 
        e.best_performance && e.best_performance.meets_precision_target
      ).length / query_evaluations.length,
      
      queries_meeting_recall_target: query_evaluations.filter(e => 
        e.best_performance && e.best_performance.meets_recall_target
      ).length / query_evaluations.length
    };
  }

  calculateAverageMetrics(metricsArray) {
    if (metricsArray.length === 0) {
      return { precision: 0, recall: 0, f1_score: 0, mrr: 0, ndcg: 0, count: 0 };
    }
    
    const sums = metricsArray.reduce((acc, metrics) => ({
      precision: acc.precision + (metrics.precision || 0),
      recall: acc.recall + (metrics.recall || 0),
      f1_score: acc.f1_score + (metrics.f1_score || 0),
      mrr: acc.mrr + (metrics.mrr || 0),
      ndcg: acc.ndcg + (metrics.ndcg || 0)
    }), { precision: 0, recall: 0, f1_score: 0, mrr: 0, ndcg: 0 });
    
    const count = metricsArray.length;
    return {
      precision: sums.precision / count,
      recall: sums.recall / count,
      f1_score: sums.f1_score / count,
      mrr: sums.mrr / count,
      ndcg: sums.ndcg / count,
      count
    };
  }

  generateReport(benchmarkData) {
    const lines = [];
    lines.push('# Lens Search Engine - Accuracy/ML Evaluation Report');
    lines.push('');
    lines.push(`**Timestamp**: ${benchmarkData.timestamp}`);
    lines.push(`**Server**: ${benchmarkData.server}`);
    lines.push(`**Total Queries Evaluated**: ${benchmarkData.total_queries}`);
    lines.push('');
    
    // Overall Performance Summary
    lines.push('## Overall Performance Summary');
    lines.push('');
    const overall = benchmarkData.summary.overall;
    const successRates = benchmarkData.summary.success_rates;
    
    lines.push(`- **Average Precision**: ${overall.precision.toFixed(3)}`);
    lines.push(`- **Average Recall**: ${overall.recall.toFixed(3)}`);
    lines.push(`- **Average F1 Score**: ${overall.f1_score.toFixed(3)}`);
    lines.push(`- **Average MRR**: ${overall.mrr.toFixed(3)}`);
    lines.push(`- **Average NDCG**: ${overall.ndcg.toFixed(3)}`);
    lines.push('');
    lines.push(`- **Queries with Results**: ${(successRates.queries_with_results * 100).toFixed(1)}%`);
    lines.push(`- **Precision Target Achievement**: ${(successRates.queries_meeting_precision_target * 100).toFixed(1)}%`);
    lines.push(`- **Recall Target Achievement**: ${(successRates.queries_meeting_recall_target * 100).toFixed(1)}%`);
    lines.push('');
    
    // Performance by Search Mode
    lines.push('## Performance by Search Mode');
    lines.push('');
    
    const modes = ['lex', 'struct', 'hybrid'];
    for (const mode of modes) {
      const modeData = benchmarkData.summary.by_search_mode[mode];
      lines.push(`### ${mode.toUpperCase()} Mode`);
      lines.push(`- **Precision**: ${modeData.precision.toFixed(3)}`);
      lines.push(`- **Recall**: ${modeData.recall.toFixed(3)}`);
      lines.push(`- **F1 Score**: ${modeData.f1_score.toFixed(3)}`);
      lines.push(`- **MRR**: ${modeData.mrr.toFixed(3)}`);
      lines.push(`- **NDCG**: ${modeData.ndcg.toFixed(3)}`);
      lines.push(`- **Query Count**: ${modeData.count}`);
      lines.push('');
    }
    
    // Performance by Query Type
    lines.push('## Performance by Query Type');
    lines.push('');
    
    for (const [queryType, typeData] of Object.entries(benchmarkData.summary.by_query_type)) {
      lines.push(`### ${queryType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}`);
      lines.push(`- **Precision**: ${typeData.precision.toFixed(3)}`);
      lines.push(`- **Recall**: ${typeData.recall.toFixed(3)}`);
      lines.push(`- **F1 Score**: ${typeData.f1_score.toFixed(3)}`);
      lines.push(`- **MRR**: ${typeData.mrr.toFixed(3)}`);
      lines.push(`- **NDCG**: ${typeData.ndcg.toFixed(3)}`);
      lines.push('');
    }
    
    // Detailed Query Results
    lines.push('## Detailed Query Analysis');
    lines.push('');
    
    for (const evaluation of benchmarkData.query_evaluations) {
      const bestMetrics = evaluation.best_performance?.metrics || {};
      lines.push(`### "${evaluation.query}" (${evaluation.query_type})`);
      lines.push(`- **Best Mode**: ${evaluation.best_mode}`);
      lines.push(`- **Precision**: ${bestMetrics.precision?.toFixed(3) || 'N/A'}`);
      lines.push(`- **Recall**: ${bestMetrics.recall?.toFixed(3) || 'N/A'}`);
      lines.push(`- **F1 Score**: ${bestMetrics.f1_score?.toFixed(3) || 'N/A'}`);
      lines.push(`- **Results Found**: ${evaluation.best_performance?.total_results || 0}`);
      
      const targets = evaluation.expected;
      const performance = evaluation.best_performance;
      if (performance) {
        lines.push(`- **Meets Targets**: Precision ${performance.meets_precision_target ? '✅' : '❌'} (${targets.precision.toFixed(2)} target), Recall ${performance.meets_recall_target ? '✅' : '❌'} (${targets.recall.toFixed(2)} target)`);
      }
      lines.push('');
    }
    
    // ML Model Performance Analysis
    lines.push('## ML Model Performance Analysis');
    lines.push('');
    
    // Grade the model performance
    const overallF1 = overall.f1_score;
    let grade, interpretation;
    
    if (overallF1 >= 0.8) {
      grade = 'A (Excellent)';
      interpretation = 'Search engine demonstrates strong semantic understanding and relevance ranking.';
    } else if (overallF1 >= 0.6) {
      grade = 'B (Good)';
      interpretation = 'Search engine performs well with room for improvement in semantic queries.';
    } else if (overallF1 >= 0.4) {
      grade = 'C (Fair)';
      interpretation = 'Search engine handles basic queries but struggles with complex semantic understanding.';
    } else {
      grade = 'D (Needs Improvement)';
      interpretation = 'Search engine requires significant optimization for production use.';
    }
    
    lines.push(`- **Overall Grade**: ${grade}`);
    lines.push(`- **Interpretation**: ${interpretation}`);
    lines.push('');
    
    // Recommendations
    lines.push('## Recommendations for Improvement');
    lines.push('');
    
    const recommendations = [];
    
    if (overall.precision < 0.7) {
      recommendations.push('- **Precision Enhancement**: Consider improving result filtering and ranking algorithms');
    }
    
    if (overall.recall < 0.6) {
      recommendations.push('- **Recall Enhancement**: Expand indexing coverage and improve query expansion techniques');
    }
    
    if (overall.mrr < 0.5) {
      recommendations.push('- **Ranking Optimization**: Improve result ordering to surface most relevant results first');
    }
    
    // Mode-specific recommendations
    const bestMode = Object.keys(benchmarkData.summary.by_search_mode).reduce((best, mode) => {
      const currentF1 = benchmarkData.summary.by_search_mode[mode].f1_score;
      const bestF1 = benchmarkData.summary.by_search_mode[best]?.f1_score || 0;
      return currentF1 > bestF1 ? mode : best;
    });
    
    recommendations.push(`- **Default Search Mode**: Consider using '${bestMode}' as default mode (best overall performance)`);
    
    // Query type specific recommendations
    const poorPerformingTypes = Object.entries(benchmarkData.summary.by_query_type)
      .filter(([_, data]) => data.f1_score < 0.4)
      .map(([type, _]) => type);
    
    if (poorPerformingTypes.length > 0) {
      recommendations.push(`- **Query Type Focus**: Improve performance for: ${poorPerformingTypes.join(', ')}`);
    }
    
    if (recommendations.length === 0) {
      recommendations.push('- **System Performance**: Search accuracy meets or exceeds targets across all metrics');
    }
    
    recommendations.forEach(rec => lines.push(rec));
    
    return lines.join('\n');
  }

  async saveBenchmarkResults(data, reportText) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Save raw JSON data
    await fs.writeFile(
      `benchmark-accuracy-${timestamp}.json`,
      JSON.stringify(data, null, 2)
    );
    
    // Save readable report
    await fs.writeFile(
      `benchmark-accuracy-report-${timestamp}.md`,
      reportText
    );
    
    console.log(`\n📊 Results saved:`);
    console.log(`   - Raw data: benchmark-accuracy-${timestamp}.json`);
    console.log(`   - Report: benchmark-accuracy-report-${timestamp}.md`);
  }
}

// Run the benchmark if this script is executed directly
if (require.main === module) {
  const benchmark = new AccuracyEvaluationBenchmark();
  
  benchmark.runFullAccuracyEvaluation()
    .then(async (results) => {
      const report = benchmark.generateReport(results);
      console.log('\n' + '='.repeat(70));
      console.log('ACCURACY EVALUATION COMPLETE');
      console.log('='.repeat(70));
      console.log(report);
      
      await benchmark.saveBenchmarkResults(results, report);
    })
    .catch(error => {
      console.error('Accuracy evaluation failed:', error);
      process.exit(1);
    });
}

module.exports = AccuracyEvaluationBenchmark;