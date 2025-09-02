#!/usr/bin/env node

/**
 * Benchmark Performance Badge Generator
 * 
 * Generates SVG badges showing current benchmark performance metrics
 * for use in README files and dashboards.
 */

import { promises as fs } from 'fs';
import path from 'path';

/**
 * SVG badge generator for benchmark metrics
 */
class BadgeGenerator {
  constructor(options = {}) {
    this.inputFile = options.input;
    this.outputFile = options.output;
    this.metric = options.metric || 'recall_at_10';
    this.format = options.format || 'percentage';
    this.label = options.label || this.getDefaultLabel(this.metric);
  }
  
  getDefaultLabel(metric) {
    const labels = {
      'recall_at_10': 'Recall@10',
      'recall_at_50': 'Recall@50',
      'ndcg_at_10': 'NDCG@10',
      'e2e_p95': 'P95 Latency',
      'error_rate': 'Error Rate'
    };
    return labels[metric] || metric;
  }
  
  async generateBadge() {
    console.log(`🏅 Generating badge for ${this.metric}...`);
    
    // Load benchmark summary
    const summary = await this.loadJsonFile(this.inputFile);
    const value = this.extractMetricValue(summary, this.metric);
    
    if (value === null || value === undefined) {
      throw new Error(`Metric ${this.metric} not found in summary data`);
    }
    
    // Format value for display
    const displayValue = this.formatValue(value, this.format);
    
    // Determine badge color based on performance
    const color = this.getColorForValue(value, this.metric);
    
    // Generate SVG
    const svg = this.createSvgBadge(this.label, displayValue, color);
    
    // Save badge
    await fs.writeFile(this.outputFile, svg);
    
    console.log(`✅ Badge generated: ${this.outputFile}`);
    console.log(`   ${this.label}: ${displayValue} (${color})`);
    
    return this.outputFile;
  }
  
  extractMetricValue(summary, metric) {
    switch (metric) {
      case 'recall_at_10':
        return summary.performance?.recall_at_10;
      case 'recall_at_50':
        return summary.performance?.recall_at_50;
      case 'ndcg_at_10':
        return summary.performance?.ndcg_at_10;
      case 'e2e_p95':
        return summary.latency?.e2e_p95;
      case 'error_rate':
        return summary.quality?.error_rate;
      default:
        return null;
    }
  }
  
  formatValue(value, format) {
    switch (format) {
      case 'percentage':
        return `${(value * 100).toFixed(1)}%`;
      case 'milliseconds':
        return `${value.toFixed(0)}ms`;
      case 'decimal':
        return value.toFixed(3);
      default:
        return value.toString();
    }
  }
  
  getColorForValue(value, metric) {
    // Define performance thresholds for color coding
    const thresholds = {
      recall_at_10: {
        excellent: 0.80,  // 80%+ = green
        good: 0.70,       // 70%+ = yellow
        poor: 0.60        // <60% = red
      },
      recall_at_50: {
        excellent: 0.90,  // 90%+ = green
        good: 0.80,       // 80%+ = yellow
        poor: 0.70        // <70% = red
      },
      ndcg_at_10: {
        excellent: 0.75,  // 75%+ = green
        good: 0.65,       // 65%+ = yellow
        poor: 0.55        // <55% = red
      },
      e2e_p95: {
        excellent: 150,   // <150ms = green
        good: 200,        // <200ms = yellow
        poor: 300         // >300ms = red (inverted logic)
      },
      error_rate: {
        excellent: 0.01,  // <1% = green
        good: 0.03,       // <3% = yellow
        poor: 0.05        // >5% = red (inverted logic)
      }
    };
    
    const threshold = thresholds[metric];
    if (!threshold) {
      return 'blue'; // Default color for unknown metrics
    }
    
    // Handle inverted metrics (where lower is better)
    const isInvertedMetric = metric.includes('latency') || metric.includes('error');
    
    if (isInvertedMetric) {
      if (value <= threshold.excellent) return 'brightgreen';
      if (value <= threshold.good) return 'yellow';
      return 'red';
    } else {
      if (value >= threshold.excellent) return 'brightgreen';
      if (value >= threshold.good) return 'yellow';
      return 'red';
    }
  }
  
  createSvgBadge(label, value, color) {
    // Calculate text widths (approximate)
    const labelWidth = label.length * 7 + 10;
    const valueWidth = value.length * 7 + 10;
    const totalWidth = labelWidth + valueWidth;
    
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${totalWidth}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a">
    <rect width="${totalWidth}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#a)">
    <path fill="#555" d="M0 0h${labelWidth}v20H0z"/>
    <path fill="${this.getColorHex(color)}" d="M${labelWidth} 0h${valueWidth}v20H${labelWidth}z"/>
    <path fill="url(#b)" d="M0 0h${totalWidth}v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110">
    <text x="${labelWidth/2 * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="${(labelWidth-10)*10}">${label}</text>
    <text x="${labelWidth/2 * 10}" y="140" transform="scale(.1)" textLength="${(labelWidth-10)*10}">${label}</text>
    <text x="${(labelWidth + valueWidth/2) * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="${(valueWidth-10)*10}">${value}</text>
    <text x="${(labelWidth + valueWidth/2) * 10}" y="140" transform="scale(.1)" textLength="${(valueWidth-10)*10}">${value}</text>
  </g>
</svg>`;
  }
  
  getColorHex(colorName) {
    const colors = {
      'brightgreen': '#4c1',
      'green': '#97CA00',
      'yellow': '#dfb317',
      'orange': '#fe7d37',
      'red': '#e05d44',
      'blue': '#007ec6',
      'lightgrey': '#9f9f9f'
    };
    return colors[colorName] || colors['blue'];
  }
  
  async loadJsonFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      throw new Error(`Failed to load ${filePath}: ${error.message}`);
    }
  }
}

/**
 * Command-line interface
 */
async function main() {
  const args = process.argv.slice(2);
  
  function getArg(name, defaultValue = null) {
    const index = args.findIndex(arg => arg === `--${name}`);
    return index !== -1 && index + 1 < args.length ? args[index + 1] : defaultValue;
  }
  
  function hasFlag(name) {
    return args.includes(`--${name}`);
  }
  
  try {
    const generator = new BadgeGenerator({
      input: getArg('input'),
      output: getArg('output'),
      metric: getArg('metric', 'recall_at_10'),
      format: getArg('format', 'percentage'),
      label: getArg('label')
    });
    
    if (!generator.inputFile) {
      console.error('❌ Missing required argument: --input');
      console.error('Usage: node generate-badge.js --input summary.json --output badge.svg [options]');
      console.error('');
      console.error('Options:');
      console.error('  --metric <name>     Metric to display (recall_at_10, ndcg_at_10, e2e_p95, error_rate)');
      console.error('  --format <type>     Value format (percentage, milliseconds, decimal)');
      console.error('  --label <text>      Custom label text');
      process.exit(1);
    }
    
    if (!generator.outputFile) {
      console.error('❌ Missing required argument: --output');
      process.exit(1);
    }
    
    await generator.generateBadge();
    
    console.log(`🎯 Badge generation completed successfully`);
    
  } catch (error) {
    console.error(`❌ Badge generation failed: ${error.message}`);
    if (hasFlag('verbose')) {
      console.error('Stack trace:', error.stack);
    }
    process.exit(1);
  }
}

// Execute main function if this script is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

export { BadgeGenerator };
