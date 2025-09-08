const express = require('express');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.COMBY_SERVER_PORT || 8081;
const corpusPath = process.env.CORPUS_PATH || '/datasets';

app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  const corpusFiles = countCorpusFiles();
  res.json({
    status: 'healthy',
    system: 'comby',
    version: getCombyVersion(),
    corpus_path: corpusPath,
    corpus_files: corpusFiles
  });
});

// Search endpoint
app.post('/search', async (req, res) => {
  const { query, match_template, rewrite_template, max_results = 50, language } = req.body;
  const queryId = uuidv4();
  const startTime = Date.now();
  
  try {
    // Build comby command
    const cmd = buildCombyCommand(query, match_template, rewrite_template, language, max_results);
    
    // Execute comby
    exec(cmd, { cwd: corpusPath, timeout: 30000 }, (error, stdout, stderr) => {
      const latencyMs = Date.now() - startTime;
      const slaViolated = latencyMs > 150;
      
      if (error) {
        console.error(`Comby execution error: ${error}`);
        return res.status(500).json({
          query_id: queryId,
          system: 'comby',
          error: error.message,
          latency_ms: latencyMs,
          sla_violated: slaViolated
        });
      }
      
      // Parse comby output
      const results = parseCombyOutput(stdout);
      
      const response = {
        query_id: queryId,
        system: 'comby',
        version: getCombyVersion(),
        latency_ms: latencyMs,
        total_hits: results.length,
        results: results.slice(0, max_results),
        sla_violated: slaViolated
      };
      
      console.log(`Search completed: query='${query}', hits=${response.total_hits}, latency=${latencyMs}ms, sla_violated=${slaViolated}`);
      res.json(response);
    });
    
  } catch (err) {
    const latencyMs = Date.now() - startTime;
    res.status(500).json({
      query_id: queryId,
      system: 'comby',
      error: err.message,
      latency_ms: latencyMs,
      sla_violated: latencyMs > 150
    });
  }
});

function buildCombyCommand(query, matchTemplate, rewriteTemplate, language, maxResults) {
  let cmd = 'comby';
  
  if (matchTemplate) {
    cmd += ` '${matchTemplate}'`;
  } else {
    // Use query as match template for simple structural search
    cmd += ` '${query}'`;
  }
  
  if (rewriteTemplate) {
    cmd += ` '${rewriteTemplate}'`;
  } else {
    cmd += ` ''`; // Empty rewrite for search-only
  }
  
  // Add language matcher if specified
  if (language) {
    cmd += ` -matcher ${language}`;
  } else {
    cmd += ` -matcher .generic`; // Default generic matcher
  }
  
  // Add JSON output and directory
  cmd += ' -json-lines .';
  
  return cmd;
}

function parseCombyOutput(stdout) {
  const results = [];
  const lines = stdout.split('\n').filter(line => line.trim());
  
  for (const line of lines) {
    try {
      const match = JSON.parse(line);
      if (match.uri && match.range) {
        results.push({
          file_path: match.uri,
          line_number: match.range.start.line,
          column_start: match.range.start.column,
          column_end: match.range.end.column,
          matched_text: match.matched || '',
          replacement: match.replacement || null,
          environment: match.environment || {}
        });
      }
    } catch (e) {
      // Skip invalid JSON lines
      continue;
    }
  }
  
  return results;
}

function getCombyVersion() {
  try {
    return require('child_process').execSync('comby -version', { encoding: 'utf8' }).trim();
  } catch (error) {
    return 'unknown';
  }
}

function countCorpusFiles() {
  try {
    const files = fs.readdirSync(corpusPath, { recursive: true });
    return files.length;
  } catch (error) {
    return 0;
  }
}

app.listen(port, () => {
  console.log(`Comby server listening on port ${port}`);
  console.log(`Corpus path: ${corpusPath}`);
  console.log(`Comby version: ${getCombyVersion()}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down gracefully');
  process.exit(0);
});