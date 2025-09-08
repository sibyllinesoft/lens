import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: [
      'src/**/*.test.ts',
      'tests/**/*.test.ts',
      'lens-src/**/*.test.ts'
    ],
    exclude: [
      'node_modules',
      'dist',
      'indexed-content/**/*',
      'sample-*/**/*',
      '**/*.d.ts'
    ],
    coverage: {
      provider: 'v8',
      include: [
        'src/core/ast-cache.ts',
        'src/indexer/lexical.ts',
        'src/storage/segments.ts',
        'src/types/config.ts',
        'src/indexer/n-gram-index.ts'
      ],
      reporter: ['text', 'html', 'json'],
      thresholds: {
        lines: 85,
        functions: 85,
        branches: 80,
        statements: 85,
      },
      exclude: [
        'node_modules',
        'dist',
        'indexed-content/**/*',
        'sample-*/**/*',
        '**/*.test.ts',
        '**/*.d.ts',
        'src/telemetry/tracer.ts', // External instrumentation
        'src/benchmark/**/*', // Exclude benchmarks from coverage
        'src/scripts/**/*', // Exclude scripts from coverage  
        'src/monitoring/**/*', // Exclude monitoring scripts
        'src/span_resolver/**/*', // Exclude external utilities
        '*.js', // Exclude root-level JS files
        '*.mjs', // Exclude root-level MJS files
      ],
    },
    globals: true,
    environment: 'node',
    testTimeout: 30000,
    hookTimeout: 10000,
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: true
      }
    }
  },
  esbuild: {
    target: 'es2022'
  }
});