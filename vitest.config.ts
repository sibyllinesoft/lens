import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      reporter: ['text', 'html', 'json'],
      thresholds: {
        lines: 85,
        functions: 85,
        branches: 85,
        statements: 85,
      },
      exclude: [
        'node_modules',
        'dist',
        '**/*.test.ts',
        '**/*.d.ts',
        'src/telemetry/tracer.ts', // External instrumentation
      ],
    },
    globals: true,
    environment: 'node',
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});