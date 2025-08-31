# Task Completion Checklist

## When a coding task is completed, ensure:

### 1. Code Quality
- [ ] Code passes TypeScript compilation (`npm run build`)
- [ ] All linting rules pass (`npm run lint`)
- [ ] Code is properly formatted (`npm run fmt`)
- [ ] Architecture constraints validated (`npm run validate:config`)

### 2. Testing
- [ ] All existing tests pass (`npm test`)
- [ ] New functionality has corresponding tests
- [ ] Test coverage meets 85% threshold (`npm run test:coverage`)
- [ ] Tests follow the vitest framework conventions

### 3. Architecture Compliance
- [ ] Changes comply with `architecture.cue` constraints
- [ ] Performance targets maintained (per TODO.md specifications)
- [ ] API contracts respected (request/response schemas)
- [ ] Resource boundaries not violated

### 4. Documentation
- [ ] Code has appropriate TSDoc comments
- [ ] README updated if public API changed
- [ ] Architecture decision records (ADRs) created if needed

### 5. Observability
- [ ] OpenTelemetry tracing integrated for new components
- [ ] Appropriate logging with structured format (Pino)
- [ ] Metrics collection for performance monitoring

### 6. Integration
- [ ] NATS messaging integration works correctly
- [ ] Memory-mapped segments operate properly
- [ ] Service health checks pass (`/health` endpoint)

### 7. Benchmarking (Per TODO.md)
- [ ] Deterministic configuration with pinned seeds
- [ ] NATS telemetry publishing for benchmark runs
- [ ] Artifacts generated: metrics.parquet, errors.ndjson, traces.ndjson, report.pdf, config_fingerprint.json