# Code Style and Conventions

## TypeScript Configuration
- **Target**: ES2022 with NodeNext modules
- **Strict mode**: Enabled with additional strict checks
  - `noUncheckedIndexedAccess: true`
  - `exactOptionalPropertyTypes: true`
  - `noImplicitReturns: true`
  - `noImplicitOverride: true`
  - `noPropertyAccessFromIndexSignature: true`

## File Organization
```
src/
├── api/           # API endpoints and server logic
├── core/          # Core messaging and orchestration
├── indexer/       # Three-layer indexing (lexical, symbols, semantic)
├── storage/       # Memory-mapped segments and persistence
├── telemetry/     # OpenTelemetry tracing integration
└── types/         # Type definitions (api, core, config)
```

## Naming Conventions
- **Files**: kebab-case (e.g., `search-engine.ts`)
- **Classes**: PascalCase (e.g., `LensSearchEngine`)
- **Interfaces**: PascalCase with descriptive names (e.g., `SearchRequest`)
- **Functions/Variables**: camelCase (e.g., `searchQuery`)
- **Constants**: SCREAMING_SNAKE_CASE for module-level constants

## Type Definitions
- Use Zod schemas for runtime validation
- Interfaces for compile-time type checking
- Enums for discrete value sets (e.g., `SearchMode`, `MatchReason`)
- Generic types for reusable components

## Import/Export Style
- Use ES modules with explicit imports
- Group imports: external packages, then internal modules
- Prefer named exports over default exports
- Use barrel exports (index.ts) for clean module interfaces

## Architecture Patterns
- **Ports & Adapters**: Domain logic isolated from I/O
- **Functional Core, Imperative Shell**: Pure functions for business logic
- **Memory-mapped segments**: Append-only storage with compaction
- **Three-stage pipeline**: Lexical → Symbol/AST → Semantic (optional)