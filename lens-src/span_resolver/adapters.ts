import { SpanResolver } from './span_resolver';

export interface SpanResolverAdapter {
  createResolver(content: string): SpanResolver;
}

export class StageAAdapter implements SpanResolverAdapter {
  createResolver(content: string): SpanResolver {
    // Stage A: Basic resolver with minimal processing - preserve original line endings
    return new SpanResolver(content, false);
  }
}

export class StageBAdapter implements SpanResolverAdapter {
  createResolver(content: string): SpanResolver {
    // Stage B: Enhanced resolver with line ending normalization
    return new SpanResolver(content, true);
  }
}

export class StageCAdapter implements SpanResolverAdapter {
  createResolver(content: string): SpanResolver {
    // Stage C: Advanced resolver with full Unicode and tab handling
    return new SpanResolver(content, true);
  }
}