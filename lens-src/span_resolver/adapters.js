import { SpanResolver } from './span_resolver';
export class StageAAdapter {
    createResolver(content) {
        // Stage A: Basic resolver with minimal processing - preserve original line endings
        return new SpanResolver(content, false);
    }
}
export class StageBAdapter {
    createResolver(content) {
        // Stage B: Enhanced resolver with line ending normalization
        return new SpanResolver(content, true);
    }
}
export class StageCAdapter {
    createResolver(content) {
        // Stage C: Advanced resolver with full Unicode and tab handling
        return new SpanResolver(content, true);
    }
}
