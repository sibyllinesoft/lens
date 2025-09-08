use criterion::{criterion_group, criterion_main, Criterion};
use lens_core::search::SearchEngine;

fn search_benchmark(c: &mut Criterion) {
    let search_engine = SearchEngine::new_in_memory().expect("Failed to create search engine");
    
    c.bench_function("simple_search", |b| {
        b.iter(|| {
            search_engine.search("test query", 10).unwrap()
        })
    });
}

criterion_group!(benches, search_benchmark);
criterion_main!(benches);