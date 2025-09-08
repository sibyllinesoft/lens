use criterion::{criterion_group, criterion_main, Criterion};
use lens_core::indexer::Indexer;

fn indexing_benchmark(c: &mut Criterion) {
    c.bench_function("index_document", |b| {
        b.iter(|| {
            let mut indexer = Indexer::new_in_memory().expect("Failed to create indexer");
            indexer.index_document("test.py", "def hello(): return 'world'").unwrap()
        })
    });
}

criterion_group!(benches, indexing_benchmark);
criterion_main!(benches);