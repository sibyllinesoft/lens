//! Performance benchmarks for the search engine
//!
//! Real benchmarks using Criterion to measure actual performance,
//! not simulation results.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lens_search_engine::{QueryBuilder, SearchEngine};
use std::{sync::Arc, time::Duration};
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn benchmark_search_engine(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Set up test data
    let temp_dir = TempDir::new().unwrap();
    let test_files_dir = temp_dir.path().join("test_files");
    std::fs::create_dir_all(&test_files_dir).unwrap();

    // Create test files with realistic content
    for i in 0..100 {
        let content = format!(
            r#"
// Test file {}
use std::collections::HashMap;

pub struct TestStruct_{} {{
    id: u32,
    name: String,
    data: HashMap<String, String>,
}}

impl TestStruct_{} {{
    pub fn new(id: u32, name: String) -> Self {{
        Self {{
            id,
            name,
            data: HashMap::new(),
        }}
    }}
    
    pub fn add_data(&mut self, key: String, value: String) {{
        self.data.insert(key, value);
    }}
    
    pub fn get_data(&self, key: &str) -> Option<&String> {{
        self.data.get(key)
    }}
    
    pub fn process_data(&self) -> Vec<String> {{
        self.data.values().cloned().collect()
    }}
}}

fn test_function_{}() {{
    let mut instance = TestStruct_{}::new({}, "test_instance".to_string());
    instance.add_data("key1".to_string(), "value1".to_string());
    instance.add_data("key2".to_string(), "value2".to_string());
    
    let processed = instance.process_data();
    println!("Processed {} items", processed.len());
}}

#[cfg(test)]
mod tests {{
    use super::*;
    
    #[test]
    fn test_struct_creation() {{
        let instance = TestStruct_{}::new(1, "test".to_string());
        assert_eq!(instance.id, 1);
        assert_eq!(instance.name, "test");
    }}
}}
"#,
            i, i, i, i, i, i, i, i
        );

        std::fs::write(test_files_dir.join(format!("test_{}.rs", i)), content).unwrap();
    }

    // Create search engine and index the test files
    let search_engine = rt.block_on(async {
        let engine = SearchEngine::new(temp_dir.path().join("index"))
            .await
            .unwrap();
        engine.index_directory(&test_files_dir).await.unwrap();
        engine
    });

    // Benchmark different query types
    c.bench_function("text_search", |b| {
        b.to_async(&rt).iter(|| async {
            let query = QueryBuilder::new("TestStruct").build();
            let results = search_engine.search(&query).await.unwrap();
            black_box(results)
        })
    });

    c.bench_function("symbol_search", |b| {
        b.to_async(&rt).iter(|| async {
            let query = QueryBuilder::new("new").symbol().build();
            let results = search_engine.search(&query).await.unwrap();
            black_box(results)
        })
    });

    c.bench_function("fuzzy_search", |b| {
        b.to_async(&rt).iter(|| async {
            let query = QueryBuilder::new("TestStrct").fuzzy().build();
            let results = search_engine.search(&query).await.unwrap();
            black_box(results)
        })
    });

    c.bench_function("large_result_search", |b| {
        b.to_async(&rt).iter(|| async {
            let query = QueryBuilder::new("String").limit(1000).build();
            let results = search_engine.search(&query).await.unwrap();
            black_box(results)
        })
    });
}

fn benchmark_indexing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("index_small_files", |b| {
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;

            for _ in 0..iters {
                let temp_dir = TempDir::new().unwrap();
                let test_files_dir = temp_dir.path().join("test_files");
                std::fs::create_dir_all(&test_files_dir).unwrap();

                // Create 10 small test files
                for i in 0..10 {
                    let content = format!(
                        "fn test_function_{}() {{\n    println!(\"Hello, world!\");\n}}",
                        i
                    );
                    std::fs::write(test_files_dir.join(format!("test_{}.rs", i)), content).unwrap();
                }

                let start = std::time::Instant::now();
                rt.block_on(async {
                    let engine = SearchEngine::new(temp_dir.path().join("index"))
                        .await
                        .unwrap();
                    let stats = engine.index_directory(&test_files_dir).await.unwrap();
                    black_box(stats);
                });
                total_duration += start.elapsed();
            }

            total_duration
        })
    });
}

fn benchmark_concurrent_searches(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("index");
    let test_files_dir = temp_dir.path().join("test_files");
    std::fs::create_dir_all(&test_files_dir).unwrap();

    for i in 0..50 {
        let content = format!(
            "pub fn function_{}() {{
    // Function implementation {}
}}",
            i, i
        );
        std::fs::write(test_files_dir.join(format!("test_{}.rs", i)), content).unwrap();
    }

    let search_engine = rt.block_on(async {
        let engine = SearchEngine::new(&index_path).await.unwrap();
        engine.index_directory(&test_files_dir).await.unwrap();
        engine
    });
    let search_engine = Arc::new(search_engine);
    let queries = Arc::new(vec![
        QueryBuilder::new("function").build(),
        QueryBuilder::new("pub").build(),
        QueryBuilder::new("fn").symbol().build(),
    ]);

    c.bench_function("concurrent_searches", |b| {
        let engine = Arc::clone(&search_engine);
        let queries = Arc::clone(&queries);
        b.to_async(&rt).iter(move || {
            let engine = Arc::clone(&engine);
            let queries = Arc::clone(&queries);
            async move {
                let futures = queries.iter().cloned().map(|query| {
                    let engine = Arc::clone(&engine);
                    async move { engine.search(&query).await.unwrap() }
                });
                let results = futures::future::join_all(futures).await;
                black_box(results)
            }
        });
    });
}

criterion_group!(
    benches,
    benchmark_search_engine,
    benchmark_indexing,
    benchmark_concurrent_searches
);
criterion_main!(benches);
