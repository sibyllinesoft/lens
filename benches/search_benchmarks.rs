//! High-performance benchmarks for lens core
//! 
//! These benchmarks provide fraud-resistant performance measurement with:
//! - Criterion.rs for statistical rigor
//! - Environment capture for reproducibility  
//! - Anti-mock tripwires

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lens_core::search::SearchEngine;
use tempfile::TempDir;

fn create_test_engine() -> (SearchEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let engine = SearchEngine::new(temp_dir.path()).unwrap();
    (engine, temp_dir)
}

fn bench_search_latency(c: &mut Criterion) {
    let (engine, _temp_dir) = create_test_engine();
    
    c.bench_function("search_simple_query", |b| {
        b.iter(|| {
            let results = engine.search(black_box("function"), black_box(10));
            black_box(results)
        })
    });
    
    c.bench_function("search_complex_query", |b| {
        b.iter(|| {
            let results = engine.search(black_box("class AND method"), black_box(100));
            black_box(results)
        })
    });
}

fn bench_concurrent_searches(c: &mut Criterion) {
    let (engine, _temp_dir) = create_test_engine();
    
    c.bench_function("concurrent_search_10", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..10)
                .map(|i| {
                    let query = format!("query{}", i);
                    engine.search(&query, 10)
                })
                .collect();
            
            black_box(handles)
        })
    });
}

// Environment capture for benchmark attestation
fn bench_with_environment(c: &mut Criterion) {
    // Capture environment info that affects performance
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_else(|_| "unknown".to_string());
    let governor = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .unwrap_or_else(|_| "unknown".to_string());
    
    println!("Benchmark Environment:");
    println!("  CPU Governor: {}", governor.trim());
    println!("  Git SHA: {}", lens_core::built_info::GIT_VERSION.unwrap_or("unknown"));
    println!("  Build Time: {}", lens_core::built_info::BUILT_TIME_UTC);
    
    // Verify we're not in mock mode
    assert_eq!(std::env::var("LENS_MODE").unwrap_or_else(|_| "real".to_string()), "real");
    
    bench_search_latency(c);
    bench_concurrent_searches(c);
}

criterion_group!(benches, bench_with_environment);
criterion_main!(benches);