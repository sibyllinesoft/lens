# Monitoring & Host Metrics

Lens exposes operational metrics for both search performance and host resource usage.

## Prometheus Metrics

Run `init_prometheus_metrics()` during application start to register counters and histograms. The following series are exported:

- `lens_queries_total`
- `lens_query_duration_seconds`
- `lens_search_latency_seconds`
- `lens_memory_usage_megabytes`
- `lens_cpu_utilization_percent`
- `lens_cache_hit_rate`

Scrape them through the `/metrics` endpoint (or whichever route you have configured inside your deployment).

## Host Health

System performance statistics are collected using [`sysinfo`](https://docs.rs/sysinfo/). Each metrics poll refreshes:

- **Memory usage (MiB)** via `System::used_memory()`
- **CPU utilization (%)** via `System::global_cpu_info().cpu_usage()`

No simulation or hard-coded values remain; the numbers reflect the actual process host. If you need higher-frequency measurements, schedule `MetricsCollector::update_performance_metrics()` accordingly.

## Recommended Alerts

| Metric | Suggested Threshold | Action |
|--------|--------------------|--------|
| CPU utilization | > 85% for 5m | Scale search workers or throttle ingest |
| Memory usage | > 80% of provisioned | Increase heap or reduce concurrent indexing |
| Search latency p95 | > 150 ms | Investigate slow queries / index fragmentation |

## Grafana & Dashboards

Sample Prometheus dashboards live under `monitoring/grafana/`. Import them into Grafana to visualize Lens throughput, latency percentiles, and host health in real time.
