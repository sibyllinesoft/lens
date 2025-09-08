use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::Result;

/// Comprehensive metrics collection for Lens system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub search_metrics: SearchMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub quality_metrics: QualityMetrics,
    pub system_health: SystemHealth,
    pub timestamp: String,
}

/// Search-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_results_per_query: f64,
    pub query_distribution: HashMap<String, u64>,
    pub language_distribution: HashMap<String, u64>,
}

/// Performance metrics with SLA tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub throughput_qps: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization_percent: f64,
    pub cache_hit_rate: f64,
    pub lsp_latency_ms: f64,
}

/// Quality metrics for search relevance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub success_at_1: f64,
    pub success_at_10: f64,
    pub success_at_20: f64,
    pub ndcg_at_10: f64,
    pub ndcg_at_50: f64,
    pub mrr: f64,
    pub sla_recall_at_50: f64,
    pub witness_coverage: f64,
    pub span_accuracy: f64,
}

/// System health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub status: HealthStatus,
    pub lsp_servers_active: u32,
    pub index_size_mb: f64,
    pub error_rate: f64,
    pub uptime_seconds: u64,
    pub last_index_update: String,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Real-time metrics collector
pub struct MetricsCollector {
    metrics: Arc<RwLock<SystemMetrics>>,
    query_latencies: Arc<RwLock<Vec<f64>>>,
    query_counts: Arc<RwLock<HashMap<String, u64>>>,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        let initial_metrics = SystemMetrics {
            search_metrics: SearchMetrics {
                total_queries: 0,
                successful_queries: 0,
                failed_queries: 0,
                average_results_per_query: 0.0,
                query_distribution: HashMap::new(),
                language_distribution: HashMap::new(),
            },
            performance_metrics: PerformanceMetrics {
                latency_p50_ms: 0.0,
                latency_p95_ms: 0.0,
                latency_p99_ms: 0.0,
                throughput_qps: 0.0,
                memory_usage_mb: 0.0,
                cpu_utilization_percent: 0.0,
                cache_hit_rate: 0.0,
                lsp_latency_ms: 0.0,
            },
            quality_metrics: QualityMetrics {
                success_at_1: 0.0,
                success_at_10: 0.0,
                success_at_20: 0.0,
                ndcg_at_10: 0.0,
                ndcg_at_50: 0.0,
                mrr: 0.0,
                sla_recall_at_50: 0.0,
                witness_coverage: 0.0,
                span_accuracy: 0.0,
            },
            system_health: SystemHealth {
                status: HealthStatus::Healthy,
                lsp_servers_active: 0,
                index_size_mb: 0.0,
                error_rate: 0.0,
                uptime_seconds: 0,
                last_index_update: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
            },
            timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
        };

        Self {
            metrics: Arc::new(RwLock::new(initial_metrics)),
            query_latencies: Arc::new(RwLock::new(Vec::new())),
            query_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a search query
    pub fn record_query(&self, query_type: &str, language: Option<&str>, latency_ms: f64, success: bool, result_count: usize) {
        let mut metrics = self.metrics.write();
        let mut latencies = self.query_latencies.write();
        let mut counts = self.query_counts.write();

        // Update basic counters
        metrics.search_metrics.total_queries += 1;
        if success {
            metrics.search_metrics.successful_queries += 1;
        } else {
            metrics.search_metrics.failed_queries += 1;
        }

        // Update latency tracking
        latencies.push(latency_ms);

        // Update query distribution
        *metrics.search_metrics.query_distribution.entry(query_type.to_string()).or_insert(0) += 1;
        *counts.entry(query_type.to_string()).or_insert(0) += 1;

        // Update language distribution
        if let Some(lang) = language {
            *metrics.search_metrics.language_distribution.entry(lang.to_string()).or_insert(0) += 1;
        }

        // Update average results per query (rolling average)
        let total_results = metrics.search_metrics.average_results_per_query * (metrics.search_metrics.total_queries - 1) as f64;
        metrics.search_metrics.average_results_per_query = (total_results + result_count as f64) / metrics.search_metrics.total_queries as f64;

        // Update timestamp
        metrics.timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    }

    /// Calculate and update performance percentiles
    pub fn update_performance_metrics(&self) {
        let mut metrics = self.metrics.write();
        let latencies = self.query_latencies.read();

        if latencies.is_empty() {
            return;
        }

        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentiles
        let len = sorted_latencies.len();
        metrics.performance_metrics.latency_p50_ms = sorted_latencies[len * 50 / 100];
        metrics.performance_metrics.latency_p95_ms = sorted_latencies[len * 95 / 100];
        metrics.performance_metrics.latency_p99_ms = sorted_latencies[len * 99 / 100];

        // Calculate throughput (queries per second over last measurement period)
        if metrics.search_metrics.total_queries > 0 {
            let elapsed_seconds = 60.0; // Simplified - would track actual elapsed time
            metrics.performance_metrics.throughput_qps = metrics.search_metrics.total_queries as f64 / elapsed_seconds;
        }

        // Update system resource metrics (would integrate with actual system monitoring)
        metrics.performance_metrics.memory_usage_mb = Self::get_memory_usage();
        metrics.performance_metrics.cpu_utilization_percent = Self::get_cpu_usage();
    }

    /// Update quality metrics from benchmark results
    pub fn update_quality_metrics(&self, quality_update: QualityMetricsUpdate) {
        let mut metrics = self.metrics.write();
        
        metrics.quality_metrics.success_at_1 = quality_update.success_at_1;
        metrics.quality_metrics.success_at_10 = quality_update.success_at_10;
        metrics.quality_metrics.success_at_20 = quality_update.success_at_20;
        metrics.quality_metrics.ndcg_at_10 = quality_update.ndcg_at_10;
        metrics.quality_metrics.ndcg_at_50 = quality_update.ndcg_at_50;
        metrics.quality_metrics.mrr = quality_update.mrr;
        metrics.quality_metrics.sla_recall_at_50 = quality_update.sla_recall_at_50;
        metrics.quality_metrics.witness_coverage = quality_update.witness_coverage;
        metrics.quality_metrics.span_accuracy = quality_update.span_accuracy;
        
        metrics.timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    }

    /// Update system health status
    pub fn update_system_health(&self, health_update: SystemHealthUpdate) {
        let mut metrics = self.metrics.write();
        
        metrics.system_health.lsp_servers_active = health_update.lsp_servers_active;
        metrics.system_health.index_size_mb = health_update.index_size_mb;
        metrics.system_health.error_rate = health_update.error_rate;
        
        // Determine overall health status
        metrics.system_health.status = if health_update.error_rate > 0.1 {
            HealthStatus::Unhealthy
        } else if health_update.error_rate > 0.05 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        
        if let Some(last_update) = health_update.last_index_update {
            metrics.system_health.last_index_update = last_update;
        }
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> SystemMetrics {
        self.metrics.read().clone()
    }

    /// Reset metrics (for benchmark runs)
    pub fn reset(&self) {
        let mut metrics = self.metrics.write();
        let mut latencies = self.query_latencies.write();
        let mut counts = self.query_counts.write();

        metrics.search_metrics = SearchMetrics {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            average_results_per_query: 0.0,
            query_distribution: HashMap::new(),
            language_distribution: HashMap::new(),
        };

        latencies.clear();
        counts.clear();
        
        metrics.timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    }

    /// Generate comprehensive metrics report
    pub fn generate_report(&self) -> MetricsReport {
        let metrics = self.get_metrics();
        self.update_performance_metrics();
        
        MetricsReport {
            metrics,
            sla_compliance: self.check_sla_compliance(),
            performance_grade: self.calculate_performance_grade(),
            recommendations: self.generate_recommendations(),
        }
    }

    fn check_sla_compliance(&self) -> SlaCompliance {
        let metrics = self.metrics.read();
        
        SlaCompliance {
            latency_p95_compliant: metrics.performance_metrics.latency_p95_ms <= 2000.0,
            success_rate_compliant: metrics.search_metrics.successful_queries as f64 / metrics.search_metrics.total_queries as f64 >= 0.95,
            availability_compliant: matches!(metrics.system_health.status, HealthStatus::Healthy),
        }
    }

    fn calculate_performance_grade(&self) -> PerformanceGrade {
        let metrics = self.metrics.read();
        
        let latency_score = if metrics.performance_metrics.latency_p95_ms <= 150.0 {
            100.0
        } else if metrics.performance_metrics.latency_p95_ms <= 500.0 {
            80.0
        } else if metrics.performance_metrics.latency_p95_ms <= 1000.0 {
            60.0
        } else {
            40.0
        };

        let success_rate = if metrics.search_metrics.total_queries > 0 {
            (metrics.search_metrics.successful_queries as f64 / metrics.search_metrics.total_queries as f64) * 100.0
        } else {
            100.0
        };

        let overall_score = (latency_score + success_rate) / 2.0;
        
        if overall_score >= 90.0 {
            PerformanceGrade::Excellent
        } else if overall_score >= 80.0 {
            PerformanceGrade::Good
        } else if overall_score >= 70.0 {
            PerformanceGrade::Fair
        } else {
            PerformanceGrade::Poor
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let metrics = self.metrics.read();
        let mut recommendations = Vec::new();

        if metrics.performance_metrics.latency_p95_ms > 500.0 {
            recommendations.push("Consider optimizing query processing for better latency".to_string());
        }

        if metrics.performance_metrics.cache_hit_rate < 0.8 {
            recommendations.push("Improve caching strategy to increase hit rate".to_string());
        }

        if metrics.search_metrics.failed_queries as f64 / metrics.search_metrics.total_queries as f64 > 0.05 {
            recommendations.push("Investigate and reduce query failure rate".to_string());
        }

        recommendations
    }

    // Placeholder system monitoring functions
    fn get_memory_usage() -> f64 {
        // Would integrate with actual system monitoring
        0.0
    }

    fn get_cpu_usage() -> f64 {
        // Would integrate with actual system monitoring
        0.0
    }
}

/// Update structure for quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetricsUpdate {
    pub success_at_1: f64,
    pub success_at_10: f64,
    pub success_at_20: f64,
    pub ndcg_at_10: f64,
    pub ndcg_at_50: f64,
    pub mrr: f64,
    pub sla_recall_at_50: f64,
    pub witness_coverage: f64,
    pub span_accuracy: f64,
}

/// Update structure for system health
#[derive(Debug, Clone)]
pub struct SystemHealthUpdate {
    pub lsp_servers_active: u32,
    pub index_size_mb: f64,
    pub error_rate: f64,
    pub last_index_update: Option<String>,
}

/// SLA compliance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaCompliance {
    pub latency_p95_compliant: bool,
    pub success_rate_compliant: bool,
    pub availability_compliant: bool,
}

/// Performance grade enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceGrade {
    Excellent,
    Good,
    Fair,
    Poor,
}

/// Comprehensive metrics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub metrics: SystemMetrics,
    pub sla_compliance: SlaCompliance,
    pub performance_grade: PerformanceGrade,
    pub recommendations: Vec<String>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// SLA metrics for benchmarking compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaMetrics {
    pub latency_p95_sla_ms: f64,
    pub success_rate_sla: f64,
    pub availability_sla: f64,
    pub current_latency_p95_ms: f64,
    pub current_success_rate: f64,
    pub current_availability: f64,
    pub sla_violations: u64,
}

impl SlaMetrics {
    pub fn new(latency_p95_sla_ms: f64, success_rate_sla: f64, availability_sla: f64) -> Self {
        Self {
            latency_p95_sla_ms,
            success_rate_sla,
            availability_sla,
            current_latency_p95_ms: 0.0,
            current_success_rate: 1.0,
            current_availability: 1.0,
            sla_violations: 0,
        }
    }

    pub fn is_compliant(&self) -> bool {
        self.current_latency_p95_ms <= self.latency_p95_sla_ms &&
        self.current_success_rate >= self.success_rate_sla &&
        self.current_availability >= self.availability_sla
    }
}

/// Performance gate for promotion decisions
#[derive(Debug, Clone)]
pub struct PerformanceGate {
    pub name: String,
    pub threshold: f64,
    pub current_value: f64,
    pub comparison: GateComparison,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub enum GateComparison {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
}

impl PerformanceGate {
    pub fn new(name: String, threshold: f64, comparison: GateComparison) -> Self {
        Self {
            name,
            threshold,
            current_value: 0.0,
            comparison,
            passed: false,
        }
    }

    pub fn evaluate(&mut self, value: f64) -> bool {
        self.current_value = value;
        self.passed = match self.comparison {
            GateComparison::LessThan => value < self.threshold,
            GateComparison::LessThanOrEqual => value <= self.threshold,
            GateComparison::GreaterThan => value > self.threshold,
            GateComparison::GreaterThanOrEqual => value >= self.threshold,
            GateComparison::Equal => (value - self.threshold).abs() < f64::EPSILON,
        };
        self.passed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_new() {
        let collector = MetricsCollector::new();
        let metrics = collector.get_metrics();
        
        assert_eq!(metrics.search_metrics.total_queries, 0);
        assert_eq!(metrics.search_metrics.successful_queries, 0);
        assert_eq!(metrics.search_metrics.failed_queries, 0);
        assert_eq!(metrics.performance_metrics.latency_p50_ms, 0.0);
        assert!(matches!(metrics.system_health.status, HealthStatus::Healthy));
    }

    #[test]
    fn test_metrics_collector_default() {
        let collector = MetricsCollector::default();
        let metrics = collector.get_metrics();
        assert_eq!(metrics.search_metrics.total_queries, 0);
    }

    #[test]
    fn test_record_successful_query() {
        let collector = MetricsCollector::new();
        
        collector.record_query("search", Some("rust"), 50.0, true, 10);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.search_metrics.total_queries, 1);
        assert_eq!(metrics.search_metrics.successful_queries, 1);
        assert_eq!(metrics.search_metrics.failed_queries, 0);
        assert_eq!(metrics.search_metrics.average_results_per_query, 10.0);
        
        // Check query distribution
        assert_eq!(metrics.search_metrics.query_distribution.get("search"), Some(&1));
        // Check language distribution
        assert_eq!(metrics.search_metrics.language_distribution.get("rust"), Some(&1));
    }

    #[test]
    fn test_record_failed_query() {
        let collector = MetricsCollector::new();
        
        collector.record_query("search", None, 100.0, false, 0);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.search_metrics.total_queries, 1);
        assert_eq!(metrics.search_metrics.successful_queries, 0);
        assert_eq!(metrics.search_metrics.failed_queries, 1);
        assert_eq!(metrics.search_metrics.average_results_per_query, 0.0);
    }

    #[test]
    fn test_multiple_queries_average() {
        let collector = MetricsCollector::new();
        
        collector.record_query("search", Some("rust"), 25.0, true, 5);
        collector.record_query("search", Some("python"), 75.0, true, 15);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.search_metrics.total_queries, 2);
        assert_eq!(metrics.search_metrics.successful_queries, 2);
        assert_eq!(metrics.search_metrics.average_results_per_query, 10.0); // (5+15)/2
        
        assert_eq!(metrics.search_metrics.query_distribution.get("search"), Some(&2));
        assert_eq!(metrics.search_metrics.language_distribution.get("rust"), Some(&1));
        assert_eq!(metrics.search_metrics.language_distribution.get("python"), Some(&1));
    }

    #[test]
    fn test_update_performance_metrics() {
        let collector = MetricsCollector::new();
        
        // Record some queries with latencies
        collector.record_query("search", None, 10.0, true, 5);
        collector.record_query("search", None, 20.0, true, 5);
        collector.record_query("search", None, 30.0, true, 5);
        collector.record_query("search", None, 40.0, true, 5);
        collector.record_query("search", None, 100.0, true, 5);
        
        collector.update_performance_metrics();
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.performance_metrics.latency_p50_ms, 30.0); // median
        assert_eq!(metrics.performance_metrics.latency_p95_ms, 100.0); // 95th percentile
        assert_eq!(metrics.performance_metrics.latency_p99_ms, 100.0); // 99th percentile
        assert!(metrics.performance_metrics.throughput_qps > 0.0);
    }

    #[test]
    fn test_update_performance_metrics_empty() {
        let collector = MetricsCollector::new();
        
        // No queries recorded
        collector.update_performance_metrics();
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.performance_metrics.latency_p50_ms, 0.0);
        assert_eq!(metrics.performance_metrics.latency_p95_ms, 0.0);
        assert_eq!(metrics.performance_metrics.latency_p99_ms, 0.0);
    }

    #[test]
    fn test_update_quality_metrics() {
        let collector = MetricsCollector::new();
        
        let quality_update = QualityMetricsUpdate {
            success_at_1: 0.8,
            success_at_10: 0.9,
            success_at_20: 0.95,
            ndcg_at_10: 0.85,
            ndcg_at_50: 0.88,
            mrr: 0.82,
            sla_recall_at_50: 0.87,
            witness_coverage: 0.91,
            span_accuracy: 0.93,
        };
        
        collector.update_quality_metrics(quality_update);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.quality_metrics.success_at_1, 0.8);
        assert_eq!(metrics.quality_metrics.success_at_10, 0.9);
        assert_eq!(metrics.quality_metrics.ndcg_at_10, 0.85);
        assert_eq!(metrics.quality_metrics.mrr, 0.82);
    }

    #[test]
    fn test_update_system_health_healthy() {
        let collector = MetricsCollector::new();
        
        let health_update = SystemHealthUpdate {
            lsp_servers_active: 4,
            index_size_mb: 1024.0,
            error_rate: 0.01, // Low error rate - should be healthy
            last_index_update: Some("2023-01-01T00:00:00Z".to_string()),
        };
        
        collector.update_system_health(health_update);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.system_health.lsp_servers_active, 4);
        assert_eq!(metrics.system_health.index_size_mb, 1024.0);
        assert_eq!(metrics.system_health.error_rate, 0.01);
        assert!(matches!(metrics.system_health.status, HealthStatus::Healthy));
        assert_eq!(metrics.system_health.last_index_update, "2023-01-01T00:00:00Z");
    }

    #[test]
    fn test_update_system_health_degraded() {
        let collector = MetricsCollector::new();
        
        let health_update = SystemHealthUpdate {
            lsp_servers_active: 2,
            index_size_mb: 512.0,
            error_rate: 0.08, // Medium error rate - should be degraded
            last_index_update: None,
        };
        
        collector.update_system_health(health_update);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.system_health.error_rate, 0.08);
        assert!(matches!(metrics.system_health.status, HealthStatus::Degraded));
    }

    #[test]
    fn test_update_system_health_unhealthy() {
        let collector = MetricsCollector::new();
        
        let health_update = SystemHealthUpdate {
            lsp_servers_active: 0,
            index_size_mb: 0.0,
            error_rate: 0.15, // High error rate - should be unhealthy
            last_index_update: None,
        };
        
        collector.update_system_health(health_update);
        
        let metrics = collector.get_metrics();
        assert_eq!(metrics.system_health.error_rate, 0.15);
        assert!(matches!(metrics.system_health.status, HealthStatus::Unhealthy));
    }

    #[test]
    fn test_reset_metrics() {
        let collector = MetricsCollector::new();
        
        // Record some data
        collector.record_query("search", Some("rust"), 50.0, true, 10);
        collector.record_query("lookup", Some("python"), 75.0, false, 5);
        
        let metrics_before = collector.get_metrics();
        assert_eq!(metrics_before.search_metrics.total_queries, 2);
        
        // Reset
        collector.reset();
        
        let metrics_after = collector.get_metrics();
        assert_eq!(metrics_after.search_metrics.total_queries, 0);
        assert_eq!(metrics_after.search_metrics.successful_queries, 0);
        assert_eq!(metrics_after.search_metrics.failed_queries, 0);
        assert!(metrics_after.search_metrics.query_distribution.is_empty());
        assert!(metrics_after.search_metrics.language_distribution.is_empty());
    }

    #[test]
    fn test_sla_metrics_creation() {
        let sla_metrics = SlaMetrics::new(150.0, 0.95, 0.99);
        
        assert_eq!(sla_metrics.latency_p95_sla_ms, 150.0);
        assert_eq!(sla_metrics.success_rate_sla, 0.95);
        assert_eq!(sla_metrics.availability_sla, 0.99);
        assert_eq!(sla_metrics.current_latency_p95_ms, 0.0);
        assert_eq!(sla_metrics.current_success_rate, 1.0);
        assert_eq!(sla_metrics.current_availability, 1.0);
        assert_eq!(sla_metrics.sla_violations, 0);
    }

    #[test]
    fn test_sla_metrics_compliance_success() {
        let mut sla_metrics = SlaMetrics::new(150.0, 0.95, 0.99);
        sla_metrics.current_latency_p95_ms = 100.0; // Under SLA
        sla_metrics.current_success_rate = 0.97; // Above SLA
        sla_metrics.current_availability = 0.995; // Above SLA
        
        assert!(sla_metrics.is_compliant());
    }

    #[test]
    fn test_sla_metrics_compliance_failure() {
        let mut sla_metrics = SlaMetrics::new(150.0, 0.95, 0.99);
        sla_metrics.current_latency_p95_ms = 200.0; // Over SLA
        sla_metrics.current_success_rate = 0.90; // Under SLA
        sla_metrics.current_availability = 0.98; // Under SLA
        
        assert!(!sla_metrics.is_compliant());
    }

    #[test]
    fn test_performance_gate_less_than() {
        let mut gate = PerformanceGate::new(
            "Latency P95".to_string(),
            150.0,
            GateComparison::LessThan
        );
        
        assert!(gate.evaluate(100.0)); // 100 < 150
        assert!(gate.passed);
        assert_eq!(gate.current_value, 100.0);
        
        assert!(!gate.evaluate(200.0)); // 200 >= 150
        assert!(!gate.passed);
        assert_eq!(gate.current_value, 200.0);
    }

    #[test]
    fn test_performance_gate_greater_than() {
        let mut gate = PerformanceGate::new(
            "Success Rate".to_string(),
            0.95,
            GateComparison::GreaterThan
        );
        
        assert!(gate.evaluate(0.98)); // 0.98 > 0.95
        assert!(gate.passed);
        
        assert!(!gate.evaluate(0.90)); // 0.90 <= 0.95
        assert!(!gate.passed);
    }

    #[test]
    fn test_performance_gate_equal() {
        let mut gate = PerformanceGate::new(
            "Exact Match".to_string(),
            1.0,
            GateComparison::Equal
        );
        
        assert!(gate.evaluate(1.0)); // Exactly 1.0
        assert!(gate.passed);
        
        assert!(!gate.evaluate(1.1)); // Not exactly 1.0
        assert!(!gate.passed);
    }

    #[test]
    fn test_performance_gate_less_than_or_equal() {
        let mut gate = PerformanceGate::new(
            "Memory Usage".to_string(),
            100.0,
            GateComparison::LessThanOrEqual
        );
        
        assert!(gate.evaluate(100.0)); // 100 <= 100
        assert!(gate.passed);
        
        assert!(gate.evaluate(50.0)); // 50 <= 100
        assert!(gate.passed);
        
        assert!(!gate.evaluate(150.0)); // 150 > 100
        assert!(!gate.passed);
    }

    #[test]
    fn test_performance_gate_greater_than_or_equal() {
        let mut gate = PerformanceGate::new(
            "Throughput".to_string(),
            10.0,
            GateComparison::GreaterThanOrEqual
        );
        
        assert!(gate.evaluate(10.0)); // 10 >= 10
        assert!(gate.passed);
        
        assert!(gate.evaluate(15.0)); // 15 >= 10
        assert!(gate.passed);
        
        assert!(!gate.evaluate(5.0)); // 5 < 10
        assert!(!gate.passed);
    }

    #[test]
    fn test_health_status_variants() {
        // Test serialization/deserialization would work
        let healthy = HealthStatus::Healthy;
        let degraded = HealthStatus::Degraded;
        let unhealthy = HealthStatus::Unhealthy;
        
        // Basic enum variant checks
        assert!(matches!(healthy, HealthStatus::Healthy));
        assert!(matches!(degraded, HealthStatus::Degraded));
        assert!(matches!(unhealthy, HealthStatus::Unhealthy));
    }

    #[test]
    fn test_performance_grade_variants() {
        let excellent = PerformanceGrade::Excellent;
        let good = PerformanceGrade::Good;
        let fair = PerformanceGrade::Fair;
        let poor = PerformanceGrade::Poor;
        
        assert!(matches!(excellent, PerformanceGrade::Excellent));
        assert!(matches!(good, PerformanceGrade::Good));
        assert!(matches!(fair, PerformanceGrade::Fair));
        assert!(matches!(poor, PerformanceGrade::Poor));
    }

    #[test]
    fn test_system_metrics_timestamp_updates() {
        let collector = MetricsCollector::new();
        
        let initial_metrics = collector.get_metrics();
        let initial_timestamp = initial_metrics.timestamp.clone();
        
        // Wait a moment and record a query
        std::thread::sleep(std::time::Duration::from_millis(10));
        collector.record_query("search", None, 50.0, true, 1);
        
        let updated_metrics = collector.get_metrics();
        assert_ne!(initial_timestamp, updated_metrics.timestamp);
    }

    #[test]
    fn test_query_distribution_multiple_types() {
        let collector = MetricsCollector::new();
        
        collector.record_query("search", None, 50.0, true, 5);
        collector.record_query("lookup", None, 30.0, true, 3);
        collector.record_query("search", None, 70.0, true, 7);
        collector.record_query("browse", None, 40.0, true, 2);
        
        let metrics = collector.get_metrics();
        
        assert_eq!(metrics.search_metrics.query_distribution.get("search"), Some(&2));
        assert_eq!(metrics.search_metrics.query_distribution.get("lookup"), Some(&1));
        assert_eq!(metrics.search_metrics.query_distribution.get("browse"), Some(&1));
        assert_eq!(metrics.search_metrics.total_queries, 4);
        
        // Average results: (5+3+7+2)/4 = 4.25
        assert!((metrics.search_metrics.average_results_per_query - 4.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_language_distribution() {
        let collector = MetricsCollector::new();
        
        collector.record_query("search", Some("rust"), 50.0, true, 5);
        collector.record_query("search", Some("python"), 30.0, true, 3);
        collector.record_query("search", Some("rust"), 70.0, true, 7);
        collector.record_query("search", None, 40.0, true, 2); // No language
        
        let metrics = collector.get_metrics();
        
        assert_eq!(metrics.search_metrics.language_distribution.get("rust"), Some(&2));
        assert_eq!(metrics.search_metrics.language_distribution.get("python"), Some(&1));
        // No language queries don't add to language distribution
        assert_eq!(metrics.search_metrics.language_distribution.len(), 2);
    }
}