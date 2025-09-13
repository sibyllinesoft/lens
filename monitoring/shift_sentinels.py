#!/usr/bin/env python3
"""
Shift & Integrity Sentinels
Per-tenant control charts with Wilson bounds and drift detection
"""
import numpy as np
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

@dataclass
class TenantMetrics:
    tenant_id: str
    timestamp: str
    pass_rate_core: float
    answerable_at_k: float
    span_recall: float
    query_count: int
    p95_latency_ms: float

@dataclass
class ControlChart:
    metric_name: str
    tenant_id: str
    baseline_mean: float
    wilson_lower: float
    wilson_upper: float
    drift_threshold: float = 0.05  # 5pp
    alert_consecutive: int = 3

class ShiftSentinel:
    def __init__(self):
        self.tenant_charts: Dict[str, Dict[str, ControlChart]] = defaultdict(dict)
        self.history: Dict[str, List[TenantMetrics]] = defaultdict(list)
        self.alerts: List[Dict] = []
    
    def wilson_bounds(self, p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
        """Calculate Wilson score confidence intervals"""
        if n == 0:
            return 0.0, 1.0
            
        center = p + z**2 / (2*n)
        spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
        denominator = 1 + z**2/n
        
        lower = max(0, (center - spread) / denominator)
        upper = min(1, (center + spread) / denominator)
        
        return lower, upper
    
    def bootstrap_baseline(self, tenant_metrics: List[TenantMetrics], metric: str) -> ControlChart:
        """Bootstrap baseline control chart from historical data"""
        values = [getattr(m, metric) for m in tenant_metrics]
        query_counts = [m.query_count for m in tenant_metrics]
        
        if not values:
            # Default control chart for new tenants
            return ControlChart(
                metric_name=metric,
                tenant_id=tenant_metrics[0].tenant_id if tenant_metrics else "unknown",
                baseline_mean=0.85,  # Conservative default
                wilson_lower=0.80,
                wilson_upper=0.90,
                drift_threshold=0.05
            )
        
        baseline_mean = np.mean(values)
        avg_query_count = int(np.mean(query_counts))
        
        # Wilson bounds for proportion metrics
        if metric in ['pass_rate_core', 'answerable_at_k', 'span_recall']:
            wilson_lower, wilson_upper = self.wilson_bounds(baseline_mean, avg_query_count)
        else:
            # Gaussian bounds for latency metrics
            std = np.std(values)
            wilson_lower = baseline_mean - 1.96 * std
            wilson_upper = baseline_mean + 1.96 * std
        
        return ControlChart(
            metric_name=metric,
            tenant_id=tenant_metrics[0].tenant_id,
            baseline_mean=baseline_mean,
            wilson_lower=wilson_lower,
            wilson_upper=wilson_upper
        )
    
    def initialize_tenant_charts(self, tenant_id: str, historical_data: List[TenantMetrics]):
        """Initialize control charts for a tenant from historical data"""
        metrics = ['pass_rate_core', 'answerable_at_k', 'span_recall', 'p95_latency_ms']
        
        for metric in metrics:
            chart = self.bootstrap_baseline(historical_data, metric)
            self.tenant_charts[tenant_id][metric] = chart
    
    def check_drift(self, current: TenantMetrics) -> List[Dict]:
        """Check for metric drift against control charts"""
        tenant_id = current.tenant_id
        alerts = []
        
        if tenant_id not in self.tenant_charts:
            # Initialize with historical data if available
            historical = self.history.get(tenant_id, [])
            if len(historical) >= 7:  # Need minimum baseline
                self.initialize_tenant_charts(tenant_id, historical[-30:])  # Last 30 measurements
            else:
                # Not enough data for reliable control chart
                return alerts
        
        for metric_name, chart in self.tenant_charts[tenant_id].items():
            current_value = getattr(current, metric_name)
            
            # Check Wilson bound violations
            if current_value < chart.wilson_lower or current_value > chart.wilson_upper:
                alert = {
                    "timestamp": current.timestamp,
                    "tenant_id": tenant_id,
                    "metric": metric_name,
                    "current_value": current_value,
                    "baseline_mean": chart.baseline_mean,
                    "wilson_bounds": [chart.wilson_lower, chart.wilson_upper],
                    "drift_magnitude": abs(current_value - chart.baseline_mean),
                    "severity": "HIGH" if abs(current_value - chart.baseline_mean) > 0.10 else "MEDIUM",
                    "alert_type": "WILSON_BOUND_VIOLATION"
                }
                alerts.append(alert)
            
            # Check drift threshold (5pp sustained change)
            recent_history = self.history[tenant_id][-5:] if len(self.history[tenant_id]) >= 5 else []
            if len(recent_history) >= 3:
                recent_values = [getattr(m, metric_name) for m in recent_history]
                avg_recent = np.mean(recent_values)
                
                if abs(avg_recent - chart.baseline_mean) > chart.drift_threshold:
                    alert = {
                        "timestamp": current.timestamp,
                        "tenant_id": tenant_id,
                        "metric": metric_name,
                        "recent_average": avg_recent,
                        "baseline_mean": chart.baseline_mean,
                        "drift_magnitude": abs(avg_recent - chart.baseline_mean),
                        "severity": "HIGH",
                        "alert_type": "SUSTAINED_DRIFT",
                        "recommendation": self.get_drift_recommendation(metric_name, avg_recent - chart.baseline_mean)
                    }
                    alerts.append(alert)
        
        return alerts
    
    def get_drift_recommendation(self, metric: str, drift_direction: float) -> str:
        """Get remediation recommendation based on drift pattern"""
        recommendations = {
            'pass_rate_core': {
                'positive': "Quality improvement detected - investigate for replication",
                'negative': "Quality degradation - check indexing, model updates, or query patterns"
            },
            'answerable_at_k': {
                'positive': "Retrieval effectiveness improved - validate against false positives",
                'negative': "Retrieval effectiveness degraded - increase k or improve semantic matching"
            },
            'span_recall': {
                'positive': "Citation accuracy improved - validate context quality",
                'negative': "Citation accuracy degraded - check chunking or boundary detection"
            },
            'p95_latency_ms': {
                'positive': "Performance improved - monitor for stability",
                'negative': "Performance degraded - check resource utilization or query complexity"
            }
        }
        
        direction = 'positive' if drift_direction > 0 else 'negative'
        return recommendations.get(metric, {}).get(direction, "Monitor trend and investigate if sustained")
    
    def process_telemetry_batch(self, telemetry_batch: List[Dict]) -> Dict:
        """Process batch of telemetry data and detect shifts"""
        batch_alerts = []
        tenant_summaries = {}
        
        # Group by tenant
        tenant_data = defaultdict(list)
        for record in telemetry_batch:
            tenant_data[record['tenant_id']].append(record)
        
        # Process each tenant
        for tenant_id, records in tenant_data.items():
            # Aggregate tenant metrics for the batch
            tenant_metrics = self.aggregate_tenant_metrics(tenant_id, records)
            
            # Add to history
            self.history[tenant_id].append(tenant_metrics)
            
            # Keep rolling window (last 100 measurements)
            if len(self.history[tenant_id]) > 100:
                self.history[tenant_id] = self.history[tenant_id][-100:]
            
            # Check for drift
            alerts = self.check_drift(tenant_metrics)
            batch_alerts.extend(alerts)
            
            # Generate tenant summary
            tenant_summaries[tenant_id] = {
                "current_metrics": asdict(tenant_metrics),
                "control_charts": {
                    metric: asdict(chart) for metric, chart in self.tenant_charts.get(tenant_id, {}).items()
                },
                "alerts": len(alerts),
                "status": "ALERT" if alerts else "HEALTHY"
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "batch_size": len(telemetry_batch),
            "tenants_processed": len(tenant_summaries),
            "alerts_generated": len(batch_alerts),
            "tenant_summaries": tenant_summaries,
            "alerts": batch_alerts
        }
    
    def aggregate_tenant_metrics(self, tenant_id: str, records: List[Dict]) -> TenantMetrics:
        """Aggregate raw telemetry records into tenant metrics"""
        # Simulate aggregation from telemetry records
        query_count = len(records)
        
        # Mock aggregation - in reality, would compute from actual telemetry
        pass_rate = np.mean([r.get('success', True) for r in records])
        answerable = np.mean([r.get('answerable_score', 0.75) for r in records])
        span_recall = np.mean([r.get('span_recall', 0.68) for r in records])
        p95_latency = np.percentile([r.get('latency_ms', 180) for r in records], 95)
        
        return TenantMetrics(
            tenant_id=tenant_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            pass_rate_core=pass_rate,
            answerable_at_k=answerable,
            span_recall=span_recall,
            query_count=query_count,
            p95_latency_ms=p95_latency
        )

def generate_demo_telemetry():
    """Generate demo telemetry data for testing"""
    tenants = ['tenant_a', 'tenant_b', 'tenant_c', 'tenant_small']
    
    telemetry_batch = []
    for tenant in tenants:
        query_count = 100 if tenant != 'tenant_small' else 12  # Small tenant with higher variance
        
        for i in range(query_count):
            # Add some realistic variance and occasional drift
            base_pass_rate = 0.89 if tenant != 'tenant_b' else 0.82  # tenant_b has degraded performance
            noise = np.random.normal(0, 0.02)  # 2% noise
            
            record = {
                'tenant_id': tenant,
                'query_id': f"{tenant}_query_{i}",
                'timestamp': datetime.utcnow().isoformat() + "Z",
                'success': np.random.random() < (base_pass_rate + noise),
                'answerable_score': max(0, min(1, 0.75 + np.random.normal(0, 0.05))),
                'span_recall': max(0, min(1, 0.68 + np.random.normal(0, 0.04))),
                'latency_ms': max(50, 180 + np.random.normal(0, 20))
            }
            telemetry_batch.append(record)
    
    return telemetry_batch

def main():
    """Demo shift sentinel system"""
    print("🛡️ SHIFT & INTEGRITY SENTINELS")
    print("=" * 50)
    
    sentinel = ShiftSentinel()
    
    # Generate demo historical data for baseline establishment
    print("📊 Establishing baselines from historical telemetry...")
    for day in range(7):  # 7 days of history
        demo_batch = generate_demo_telemetry()
        result = sentinel.process_telemetry_batch(demo_batch)
        print(f"   Day {day+1}: {result['tenants_processed']} tenants, {result['alerts_generated']} alerts")
    
    print("\n🔍 Processing current telemetry batch...")
    current_batch = generate_demo_telemetry()
    
    # Add some drift to tenant_b
    for record in current_batch:
        if record['tenant_id'] == 'tenant_b':
            record['success'] = np.random.random() < 0.79  # 10pp degradation
    
    result = sentinel.process_telemetry_batch(current_batch)
    
    print(f"\n📈 BATCH PROCESSING RESULTS")
    print(f"   Telemetry Records: {result['batch_size']}")
    print(f"   Tenants Processed: {result['tenants_processed']}")
    print(f"   Alerts Generated: {result['alerts_generated']}")
    
    # Show tenant status
    print(f"\n🏢 TENANT STATUS SUMMARY")
    for tenant_id, summary in result['tenant_summaries'].items():
        status = summary['status']
        metrics = summary['current_metrics']
        print(f"   {tenant_id}: {status} | Pass-rate: {metrics['pass_rate_core']:.3f} | Queries: {metrics['query_count']}")
    
    # Show alerts
    if result['alerts']:
        print(f"\n🚨 DRIFT ALERTS")
        for alert in result['alerts']:
            print(f"   ⚠️  {alert['tenant_id']}.{alert['metric']}: {alert['alert_type']}")
            if 'current_value' in alert:
                print(f"      Current: {alert['current_value']:.3f}, Baseline: {alert['baseline_mean']:.3f}")
            elif 'recent_average' in alert:
                print(f"      Recent Avg: {alert['recent_average']:.3f}, Baseline: {alert['baseline_mean']:.3f}")
                print(f"      Drift: {alert['drift_magnitude']:.3f} ({alert['severity']})")
            if 'recommendation' in alert:
                print(f"      → {alert['recommendation']}")
    
    # Save results
    with open('shift-sentinel-report.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Shift sentinel report saved: shift-sentinel-report.json")

if __name__ == "__main__":
    main()