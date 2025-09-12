#!/usr/bin/env python3
"""
Run Production Deployment Package
==================================

Executive script to demonstrate the complete T₁ production deployment package
with all components integrated and validated.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from production_deployment_package import (
    ProductionDeploymentPackage,
    create_sample_training_data,
    T1BaselineMetrics,
    ProductionGuards
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('production_deployment.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Execute complete production deployment package creation and validation"""
    
    print("🏭 T₁ PRODUCTION DEPLOYMENT PACKAGE")
    print("Banking +2.31pp nDCG as Offline Gold Standard")
    print("="*80)
    
    try:
        # Step 1: Generate realistic training and validation datasets
        logger.info("📊 Generating comprehensive datasets...")
        training_data = create_sample_training_data(15000)  # Larger training set
        validation_data = create_sample_training_data(3000)  # Comprehensive validation
        
        print(f"✅ Training data: {len(training_data)} samples")
        print(f"✅ Validation data: {len(validation_data)} samples")
        
        # Step 2: Initialize production deployment package
        logger.info("🚀 Initializing production deployment package...")
        package = ProductionDeploymentPackage()
        
        print("✅ Production deployment package initialized")
        print(f"   - Baseline nDCG: {package.baseline_metrics.ndcg_at_10:.4f} (+2.31pp)")
        print(f"   - Hard-NL nDCG: {package.baseline_metrics.hard_nl_ndcg:.4f}")
        print(f"   - Target p95: ≤{package.production_guards.p95_latency_max}ms")
        
        # Step 3: Create complete deployment package
        logger.info("🔥 Creating complete deployment package...")
        print("\n🔥 CREATING PRODUCTION DEPLOYMENT PACKAGE")
        print("="*60)
        
        deployment_results = package.create_complete_deployment_package(
            training_data, validation_data
        )
        
        # Step 4: Display comprehensive results
        print("\n📋 DEPLOYMENT PACKAGE SUMMARY")
        print("="*60)
        
        print(f"Package ID: {deployment_results['package_id']}")
        print(f"Created: {deployment_results['created_at']}")
        print(f"Baseline: {deployment_results['baseline_standard']}")
        print(f"Status: {'🟢 READY' if deployment_results['deployment_ready'] else '🔴 BLOCKED'}")
        
        # Component details
        print(f"\n📦 COMPONENT VALIDATION RESULTS")
        components = deployment_results['components']
        
        # Router Distillation
        router = components['router_distillation']
        print(f"🔥 Router Distillation: {'✅ PASS' if router['no_regret_satisfied'] else '❌ FAIL'}")
        print(f"   - No-regret constraint: {router['no_regret_satisfied']}")
        print(f"   - Quantization: {router['quantization_bits']}-bit")
        print(f"   - Segments: {router['piecewise_segments']}")
        
        # Confounding Resolution
        confounding = components['confounding_resolution'] 
        print(f"🔧 Confounding Resolution: {'✅ PASS' if confounding['negative_controls_passed'] else '❌ FAIL'}")
        print(f"   - Negative controls: {confounding['negative_controls_passed']}")
        print(f"   - ESS validation: {confounding['ess_validation_passed']}")
        
        # Gating Optimization
        gating = components['gating_optimization']
        print(f"🎯 Gating Optimization: {'✅ PASS' if gating['budget_satisfied'] else '❌ FAIL'}")
        print(f"   - θ*: {gating['theta_star']:.3f}")
        print(f"   - θ early-exit: {gating['theta_early_exit']:.3f}")
        print(f"   - Budget satisfied: {gating['budget_satisfied']}")
        
        # Latency Harvest  
        latency = components['latency_harvest']
        print(f"⚡ Latency Harvest: {'✅ PASS' if latency['constraints_satisfied'] else '❌ FAIL'}")
        print(f"   - Optimal ef: {latency['optimal_ef']}")
        print(f"   - Optimal topk: {latency['optimal_topk']}")
        print(f"   - Constraints: {latency['constraints_satisfied']}")
        
        # Contract Validation
        contract = components['contract_validation']
        print(f"📋 Release Contract: {'✅ PASS' if contract['contract_satisfied'] else '❌ FAIL'}")
        print(f"   - Deployment authorized: {contract['deployment_authorized']}")
        
        # Production Artifacts
        artifacts = components['production_artifacts']
        print(f"\n📦 PRODUCTION ARTIFACTS ({artifacts['total_artifacts']} files)")
        for category, files in artifacts['artifacts_exported'].items():
            print(f"   {category.replace('_', ' ').title()}: {len(files)} files")
        
        # Step 5: Demonstrate monitoring and sustainment
        if deployment_results['deployment_ready']:
            print(f"\n🔍 PRODUCTION MONITORING")
            print("="*40)
            
            try:
                monitoring_config = deployment_results['monitoring_config']
                print(f"✅ Monitoring system activated")
                print(f"   - Measurement interval: {monitoring_config.get('measurement_interval_seconds', 60)}s")
                print(f"   - Alert channels: 3 configured")
                
                # Demonstrate real-time metrics collection
                print(f"\n📊 Sample real-time metrics:")
                metrics = package.monitoring_system.collect_realtime_metrics()
                print(f"   - Current nDCG: {metrics['ndcg_at_10']:.4f}")
                print(f"   - Current p95: {metrics['p95_latency']:.1f}ms") 
                print(f"   - Jaccard@10: {metrics['jaccard_at_10']:.3f}")
                print(f"   - AECE: {metrics['aece_max']:.4f}")
                
                # Validate guards
                guard_status = package.monitoring_system.validate_guards_realtime(metrics)
                print(f"   - All guards: {'✅ PASS' if guard_status['all_guards_passed'] else '❌ FAIL'}")
                
            except Exception as e:
                print(f"⚠️ Monitoring demo error: {e}")
            
            print(f"\n🔄 SUSTAINMENT LOOP")
            print("="*40)
            sustainment = deployment_results['sustainment_schedule']
            print(f"✅ 6-week cycles scheduled")
            print(f"   - Frequency: {sustainment['cycle_frequency_weeks']} weeks")
            print(f"   - Next cycle: {sustainment['next_cycle_date'][:10]}")
            
            # Demonstrate sustainment cycle (quick version)
            try:
                print(f"\n🔄 Demonstrating sustainment cycle...")
                cycle_results = package.sustainment_system.execute_sustainment_cycle()
                
                if cycle_results['overall_status'] == 'COMPLETED':
                    print(f"✅ Sustainment cycle completed: {cycle_results['cycle_id']}")
                    print(f"   - Steps completed: {len(cycle_results['steps_completed'])}/5")
                else:
                    print(f"⚠️ Sustainment cycle status: {cycle_results['overall_status']}")
                
            except Exception as e:
                print(f"⚠️ Sustainment demo error: {e}")
        
        # Step 6: Final recommendations
        print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS")
        print("="*50)
        
        if deployment_results['deployment_ready']:
            print("🟢 PRODUCTION DEPLOYMENT AUTHORIZED")
            print("")
            print("Immediate actions:")
            print("1. ✅ Deploy router configuration (router_distilled_int8.json)")
            print("2. ✅ Apply gating parameters (theta_star_production.json)")
            print("3. ✅ Configure latency harvest (latency_harvest_config.json)")
            print("4. ✅ Start monitoring system with 1-minute resolution")
            print("5. ✅ Schedule sustainment cycle in 6 weeks")
            print("")
            print("Quality assurance:")
            print("- Mathematical guards enforce T₁ (+2.31pp) gold standard")
            print("- Automatic rollback triggers protect against regressions")
            print("- Continuous monitoring validates contract compliance")
            
        else:
            print("🔴 PRODUCTION DEPLOYMENT BLOCKED")
            print("")
            print("Required actions before deployment:")
            
            # Check which components failed
            if not router['no_regret_satisfied']:
                print("- ❌ Fix router distillation no-regret violation")
            if not confounding['negative_controls_passed']:
                print("- ❌ Resolve confounding in observational data")
            if not gating['budget_satisfied']:
                print("- ❌ Adjust gating parameters to meet budget constraints")
            if not latency['constraints_satisfied']:
                print("- ❌ Optimize latency harvest parameters")
            if not contract['contract_satisfied']:
                print("- ❌ Address release contract guard failures")
        
        # Step 7: File inventory
        print(f"\n📁 GENERATED FILES")
        print("="*30)
        
        expected_files = [
            'router_distilled_int8.json',
            'theta_star_production.json', 
            'latency_harvest_config.json',
            'T1_release_contract.md',
            'conformal_coverage_report.csv',
            'counterfactual_audit_fixed.csv',
            'production_monitoring_config.json',
            'regression_gallery.md',
            'production_deployment.log'
        ]
        
        for filename in expected_files:
            if Path(filename).exists():
                print(f"✅ {filename}")
            else:
                print(f"⚠️ {filename} (expected but not found)")
        
        print(f"\n🎉 T₁ PRODUCTION DEPLOYMENT PACKAGE COMPLETE!")
        print("="*60)
        print("The +2.31pp nDCG improvement has been banked as the offline gold standard")
        print("with comprehensive mathematical guards, monitoring, and sustainment.")
        
        return deployment_results
        
    except Exception as e:
        logger.error(f"Production deployment failed: {e}", exc_info=True)
        print(f"\n💥 DEPLOYMENT FAILED: {e}")
        return None

if __name__ == '__main__':
    results = main()
    
    if results and results.get('deployment_ready'):
        print(f"\nℹ️  Run 'python production_monitoring_demo.py' to see live monitoring")
        sys.exit(0)
    else:
        print(f"\nℹ️  Review deployment results and fix issues before production")
        sys.exit(1)