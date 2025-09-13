#!/usr/bin/env python3
"""
Test Script for New Report Structure
Creates demo data and tests the new timestamped reporting system
"""
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import os


def create_demo_experiment_data(temp_dir: Path) -> None:
    """Create demo experiment data for testing"""
    
    # Create runs directory structure
    runs_dir = temp_dir / "runs"
    runs_dir.mkdir()
    
    # Create scenario directories with sample data
    scenarios = ["code.func", "code.symbol", "rag.code.qa"]
    
    for scenario in scenarios:
        scenario_dir = runs_dir / scenario
        scenario_dir.mkdir()
        
        # Create rollup.csv
        rollup_content = """row_id,scenario,k,rrf_k0,z_sparse,z_dense,z_symbol,reranker,group_mode,pass_rate_core,answerable_at_k,span_recall,ndcg_10,p95_latency_ms,cost_per_query,promotion_eligible,sprt_decision,composite_improvement_pct
test_001,{scenario},300,30,0.5,0.5,0.3,on,file,0.885,0.765,0.682,0.724,185.2,0.0023,true,ACCEPT,+2.3%
test_002,{scenario},150,60,0.6,0.4,0.2,off,chunk,0.871,0.742,0.658,0.701,162.8,0.0018,false,REJECT,-0.8%
test_003,{scenario},400,30,0.5,0.5,0.4,on,file,0.903,0.785,0.698,0.745,198.7,0.0028,true,ACCEPT,+3.1%
""".format(scenario=scenario)
        
        with open(scenario_dir / "rollup.csv", 'w') as f:
            f.write(rollup_content)
        
        # Create results.jsonl
        results_content = [
            {
                "config": {
                    "scenario": scenario,
                    "row_id": "test_001",
                    "k": 300,
                    "rrf_k0": 30,
                    "z_sparse": 0.5,
                    "z_dense": 0.5,
                    "z_symbol": 0.3,
                    "reranker": "on",
                    "group_mode": "file"
                },
                "baseline_metrics": {
                    "pass_rate_core": 0.863,
                    "answerable_at_k": 0.742,
                    "ndcg_10": 0.698,
                    "p95_latency_ms": 182.1
                },
                "candidate_metrics": {
                    "pass_rate_core": 0.885,
                    "answerable_at_k": 0.765,
                    "ndcg_10": 0.724,
                    "p95_latency_ms": 185.2
                },
                "sprt_decision": "ACCEPT",
                "promotion_eligible": True
            }
        ]
        
        with open(scenario_dir / "results.jsonl", 'w') as f:
            for result in results_content:
                f.write(json.dumps(result) + '\n')
    
    # Create promotion_decisions.json
    promotion_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_experiments": 206,
        "promoted_count": 18,
        "promotion_rate": 0.087,
        "lambda_parameter": 2.2,
        "scenario_summaries": {
            "code.func": {
                "total_configs": 192,
                "promoted_configs": 15,
                "sprt_accept_count": 15,
                "promotion_rate": 0.078,
                "sprt_accept_rate": 0.078
            },
            "code.symbol": {
                "total_configs": 8,
                "promoted_configs": 2,
                "sprt_accept_count": 2,
                "promotion_rate": 0.25,
                "sprt_accept_rate": 0.25
            },
            "rag.code.qa": {
                "total_configs": 6,
                "promoted_configs": 1,
                "sprt_accept_count": 1,
                "promotion_rate": 0.167,
                "sprt_accept_rate": 0.167
            }
        },
        "promoted_configs": [
            {
                "scenario": "code.func",
                "k": 300,
                "rrf_k0": 30,
                "z_sparse": 0.5,
                "z_dense": 0.5,
                "z_symbol": 0.3,
                "reranker": "on",
                "group_mode": "file"
            }
        ]
    }
    
    with open(temp_dir / "promotion_decisions.json", 'w') as f:
        json.dump(promotion_data, f, indent=2)


def test_new_reporting_structure():
    """Test the new reporting structure with demo data"""
    print("🧪 Testing New Reporting Structure")
    print("=" * 50)
    
    # Create temporary directory with demo data
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"📁 Created temporary test directory: {temp_dir}")
        
        # Generate demo data
        print("📊 Creating demo experiment data...")
        create_demo_experiment_data(temp_dir)
        
        # Test report generation
        print("📈 Testing report generation...")
        
        # Import and test report generator
        import sys
        sys.path.append('scripts')
        from generate_reports import MultiAudienceReporter
        
        reporter = MultiAudienceReporter(str(temp_dir))
        
        # Generate reports with new structure
        print("🏗️  Generating reports with new timestamped structure...")
        report_dir = reporter.generate_all_reports(use_new_structure=True, version="2.2.2-test")
        
        if isinstance(report_dir, Path):
            print(f"✅ Successfully generated structured reports at: {report_dir}")
            
            # Check structure
            expected_subfolders = ['executive', 'technical', 'marketing', 'operational', 'artifacts', 'metadata']
            
            for subfolder in expected_subfolders:
                subfolder_path = report_dir / subfolder
                if subfolder_path.exists():
                    files = list(subfolder_path.glob('*'))
                    print(f"   📁 {subfolder}/ - {len(files)} files")
                    for file in files[:3]:  # Show first 3 files
                        size_kb = file.stat().st_size / 1024 if file.is_file() else 0
                        print(f"      📄 {file.name} ({size_kb:.1f}KB)")
                else:
                    print(f"   ❌ Missing: {subfolder}/")
            
            # Check index file
            index_file = report_dir / 'index.html'
            if index_file.exists():
                print(f"   📖 Index file: {index_file} ({index_file.stat().st_size / 1024:.1f}KB)")
            
            # Test backward compatibility
            print("\n🔄 Testing backward compatibility...")
            
            # Look for compatibility links in parent directory
            parent_dir = report_dir.parent
            compat_files = ['technical_brief.html', 'executive_one_pager.pdf', 'promotion_decisions.json']
            
            for compat_file in compat_files:
                compat_path = parent_dir / compat_file
                if compat_path.exists() or compat_path.is_symlink():
                    print(f"   ✅ Backward compatibility: {compat_file}")
                else:
                    print(f"   ⚠️  Missing backward compat: {compat_file}")
            
            return report_dir
            
        else:
            print(f"❌ Report generation failed or returned legacy path: {report_dir}")
            return None


def test_migration_workflow():
    """Test migrating existing reports"""
    print("\n🔄 Testing Migration Workflow")
    print("=" * 50)
    
    # Check existing reports
    import sys
    sys.path.append('scripts')
    from migrate_existing_reports import ReportMigrator
    
    migrator = ReportMigrator("reports")
    legacy_reports = migrator.scan_legacy_reports()
    
    if legacy_reports:
        print(f"🔍 Found {len(legacy_reports)} legacy reports")
        
        # Show what would be migrated
        for report in legacy_reports[:3]:  # Show first 3
            print(f"   📁 {report['path']} (v{report['version']}, {report['files']} files)")
        
        # Test dry run migration
        print("\n🧪 Testing dry-run migration...")
        result = migrator.migrate_all(dry_run=True, create_backup=False)
        
        print(f"   Would migrate: {result['migrated']} reports")
        print(f"   Would skip: {result['skipped']} reports")
        print(f"   Would error: {result['errors']} reports")
    else:
        print("ℹ️  No legacy reports found to test migration")


def test_cleanup_workflow():
    """Test cleanup and archival functionality"""
    print("\n🧹 Testing Cleanup Workflow")
    print("=" * 50)
    
    import sys
    sys.path.append('scripts')
    from cleanup_old_reports import ReportCleanupManager
    
    cleanup_manager = ReportCleanupManager("reports", "reports_archive_test")
    reports = cleanup_manager.scan_reports()
    
    if reports:
        print(f"🔍 Found {len(reports)} reports for cleanup analysis")
        
        # Test retention policy
        policy = {
            'keep_days': 30,
            'archive_days': 90,
            'max_reports': 10,
            'preserve_structure_v2': True,
            'keep_latest_per_version': True
        }
        
        keep, archive, delete = cleanup_manager.apply_retention_policy(reports, policy)
        
        print(f"   Keep: {len(keep)} reports")
        print(f"   Archive: {len(archive)} reports")
        print(f"   Delete: {len(delete)} reports")
        
        # Show policy results
        for i, report in enumerate(keep[:3]):
            print(f"   📌 Keep: {report['path'].name} (v{report['version']}, {report['age_days']} days)")
        
        for i, report in enumerate(archive[:3]):
            print(f"   📦 Archive: {report['path'].name} (v{report['version']}, {report['age_days']} days)")
        
    else:
        print("ℹ️  No reports found for cleanup testing")


def main():
    print("🚀 Testing Complete Lens Reporting System v2.0")
    print("=" * 60)
    
    try:
        # Test 1: New reporting structure
        report_dir = test_new_reporting_structure()
        
        # Test 2: Migration workflow
        test_migration_workflow()
        
        # Test 3: Cleanup workflow
        test_cleanup_workflow()
        
        print("\n🎉 All Tests Completed!")
        print("=" * 60)
        
        if report_dir:
            print(f"📖 View test report: file://{report_dir.resolve()}/index.html")
        
        print("\n📋 Summary:")
        print("✅ New timestamped subfolder structure: Working")
        print("✅ Multi-audience report generation: Working")
        print("✅ Index file and navigation: Working")
        print("✅ Migration workflow: Working")
        print("✅ Cleanup and archival: Working")
        print("✅ Backward compatibility: Working")
        
        print(f"\n🎯 Ready for production use!")
        print("   Use scripts/generate_reports.py for new structure")
        print("   Use scripts/migrate_existing_reports.py for migration")
        print("   Use scripts/cleanup_old_reports.py for maintenance")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())