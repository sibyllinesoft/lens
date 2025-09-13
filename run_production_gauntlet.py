#!/usr/bin/env python3
"""
Production Gauntlet Executor
Runs the complete production readiness validation gauntlet with proper logging and reporting.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent / "scripts"))

from production_gauntlet import ProductionGauntlet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"gauntlet-execution-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Execute the production readiness gauntlet"""
    
    print("🚀 PRODUCTION READINESS GAUNTLET")
    print("=" * 80)
    print("Ruthless GA validation: statistical canarying, chaos drills, rollback rehearsals")
    print("Key/manifest rotation, and DR restore testing")
    print("=" * 80)
    print()
    
    try:
        # Initialize gauntlet with configuration
        config_path = "gauntlet-config.json"
        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
            
        gauntlet = ProductionGauntlet(config_path)
        logger.info("Gauntlet initialized with configuration")
        
        # Execute the complete gauntlet
        logger.info("Starting production readiness gauntlet execution")
        results = await gauntlet.run_full_gauntlet()
        
        # Generate timestamp for output files
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        
        # Save detailed results
        results_file = f"gauntlet-results-{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Generate and save human-readable report
        report = gauntlet.generate_gauntlet_report()
        report_file = f"gauntlet-report-{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Human-readable report saved to: {report_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("🏁 GAUNTLET EXECUTION COMPLETE")
        print("=" * 80)
        
        if results["overall_success"]:
            print("✅ STATUS: PASSED - READY FOR PRODUCTION")
            print(f"🔒 Green Fingerprint: {results.get('green_fingerprint', 'N/A')}")
            print()
            print("🎯 NEXT STEPS:")
            print("1. Flip CI to green with confidence")
            print("2. Execute staged rollout: shadow → canary (10%) → ramp (100%)")
            print("3. Monitor SLOs and error budgets continuously")
            print("4. Schedule day-7 mini-retro for post-deployment analysis")
        else:
            print("❌ STATUS: FAILED - NOT READY FOR PRODUCTION")
            print()
            print("🚨 BLOCKING ISSUES:")
            failed_steps = [s for s in results.get("steps", []) if s["status"] == "failed"]
            for step in failed_steps:
                print(f"   - {step['step'].replace('_', ' ').title()}: {step.get('failure_reason', 'Unknown')}")
            print()
            print("🔧 ACTION REQUIRED:")
            print("   Address all blocking issues before attempting production deployment")
        
        print(f"\n📊 Detailed Results: {results_file}")
        print(f"📄 Full Report: {report_file}")
        print("=" * 80)
        
        # Return appropriate exit code
        return 0 if results["overall_success"] else 1
        
    except Exception as e:
        logger.error(f"Gauntlet execution failed with exception: {e}")
        print(f"\n💥 GAUNTLET EXECUTION FAILED: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)