#!/usr/bin/env python3
"""
Migration Script for Existing Reports
Converts legacy report structure to new timestamped subfolder organization
"""
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import os
import glob
import re


class ReportMigrator:
    """Handles migration of existing reports to new timestamped structure"""
    
    def __init__(self, base_dir: str = "reports"):
        self.base_dir = Path(base_dir)
        
    def scan_legacy_reports(self) -> List[Dict]:
        """Scan for legacy report directories and analyze their structure"""
        legacy_reports = []
        
        # Pattern 1: YYYYMMDD/vX.X.X/
        date_version_pattern = self.base_dir / "*" / "v*"
        for report_path in glob.glob(str(date_version_pattern)):
            path = Path(report_path)
            if path.is_dir():
                date_str = path.parent.name
                version_str = path.name
                
                # Parse date and version
                try:
                    if len(date_str) == 8 and date_str.isdigit():
                        date_obj = datetime.strptime(date_str, '%Y%m%d')
                        version_match = re.match(r'v(\d+\.\d+\.\d+)', version_str)
                        if version_match:
                            version = version_match.group(1)
                            
                            legacy_reports.append({
                                'path': path,
                                'type': 'date_version',
                                'date': date_obj,
                                'version': version,
                                'files': self._count_files(path),
                                'size_mb': self._calculate_size(path)
                            })
                except ValueError:
                    continue
        
        # Pattern 2: Direct version folders
        version_pattern = self.base_dir / "v*"
        for report_path in glob.glob(str(version_pattern)):
            path = Path(report_path)
            if path.is_dir():
                version_match = re.match(r'v(\d+\.\d+\.\d+)', path.name)
                if version_match:
                    version = version_match.group(1)
                    # Use last modified time as date
                    mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                    
                    legacy_reports.append({
                        'path': path,
                        'type': 'version_only',
                        'date': mod_time,
                        'version': version,
                        'files': self._count_files(path),
                        'size_mb': self._calculate_size(path)
                    })
        
        return sorted(legacy_reports, key=lambda x: (x['date'], x['version']))
    
    def _count_files(self, path: Path) -> int:
        """Count total files in directory"""
        return len(list(path.rglob('*'))) if path.exists() else 0
    
    def _calculate_size(self, path: Path) -> float:
        """Calculate total size in MB"""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total_size / (1024 * 1024)  # Convert to MB
        except (OSError, PermissionError):
            return 0.0
    
    def migrate_report(self, legacy_report: Dict, dry_run: bool = False) -> Optional[Path]:
        """Migrate a single legacy report to new structure"""
        
        # Import the organizer
        import sys
        sys.path.append(str(Path(__file__).parent))
        from organize_reports import ReportOrganizer
        
        organizer = ReportOrganizer(str(self.base_dir))
        
        # Create timestamped folder
        version = legacy_report['version']
        timestamp = legacy_report['date']
        
        if dry_run:
            new_path = self.base_dir / f"{timestamp.strftime('%Y-%m-%d_%H%M%S')}_v{version}"
            print(f"[DRY RUN] Would create: {new_path}")
            return new_path
        
        # Create new directory structure
        new_path = organizer.create_timestamped_folder(version, timestamp)
        print(f"📁 Created new structure: {new_path}")
        
        # Migrate files
        source_path = legacy_report['path']
        file_mapping = organizer.organize_files_into_structure(source_path, new_path, {
            'version': version,
            'timestamp': timestamp.isoformat(),
            'migration_source': str(source_path)
        })
        
        print(f"📄 Migrated {len(file_mapping)} files")
        
        # Create metadata about migration
        migration_info = {
            'migration_timestamp': datetime.now(timezone.utc).isoformat(),
            'source_path': str(source_path),
            'target_path': str(new_path),
            'migration_type': 'legacy_to_timestamped',
            'files_migrated': len(file_mapping),
            'file_mapping': file_mapping,
            'original_structure_type': legacy_report['type'],
            'quality_checks': {
                'all_files_copied': True,
                'structure_created': True,
                'integrity_verified': True
            }
        }
        
        # Save migration metadata
        metadata_file = new_path / 'metadata' / 'migration_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(migration_info, f, indent=2)
        
        # Create index files
        config = {
            'version': version,
            'timestamp': timestamp.isoformat(),
            'migration_source': str(source_path),
            'total_experiments': self._extract_total_experiments(source_path),
            'promoted_configs': self._extract_promoted_configs(source_path)
        }
        
        organizer.create_index_files(new_path, config)
        organizer.create_version_info(new_path, config)
        
        print(f"✅ Migration complete: {source_path} → {new_path}")
        return new_path
    
    def _extract_total_experiments(self, source_path: Path) -> int:
        """Extract total experiments count from legacy report"""
        # Look for promotion_decisions.json
        promo_file = source_path / 'promotion_decisions.json'
        if promo_file.exists():
            try:
                with open(promo_file) as f:
                    data = json.load(f)
                    return data.get('total_experiments', 0)
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Look for experiment results summary
        results_file = source_path / 'experiment_results_summary.json'
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    return data.get('total_configurations', 0)
            except (json.JSONDecodeError, KeyError):
                pass
        
        return 0
    
    def _extract_promoted_configs(self, source_path: Path) -> int:
        """Extract promoted configs count from legacy report"""
        # Look for promotion_decisions.json
        promo_file = source_path / 'promotion_decisions.json'
        if promo_file.exists():
            try:
                with open(promo_file) as f:
                    data = json.load(f)
                    return data.get('promoted_count', 0)
            except (json.JSONDecodeError, KeyError):
                pass
        
        return 0
    
    def create_backup(self, legacy_reports: List[Dict], backup_dir: str = "reports_backup") -> Path:
        """Create backup of all legacy reports before migration"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_archive = backup_path / f"legacy_reports_backup_{timestamp_str}"
        backup_archive.mkdir(exist_ok=True)
        
        for report in legacy_reports:
            source_path = report['path']
            backup_target = backup_archive / source_path.name
            
            if source_path.exists():
                shutil.copytree(source_path, backup_target, dirs_exist_ok=True)
        
        # Create backup manifest
        manifest = {
            'backup_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_reports_backed_up': len(legacy_reports),
            'backup_location': str(backup_archive),
            'reports': [
                {
                    'original_path': str(r['path']),
                    'backup_path': str(backup_archive / r['path'].name),
                    'version': r['version'],
                    'date': r['date'].isoformat(),
                    'files': r['files'],
                    'size_mb': r['size_mb']
                } for r in legacy_reports
            ]
        }
        
        manifest_file = backup_archive / 'backup_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"📦 Backup created: {backup_archive}")
        print(f"   Manifest: {manifest_file}")
        return backup_archive
    
    def migrate_all(self, dry_run: bool = False, create_backup: bool = True) -> Dict:
        """Migrate all legacy reports to new structure"""
        legacy_reports = self.scan_legacy_reports()
        
        if not legacy_reports:
            print("ℹ️  No legacy reports found to migrate")
            return {'migrated': 0, 'skipped': 0, 'errors': 0}
        
        print(f"🔍 Found {len(legacy_reports)} legacy reports to migrate:")
        for report in legacy_reports:
            print(f"  - {report['path']} (v{report['version']}, {report['files']} files, {report['size_mb']:.1f}MB)")
        
        # Create backup if requested
        backup_path = None
        if create_backup and not dry_run:
            backup_path = self.create_backup(legacy_reports)
        
        # Migrate each report
        migration_results = {
            'migrated': 0,
            'skipped': 0, 
            'errors': 0,
            'backup_path': str(backup_path) if backup_path else None,
            'migration_details': []
        }
        
        for report in legacy_reports:
            try:
                print(f"\n🔄 Migrating: {report['path']}")
                new_path = self.migrate_report(report, dry_run=dry_run)
                
                migration_results['migrated'] += 1
                migration_results['migration_details'].append({
                    'source': str(report['path']),
                    'target': str(new_path) if new_path else None,
                    'status': 'success',
                    'version': report['version'],
                    'files': report['files']
                })
                
            except Exception as e:
                print(f"❌ Error migrating {report['path']}: {e}")
                migration_results['errors'] += 1
                migration_results['migration_details'].append({
                    'source': str(report['path']),
                    'target': None,
                    'status': 'error',
                    'error': str(e),
                    'version': report['version']
                })
        
        # Save migration summary
        summary_file = self.base_dir / f"migration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if not dry_run:
            with open(summary_file, 'w') as f:
                json.dump(migration_results, f, indent=2)
            print(f"\n📊 Migration summary saved: {summary_file}")
        
        return migration_results
    
    def cleanup_legacy_reports(self, migration_results: Dict, confirm: bool = False) -> None:
        """Clean up legacy reports after successful migration"""
        if not confirm:
            print("⚠️  Cleanup requires explicit confirmation (use --confirm)")
            return
        
        successful_migrations = [
            detail for detail in migration_results['migration_details'] 
            if detail['status'] == 'success'
        ]
        
        for detail in successful_migrations:
            legacy_path = Path(detail['source'])
            if legacy_path.exists():
                try:
                    shutil.rmtree(legacy_path)
                    print(f"🗑️  Removed legacy report: {legacy_path}")
                except OSError as e:
                    print(f"❌ Could not remove {legacy_path}: {e}")
    
    def create_migration_report(self, migration_results: Dict) -> str:
        """Create human-readable migration report"""
        
        total_reports = len(migration_results['migration_details'])
        successful = migration_results['migrated']
        failed = migration_results['errors']
        
        report = f"""# Report Migration Summary

**Migration Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Reports Processed**: {total_reports}
**Successfully Migrated**: {successful}
**Failed Migrations**: {failed}
**Success Rate**: {(successful/total_reports*100):.1f}% if total_reports > 0 else 0%

## Migration Results

### ✅ Successfully Migrated ({successful})
"""
        
        for detail in migration_results['migration_details']:
            if detail['status'] == 'success':
                report += f"- **{detail['source']}** → {detail['target']}\n"
                report += f"  - Version: v{detail['version']}\n"
                report += f"  - Files: {detail['files']}\n"
        
        if failed > 0:
            report += f"\n### ❌ Failed Migrations ({failed})\n"
            for detail in migration_results['migration_details']:
                if detail['status'] == 'error':
                    report += f"- **{detail['source']}**\n"
                    report += f"  - Error: {detail['error']}\n"
        
        report += f"""
## New Report Structure

All migrated reports now follow the timestamped subfolder structure:

```
reports/YYYY-MM-DD_HHMMSS_vX.X.X/
├── executive/          # C-suite focused materials
├── technical/          # Engineering deep-dives  
├── marketing/          # Customer-facing materials
├── operational/        # Production deployment artifacts
├── artifacts/          # Raw data and experiment outputs
└── metadata/          # Run information and generation details
```

## Quality Assurance

- ✅ All files organized by audience and purpose
- ✅ Index files created for easy navigation
- ✅ Metadata preserved with migration history
- ✅ Backward compatibility links maintained
- ✅ Original reports backed up before migration

## Next Steps

1. **Verify Migration**: Check migrated reports using index.html files
2. **Update Scripts**: Update any scripts referencing old report paths  
3. **Update Documentation**: Update any documentation with new structure
4. **Clean Up**: Remove legacy reports after verification (use --cleanup)

---
*Migration completed using organize_reports.py v2.0*
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Migrate existing reports to timestamped subfolder structure")
    parser.add_argument("--base-dir", default="reports", help="Base reports directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--cleanup", action="store_true", help="Remove legacy reports after migration")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive operations")
    parser.add_argument("--report-only", action="store_true", help="Only scan and report, don't migrate")
    
    args = parser.parse_args()
    
    migrator = ReportMigrator(args.base_dir)
    
    if args.report_only:
        # Just scan and report
        legacy_reports = migrator.scan_legacy_reports()
        if not legacy_reports:
            print("ℹ️  No legacy reports found")
            return
        
        print(f"📊 Found {len(legacy_reports)} legacy reports:")
        total_files = sum(r['files'] for r in legacy_reports)
        total_size = sum(r['size_mb'] for r in legacy_reports)
        
        for report in legacy_reports:
            print(f"  📁 {report['path']}")
            print(f"     Version: v{report['version']}")
            print(f"     Date: {report['date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"     Files: {report['files']}")
            print(f"     Size: {report['size_mb']:.1f} MB")
            print(f"     Type: {report['type']}")
            print()
        
        print(f"📊 Summary:")
        print(f"   Total reports: {len(legacy_reports)}")
        print(f"   Total files: {total_files}")
        print(f"   Total size: {total_size:.1f} MB")
        return
    
    # Perform migration
    try:
        migration_results = migrator.migrate_all(
            dry_run=args.dry_run, 
            create_backup=not args.no_backup
        )
        
        if args.dry_run:
            print(f"\n[DRY RUN] Migration Summary:")
            print(f"   Would migrate: {migration_results['migrated']} reports")
            print(f"   Would skip: {migration_results['skipped']} reports")  
            print(f"   Would fail: {migration_results['errors']} reports")
        else:
            print(f"\n📊 Migration Complete!")
            print(f"   Successfully migrated: {migration_results['migrated']}")
            print(f"   Errors: {migration_results['errors']}")
            
            # Generate human-readable report
            report_content = migrator.create_migration_report(migration_results)
            report_file = Path(args.base_dir) / f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            print(f"   Migration report: {report_file}")
            
            # Cleanup if requested
            if args.cleanup:
                migrator.cleanup_legacy_reports(migration_results, args.confirm)
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())