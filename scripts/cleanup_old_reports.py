#!/usr/bin/env python3
"""
Report Archive and Cleanup Management
Implements retention policies and cleanup for lens report system
"""
import argparse
import json
import shutil
import gzip
import tarfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import re


class ReportCleanupManager:
    """Manages archival, compression, and cleanup of reports with retention policies"""
    
    def __init__(self, base_dir: str = "reports", archive_dir: str = "reports_archive"):
        self.base_dir = Path(base_dir)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=True)
    
    def scan_reports(self) -> List[Dict]:
        """Scan all reports and classify by structure type and age"""
        reports = []
        
        # Scan for timestamped reports (new structure)
        timestamped_pattern = r'^(\d{4}-\d{2}-\d{2})_(\d{6})_v(\d+\.\d+\.\d+)$'
        
        for item in self.base_dir.iterdir():
            if not item.is_dir():
                continue
            
            match = re.match(timestamped_pattern, item.name)
            if match:
                date_str, time_str, version = match.groups()
                report_datetime = datetime.strptime(f"{date_str}_{time_str}", '%Y-%m-%d_%H%M%S')
                
                reports.append({
                    'path': item,
                    'type': 'timestamped',
                    'datetime': report_datetime,
                    'version': version,
                    'size_mb': self._calculate_size(item),
                    'files_count': self._count_files(item),
                    'age_days': (datetime.now() - report_datetime).days,
                    'structure_version': self._detect_structure_version(item)
                })
        
        # Scan for legacy reports (old structure)
        legacy_patterns = [
            (r'^(\d{8})$', 'date_only'),  # YYYYMMDD
            (r'^v(\d+\.\d+\.\d+)$', 'version_only'),  # vX.X.X
        ]
        
        for pattern, pattern_type in legacy_patterns:
            for item in self.base_dir.iterdir():
                if not item.is_dir():
                    continue
                
                match = re.match(pattern, item.name)
                if match and item not in [r['path'] for r in reports]:
                    mod_time = datetime.fromtimestamp(item.stat().st_mtime)
                    
                    reports.append({
                        'path': item,
                        'type': pattern_type,
                        'datetime': mod_time,
                        'version': match.group(1) if pattern_type == 'version_only' else 'unknown',
                        'size_mb': self._calculate_size(item),
                        'files_count': self._count_files(item),
                        'age_days': (datetime.now() - mod_time).days,
                        'structure_version': 'legacy'
                    })
        
        # Check for nested legacy reports (YYYYMMDD/vX.X.X/)
        for date_dir in self.base_dir.iterdir():
            if date_dir.is_dir() and re.match(r'^\d{8}$', date_dir.name):
                for version_dir in date_dir.iterdir():
                    if version_dir.is_dir() and re.match(r'^v\d+\.\d+\.\d+$', version_dir.name):
                        mod_time = datetime.fromtimestamp(version_dir.stat().st_mtime)
                        
                        reports.append({
                            'path': version_dir,
                            'type': 'nested_legacy',
                            'datetime': mod_time,
                            'version': version_dir.name[1:],  # Remove 'v' prefix
                            'size_mb': self._calculate_size(version_dir),
                            'files_count': self._count_files(version_dir),
                            'age_days': (datetime.now() - mod_time).days,
                            'structure_version': 'legacy'
                        })
        
        return sorted(reports, key=lambda x: x['datetime'], reverse=True)
    
    def _calculate_size(self, path: Path) -> float:
        """Calculate total size in MB"""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total_size / (1024 * 1024)
        except (OSError, PermissionError):
            return 0.0
    
    def _count_files(self, path: Path) -> int:
        """Count total files in directory"""
        try:
            return len([f for f in path.rglob('*') if f.is_file()])
        except (OSError, PermissionError):
            return 0
    
    def _detect_structure_version(self, path: Path) -> str:
        """Detect whether report uses new structured organization"""
        # Check for new structure subfolders
        expected_subfolders = ['executive', 'technical', 'marketing', 'operational', 'artifacts', 'metadata']
        existing_subfolders = [d.name for d in path.iterdir() if d.is_dir()]
        
        if any(subfolder in existing_subfolders for subfolder in expected_subfolders):
            if (path / 'index.html').exists():
                return '2.0'
            else:
                return '2.0-partial'
        
        return 'legacy'
    
    def apply_retention_policy(self, reports: List[Dict], policy: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Apply retention policy to classify reports for keep/archive/delete"""
        
        keep_reports = []
        archive_reports = []
        delete_reports = []
        
        keep_days = policy.get('keep_days', 30)
        archive_days = policy.get('archive_days', 90)
        max_reports = policy.get('max_reports', 50)
        keep_latest_per_version = policy.get('keep_latest_per_version', True)
        preserve_structure_v2 = policy.get('preserve_structure_v2', True)
        
        # First, ensure we keep the latest reports
        recent_reports = [r for r in reports if r['age_days'] <= keep_days]
        
        # Keep latest per version if requested
        if keep_latest_per_version:
            versions_seen = set()
            for report in reports:
                if report['version'] not in versions_seen:
                    recent_reports.append(report)
                    versions_seen.add(report['version'])
        
        # Preserve structure v2.0 reports if requested
        if preserve_structure_v2:
            v2_reports = [r for r in reports if r['structure_version'].startswith('2.0')]
            recent_reports.extend(v2_reports)
        
        # Remove duplicates while preserving order
        seen_paths = set()
        unique_recent = []
        for report in recent_reports:
            if report['path'] not in seen_paths:
                unique_recent.append(report)
                seen_paths.add(report['path'])
        
        keep_reports = unique_recent[:max_reports]
        
        # Classify remaining reports
        for report in reports:
            if report in keep_reports:
                continue
            
            if report['age_days'] <= archive_days:
                archive_reports.append(report)
            else:
                delete_reports.append(report)
        
        return keep_reports, archive_reports, delete_reports
    
    def archive_reports(self, reports: List[Dict], compression: str = 'gzip') -> List[str]:
        """Archive reports with compression"""
        archived_files = []
        
        for report in reports:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"{report['path'].name}_{timestamp}"
            
            if compression == 'gzip':
                archive_path = self.archive_dir / f"{archive_name}.tar.gz"
                
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(report['path'], arcname=report['path'].name)
                
            elif compression == 'tar':
                archive_path = self.archive_dir / f"{archive_name}.tar"
                
                with tarfile.open(archive_path, 'w') as tar:
                    tar.add(report['path'], arcname=report['path'].name)
            
            else:
                # Simple directory copy
                archive_path = self.archive_dir / archive_name
                shutil.copytree(report['path'], archive_path)
            
            archived_files.append(str(archive_path))
            print(f"📦 Archived: {report['path']} → {archive_path}")
        
        return archived_files
    
    def delete_reports(self, reports: List[Dict], dry_run: bool = False) -> List[str]:
        """Delete reports permanently"""
        deleted_paths = []
        
        for report in reports:
            if dry_run:
                print(f"[DRY RUN] Would delete: {report['path']} ({report['size_mb']:.1f}MB)")
            else:
                try:
                    shutil.rmtree(report['path'])
                    deleted_paths.append(str(report['path']))
                    print(f"🗑️  Deleted: {report['path']}")
                except OSError as e:
                    print(f"❌ Could not delete {report['path']}: {e}")
        
        return deleted_paths
    
    def generate_cleanup_report(self, keep: List[Dict], archived: List[Dict], 
                              deleted: List[Dict], policy: Dict) -> str:
        """Generate human-readable cleanup report"""
        
        total_reports = len(keep) + len(archived) + len(deleted)
        total_size = sum(r['size_mb'] for r in keep + archived + deleted)
        freed_size = sum(r['size_mb'] for r in archived + deleted)
        
        report = f"""# Report Cleanup Summary

**Cleanup Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Reports Processed**: {total_reports}
**Total Size Processed**: {total_size:.1f} MB
**Space Freed**: {freed_size:.1f} MB

## Retention Policy Applied

- **Keep Recent**: {policy.get('keep_days', 30)} days
- **Archive Before Delete**: {policy.get('archive_days', 90)} days  
- **Maximum Reports**: {policy.get('max_reports', 50)}
- **Keep Latest Per Version**: {policy.get('keep_latest_per_version', True)}
- **Preserve Structure v2.0**: {policy.get('preserve_structure_v2', True)}

## Results

### ✅ Reports Kept ({len(keep)})
"""
        
        for report in keep[:10]:  # Show first 10
            report += f"- **{report['path'].name}** (v{report['version']}, {report['age_days']} days old, {report['size_mb']:.1f}MB)\n"
        
        if len(keep) > 10:
            report += f"- ... and {len(keep) - 10} more reports\n"
        
        if archived:
            report += f"\n### 📦 Reports Archived ({len(archived)})\n"
            for report_item in archived[:10]:
                report += f"- **{report_item['path'].name}** (v{report_item['version']}, {report_item['age_days']} days old, {report_item['size_mb']:.1f}MB)\n"
            
            if len(archived) > 10:
                report += f"- ... and {len(archived) - 10} more reports\n"
        
        if deleted:
            report += f"\n### 🗑️ Reports Deleted ({len(deleted)})\n"
            for report_item in deleted[:10]:
                report += f"- **{report_item['path'].name}** (v{report_item['version']}, {report_item['age_days']} days old, {report_item['size_mb']:.1f}MB)\n"
            
            if len(deleted) > 10:
                report += f"- ... and {len(deleted) - 10} more reports\n"
        
        report += f"""
## Storage Summary

- **Active Reports**: {len(keep)} ({sum(r['size_mb'] for r in keep):.1f} MB)
- **Archived Reports**: {len(archived)} ({sum(r['size_mb'] for r in archived):.1f} MB)
- **Space Freed by Cleanup**: {freed_size:.1f} MB

## Recommendations

{"✅ Archive directory is healthy" if len(archived) < 100 else "⚠️  Consider cleaning archive directory"}
{"✅ Storage usage is reasonable" if sum(r['size_mb'] for r in keep) < 1000 else "⚠️  Consider more aggressive retention policy"}

---
*Generated by cleanup_old_reports.py*
"""
        
        return report
    
    def cleanup_with_policy(self, policy: Dict, dry_run: bool = False) -> Dict:
        """Execute full cleanup workflow with given policy"""
        
        print("🔍 Scanning reports...")
        reports = self.scan_reports()
        
        if not reports:
            print("ℹ️  No reports found to process")
            return {'status': 'no_reports'}
        
        print(f"📊 Found {len(reports)} reports:")
        total_size = sum(r['size_mb'] for r in reports)
        print(f"   Total size: {total_size:.1f} MB")
        
        # Apply retention policy
        keep_reports, archive_reports, delete_reports = self.apply_retention_policy(reports, policy)
        
        print(f"\n📋 Retention Policy Results:")
        print(f"   Keep: {len(keep_reports)} reports")
        print(f"   Archive: {len(archive_reports)} reports")
        print(f"   Delete: {len(delete_reports)} reports")
        
        if dry_run:
            print(f"\n[DRY RUN] Would process reports as follows:")
            
            for report in keep_reports[:5]:
                print(f"   KEEP: {report['path'].name} (v{report['version']}, {report['age_days']} days)")
            
            for report in archive_reports[:5]:
                print(f"   ARCHIVE: {report['path'].name} (v{report['version']}, {report['age_days']} days)")
            
            for report in delete_reports[:5]:
                print(f"   DELETE: {report['path'].name} (v{report['version']}, {report['age_days']} days)")
            
            return {
                'status': 'dry_run',
                'keep_count': len(keep_reports),
                'archive_count': len(archive_reports),
                'delete_count': len(delete_reports)
            }
        
        # Execute archival
        archived_files = []
        if archive_reports:
            print(f"\n📦 Archiving {len(archive_reports)} reports...")
            archived_files = self.archive_reports(archive_reports, policy.get('compression', 'gzip'))
        
        # Execute deletion
        deleted_paths = []
        if delete_reports:
            print(f"\n🗑️  Deleting {len(delete_reports)} reports...")
            deleted_paths = self.delete_reports(delete_reports)
        
        # Generate report
        cleanup_report = self.generate_cleanup_report(keep_reports, archive_reports, delete_reports, policy)
        report_file = self.base_dir / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(cleanup_report)
        
        print(f"\n📊 Cleanup complete!")
        print(f"   Cleanup report: {report_file}")
        print(f"   Archive directory: {self.archive_dir}")
        
        return {
            'status': 'complete',
            'keep_count': len(keep_reports),
            'archive_count': len(archive_reports),
            'delete_count': len(delete_reports),
            'archived_files': archived_files,
            'deleted_paths': deleted_paths,
            'report_file': str(report_file)
        }


def main():
    parser = argparse.ArgumentParser(description="Archive and cleanup old reports")
    parser.add_argument("--base-dir", default="reports", help="Base reports directory")
    parser.add_argument("--archive-dir", default="reports_archive", help="Archive directory")
    parser.add_argument("--keep-days", type=int, default=30, help="Days to keep reports active")
    parser.add_argument("--archive-days", type=int, default=90, help="Days before deleting archived reports")
    parser.add_argument("--max-reports", type=int, default=50, help="Maximum reports to keep active")
    parser.add_argument("--compression", choices=['gzip', 'tar', 'none'], default='gzip', help="Archive compression method")
    parser.add_argument("--no-version-keep", action="store_true", help="Don't keep latest of each version")
    parser.add_argument("--no-preserve-v2", action="store_true", help="Don't preserve structure v2.0 reports")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--scan-only", action="store_true", help="Only scan and report, don't clean up")
    
    args = parser.parse_args()
    
    cleanup_manager = ReportCleanupManager(args.base_dir, args.archive_dir)
    
    if args.scan_only:
        reports = cleanup_manager.scan_reports()
        if not reports:
            print("ℹ️  No reports found")
            return 0
        
        print(f"📊 Report Inventory ({len(reports)} reports):")
        
        by_structure = {}
        by_version = {}
        total_size = 0
        
        for report in reports:
            # Group by structure version
            struct = report['structure_version'] 
            by_structure[struct] = by_structure.get(struct, 0) + 1
            
            # Group by version
            version = report['version']
            by_version[version] = by_version.get(version, 0) + 1
            
            total_size += report['size_mb']
            
            print(f"  📁 {report['path'].name}")
            print(f"     Structure: {struct}, Version: v{version}")
            print(f"     Age: {report['age_days']} days, Size: {report['size_mb']:.1f}MB")
            print(f"     Files: {report['files_count']}")
            print()
        
        print(f"📈 Summary:")
        print(f"   Total Reports: {len(reports)}")
        print(f"   Total Size: {total_size:.1f} MB")
        print(f"   By Structure: {dict(by_structure)}")
        print(f"   By Version: {dict(by_version)}")
        
        return 0
    
    # Build retention policy
    policy = {
        'keep_days': args.keep_days,
        'archive_days': args.archive_days,
        'max_reports': args.max_reports,
        'compression': args.compression,
        'keep_latest_per_version': not args.no_version_keep,
        'preserve_structure_v2': not args.no_preserve_v2
    }
    
    print("🔧 Cleanup Policy:")
    for key, value in policy.items():
        print(f"   {key}: {value}")
    
    # Execute cleanup
    try:
        result = cleanup_manager.cleanup_with_policy(policy, dry_run=args.dry_run)
        
        if result['status'] == 'no_reports':
            print("✅ No reports found - nothing to clean up")
        elif result['status'] == 'dry_run':
            print(f"✅ Dry run complete - would process {result['keep_count'] + result['archive_count'] + result['delete_count']} reports")
        else:
            print(f"✅ Cleanup complete!")
            print(f"   Active: {result['keep_count']} reports")
            print(f"   Archived: {result['archive_count']} reports") 
            print(f"   Deleted: {result['delete_count']} reports")
            
        return 0
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())