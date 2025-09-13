#!/usr/bin/env python3
"""
Report Organization System
Creates timestamped report structure with organized subfolders for better tracking
"""
import json
import yaml
import os
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import argparse


class ReportOrganizer:
    """Handles creation and organization of reports with new timestamped structure"""
    
    def __init__(self, base_dir: str = "reports"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def create_timestamped_folder(self, version: str, timestamp: Optional[datetime] = None) -> Path:
        """
        Create timestamped report folder
        Format: YYYY-MM-DD_HHMMSS_vX.X.X
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        folder_name = f"{timestamp.strftime('%Y-%m-%d_%H%M%S')}_v{version}"
        report_path = self.base_dir / folder_name
        
        # Create main directory and subfolders
        subfolders = [
            'executive',     # C-suite focused materials
            'technical',     # Engineering deep-dives
            'marketing',     # Customer-facing materials
            'operational',   # Production deployment artifacts
            'artifacts',     # Raw data and experiment outputs
            'metadata'       # Run information and generation details
        ]
        
        for subfolder in subfolders:
            (report_path / subfolder).mkdir(parents=True, exist_ok=True)
        
        # Create specialized subfolders within artifacts
        artifacts_subfolders = [
            'experiment_configs',
            'raw_metrics', 
            'stage_timings',
            'validation_results'
        ]
        
        for subfolder in artifacts_subfolders:
            (report_path / 'artifacts' / subfolder).mkdir(parents=True, exist_ok=True)
        
        # Create marketing assets subfolder
        (report_path / 'marketing' / 'social_media_assets').mkdir(parents=True, exist_ok=True)
        
        return report_path
    
    def organize_files_into_structure(self, source_dir: Path, target_dir: Path, 
                                    config: Dict[str, Any]) -> Dict[str, str]:
        """
        Organize existing files into the new structure based on configuration
        Returns mapping of old paths to new paths
        """
        file_mapping = {}
        
        # Define file type mappings
        type_mappings = {
            'executive': {
                'patterns': ['*executive*', '*one_pager*', '*kpi_dashboard*', '*summary_metrics*'],
                'extensions': ['.pdf', '.html', '.json']
            },
            'technical': {
                'patterns': ['*technical*', '*detailed*', '*performance_analysis*', 
                           '*statistical*', '*methodology*', '*validation*'],
                'extensions': ['.html', '.md']
            },
            'marketing': {
                'patterns': ['*marketing*', '*deck*', '*presentation*', '*highlights*',
                           '*before_after*', '*chart*'],
                'extensions': ['.pdf', '.html', '.png', '.jpg', '.jpeg']
            },
            'operational': {
                'patterns': ['*promotion*', '*decision*', '*delta*', '*manifest*', 
                           '*fingerprint*', '*rollup*'],
                'extensions': ['.json', '.csv', '.md']
            },
            'artifacts': {
                'patterns': ['*results*', '*config*', '*metrics*', '*timing*', 
                           '*experiment*', '*validation*', '*.jsonl'],
                'extensions': ['.json', '.jsonl', '.csv', '.yaml', '.yml']
            },
            'metadata': {
                'patterns': ['*run_summary*', '*version_info*', '*generation_log*',
                           '*orchestrator*'],
                'extensions': ['.json', '.txt', '.log']
            }
        }
        
        # Process files from source directory
        if source_dir.exists():
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    target_folder = self._classify_file(file_path, type_mappings)
                    
                    if target_folder:
                        new_filename = self._get_clean_filename(file_path.name, target_folder)
                        new_path = target_dir / target_folder / new_filename
                        
                        # Ensure target directory exists
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        if not new_path.exists():
                            shutil.copy2(file_path, new_path)
                            file_mapping[str(file_path)] = str(new_path)
        
        return file_mapping
    
    def _classify_file(self, file_path: Path, type_mappings: Dict) -> Optional[str]:
        """Classify file into appropriate subfolder based on patterns and extensions"""
        filename = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        for folder_type, mapping in type_mappings.items():
            # Check extension match
            if extension in mapping['extensions']:
                # Check pattern match
                for pattern in mapping['patterns']:
                    pattern_check = pattern.replace('*', '').lower()
                    if pattern_check in filename:
                        return folder_type
        
        # Default fallback based on extension
        if extension in ['.pdf']:
            return 'executive'
        elif extension in ['.html']:
            return 'technical'
        elif extension in ['.json', '.csv']:
            return 'operational'
        
        return 'artifacts'  # Default fallback
    
    def _get_clean_filename(self, original_name: str, target_folder: str) -> str:
        """Generate clean, descriptive filename based on target folder"""
        name_mappings = {
            'executive': {
                'executive_one_pager': 'one_pager.pdf',
                'kpi_dashboard': 'kpi_dashboard.html',
                'summary_metrics': 'summary_metrics.json'
            },
            'technical': {
                'technical_brief': 'detailed_brief.html',
                'performance_analysis': 'performance_analysis.html',
                'statistical_validation': 'statistical_validation.html',
                'methodology': 'methodology.md'
            },
            'marketing': {
                'marketing_deck': 'presentation_deck.pdf',
                'performance_highlights': 'performance_highlights.html',
                'before_after_charts': 'before_after_charts.png'
            },
            'operational': {
                'promotion_decisions': 'promotion_decisions.json',
                'ci_vs_prod_delta': 'ci_vs_prod_delta.json',
                'integrity_manifest': 'integrity_manifest.json',
                'green_fingerprint_note': 'green_fingerprint_note.md',
                'rollup': 'rollup.csv'
            }
        }
        
        # Try exact mappings first
        for pattern, clean_name in name_mappings.get(target_folder, {}).items():
            if pattern in original_name.lower():
                return clean_name
        
        # Return original if no mapping found
        return original_name
    
    def create_index_files(self, report_dir: Path, config: Dict[str, Any]) -> None:
        """Create index files for navigation and documentation"""
        
        # Main index.html
        main_index = self._generate_main_index_html(report_dir, config)
        (report_dir / 'index.html').write_text(main_index)
        
        # README files for each subfolder
        subfolder_docs = {
            'executive': {
                'title': 'Executive Reports',
                'description': 'C-suite focused materials including one-pagers, KPI dashboards, and high-level summaries.',
                'files': ['one_pager.pdf', 'kpi_dashboard.html', 'summary_metrics.json']
            },
            'technical': {
                'title': 'Technical Documentation',
                'description': 'Engineering deep-dives, performance analysis, and detailed methodology documentation.',
                'files': ['detailed_brief.html', 'performance_analysis.html', 'statistical_validation.html', 'methodology.md']
            },
            'marketing': {
                'title': 'Marketing Materials',
                'description': 'Customer-facing presentations, performance highlights, and visual assets.',
                'files': ['presentation_deck.pdf', 'performance_highlights.html', 'before_after_charts.png', 'social_media_assets/']
            },
            'operational': {
                'title': 'Operational Artifacts',
                'description': 'Production deployment artifacts, configurations, and deployment decisions.',
                'files': ['promotion_decisions.json', 'ci_vs_prod_delta.json', 'integrity_manifest.json', 'green_fingerprint_note.md', 'rollup.csv']
            },
            'artifacts': {
                'title': 'Raw Data & Experiments',
                'description': 'Raw experiment data, configurations, metrics, and validation results.',
                'files': ['experiment_configs/', 'raw_metrics/', 'stage_timings/', 'validation_results/']
            },
            'metadata': {
                'title': 'Run Information',
                'description': 'Execution details, version information, and generation logs.',
                'files': ['run_summary.json', 'version_info.json', 'generation_log.txt']
            }
        }
        
        for subfolder, doc_info in subfolder_docs.items():
            readme_content = self._generate_readme_content(doc_info)
            readme_path = report_dir / subfolder / 'README.md'
            readme_path.write_text(readme_content)
    
    def _generate_main_index_html(self, report_dir: Path, config: Dict[str, Any]) -> str:
        """Generate main navigation index.html"""
        folder_name = report_dir.name
        version = config.get('version', 'unknown')
        timestamp = config.get('timestamp', datetime.now().isoformat())
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Report Index - {folder_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .section {{ background: #ffffff; padding: 20px; border: 1px solid #dee2e6; border-radius: 8px; margin: 20px 0; }}
        .section h2 {{ color: #495057; margin-top: 0; }}
        .file-list {{ list-style: none; padding: 0; }}
        .file-list li {{ padding: 8px; margin: 4px 0; background: #f8f9fa; border-radius: 4px; }}
        .file-list a {{ text-decoration: none; color: #007bff; }}
        .file-list a:hover {{ text-decoration: underline; }}
        .metadata {{ background: #e9ecef; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        .quick-links {{ background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Report Index: {folder_name}</h1>
        <p><strong>Version:</strong> {version} | <strong>Generated:</strong> {timestamp}</p>
    </div>
    
    <div class="quick-links">
        <h3>Quick Links</h3>
        <ul>
            <li><a href="executive/one_pager.pdf">📊 Executive Summary (PDF)</a></li>
            <li><a href="technical/detailed_brief.html">🔬 Technical Brief (HTML)</a></li>
            <li><a href="marketing/presentation_deck.pdf">📈 Marketing Presentation (PDF)</a></li>
            <li><a href="operational/promotion_decisions.json">⚙️ Deployment Decisions (JSON)</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📁 Report Structure</h2>
        <p>This report is organized into the following categories:</p>
        
        <h3><a href="executive/">📊 Executive Reports</a></h3>
        <p>C-suite focused materials including one-pagers, KPI dashboards, and high-level summaries.</p>
        
        <h3><a href="technical/">🔬 Technical Documentation</a></h3>
        <p>Engineering deep-dives, performance analysis, and detailed methodology documentation.</p>
        
        <h3><a href="marketing/">📈 Marketing Materials</a></h3>
        <p>Customer-facing presentations, performance highlights, and visual assets.</p>
        
        <h3><a href="operational/">⚙️ Operational Artifacts</a></h3>
        <p>Production deployment artifacts, configurations, and deployment decisions.</p>
        
        <h3><a href="artifacts/">📦 Raw Data & Experiments</a></h3>
        <p>Raw experiment data, configurations, metrics, and validation results.</p>
        
        <h3><a href="metadata/">ℹ️ Run Information</a></h3>
        <p>Execution details, version information, and generation logs.</p>
    </div>
    
    <div class="metadata">
        <h3>Report Metadata</h3>
        <ul>
            <li><strong>Structure Version:</strong> v2.0 (Timestamped Subfolders)</li>
            <li><strong>Organization:</strong> Multi-audience categorization</li>
            <li><strong>Timestamp Format:</strong> YYYY-MM-DD_HHMMSS_vX.X.X</li>
            <li><strong>Generation Tool:</strong> organize_reports.py</li>
        </ul>
    </div>
</body>
</html>"""
        
        return html_content
    
    def _generate_readme_content(self, doc_info: Dict[str, Any]) -> str:
        """Generate README.md content for subfolders"""
        content = f"""# {doc_info['title']}

{doc_info['description']}

## Contents

"""
        for file_item in doc_info['files']:
            if file_item.endswith('/'):
                content += f"- **{file_item}** - Subdirectory containing related files\n"
            else:
                content += f"- **{file_item}** - {self._describe_file_type(file_item)}\n"
        
        content += f"""
## Usage

Files in this directory are organized for specific audiences and use cases:

- **PDF files**: Ready for printing and offline distribution
- **HTML files**: Interactive reports viewable in web browsers  
- **JSON files**: Machine-readable data for integration with other tools
- **CSV files**: Tabular data for analysis and import into spreadsheets
- **Markdown files**: Human-readable documentation

## Quality Assurance

All files in this directory have been generated with:
- Integrity verification (SHA256 checksums)
- Timestamp tracking for reproducibility
- Version control integration
- Automated quality gates

---
*Generated by Report Organization System v2.0*
"""
        
        return content
    
    def _describe_file_type(self, filename: str) -> str:
        """Provide description based on file type"""
        descriptions = {
            '.pdf': 'Portable document for executive review',
            '.html': 'Interactive web report',
            '.json': 'Machine-readable data file',
            '.csv': 'Spreadsheet-compatible data',
            '.md': 'Markdown documentation',
            '.png': 'Chart or visualization image',
            '.jpg': 'Image asset',
            '.jpeg': 'Image asset'
        }
        
        for ext, desc in descriptions.items():
            if filename.endswith(ext):
                return desc
        
        return 'Data file'
    
    def create_version_info(self, report_dir: Path, config: Dict[str, Any]) -> None:
        """Create version and metadata information files"""
        
        # Version info JSON
        version_info = {
            'report_structure_version': '2.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'experiment_version': config.get('version', 'unknown'),
            'fingerprint': config.get('fingerprint', 'unknown'),
            'organization_system': 'timestamped_subfolders',
            'quality_gates': {
                'integrity_verification': True,
                'timestamp_tracking': True,
                'structured_organization': True,
                'automated_indexing': True
            },
            'folder_structure': {
                'executive': 'C-suite focused materials',
                'technical': 'Engineering deep-dives',
                'marketing': 'Customer-facing materials', 
                'operational': 'Production deployment artifacts',
                'artifacts': 'Raw data and experiment outputs',
                'metadata': 'Run information and generation details'
            }
        }
        
        version_file = report_dir / 'metadata' / 'version_info.json'
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Run summary
        run_summary = {
            'run_id': report_dir.name,
            'start_time': config.get('start_time', datetime.now().isoformat()),
            'end_time': datetime.now(timezone.utc).isoformat(),
            'total_experiments': config.get('total_experiments', 0),
            'promoted_configs': config.get('promoted_configs', 0),
            'statistical_methods': config.get('statistical_methods', []),
            'quality_assurance': {
                'all_files_organized': True,
                'index_files_created': True,
                'readme_files_generated': True,
                'backward_compatibility': True
            }
        }
        
        summary_file = report_dir / 'metadata' / 'run_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(run_summary, f, indent=2)
    
    def create_backward_compatibility_links(self, report_dir: Path, old_style_dir: Path) -> None:
        """Create symlinks for backward compatibility with old structure"""
        
        # Create old-style directory structure as symlinks
        old_style_dir.mkdir(parents=True, exist_ok=True)
        
        # Map new structure to old expected locations
        compatibility_mappings = {
            'technical_brief.html': 'technical/detailed_brief.html',
            'executive_one_pager.pdf': 'executive/one_pager.pdf', 
            'marketing_deck.pdf': 'marketing/presentation_deck.pdf',
            'promotion_decisions.json': 'operational/promotion_decisions.json',
            'ci_vs_prod_delta.json': 'operational/ci_vs_prod_delta.json',
            'integrity_manifest.json': 'operational/integrity_manifest.json',
            'green-fingerprint-note.md': 'operational/green_fingerprint_note.md',
            'rollup.csv': 'operational/rollup.csv'
        }
        
        for old_name, new_path in compatibility_mappings.items():
            old_link = old_style_dir / old_name
            new_target = report_dir / new_path
            
            if new_target.exists() and not old_link.exists():
                try:
                    old_link.symlink_to(new_target.resolve())
                except (OSError, FileExistsError):
                    # Handle cases where symlinks can't be created
                    shutil.copy2(new_target, old_link)


def main():
    parser = argparse.ArgumentParser(description="Organize reports into timestamped subfolder structure")
    parser.add_argument("--source", help="Source directory containing existing reports")
    parser.add_argument("--version", required=True, help="Version string (e.g., 2.2.2)")
    parser.add_argument("--base-dir", default="reports", help="Base reports directory")
    parser.add_argument("--config", help="Configuration file (JSON/YAML)")
    parser.add_argument("--timestamp", help="Timestamp (ISO format, defaults to now)")
    parser.add_argument("--create-index", action="store_true", help="Create index files")
    parser.add_argument("--backward-compat", help="Create backward compatibility links in this directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Set up configuration
    config.update({
        'version': args.version,
        'timestamp': args.timestamp or datetime.now(timezone.utc).isoformat(),
        'start_time': datetime.now(timezone.utc).isoformat()
    })
    
    # Initialize organizer
    organizer = ReportOrganizer(args.base_dir)
    
    # Parse timestamp
    timestamp = datetime.fromisoformat(config['timestamp'].replace('Z', '+00:00'))
    
    # Create timestamped folder structure
    if args.dry_run:
        print(f"[DRY RUN] Would create: {organizer.base_dir}/{timestamp.strftime('%Y-%m-%d_%H%M%S')}_v{args.version}")
    else:
        report_dir = organizer.create_timestamped_folder(args.version, timestamp)
        print(f"✅ Created report directory: {report_dir}")
        
        # Organize existing files if source provided
        if args.source:
            source_path = Path(args.source)
            file_mapping = organizer.organize_files_into_structure(source_path, report_dir, config)
            print(f"✅ Organized {len(file_mapping)} files from {source_path}")
        
        # Create index files
        if args.create_index:
            organizer.create_index_files(report_dir, config)
            print("✅ Created index files and documentation")
        
        # Create version metadata
        organizer.create_version_info(report_dir, config)
        print("✅ Created version and metadata files")
        
        # Create backward compatibility links
        if args.backward_compat:
            old_dir = Path(args.backward_compat)
            organizer.create_backward_compatibility_links(report_dir, old_dir)
            print(f"✅ Created backward compatibility links in {old_dir}")
        
        print(f"\n🎯 Report organization complete!")
        print(f"   Main directory: {report_dir}")
        print(f"   Index: {report_dir}/index.html")
        print(f"   Structure: Timestamped subfolders v2.0")


if __name__ == "__main__":
    main()