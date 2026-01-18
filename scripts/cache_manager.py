#!/usr/bin/env python3
"""
Automated Cache Management System
Manages Python cache and prevents stale method signature issues
"""

import os
import sys
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(logs_dir / "cache_manager.log"),
    filemode="a",
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Automated cache management system
    """

    def __init__(self):
        self.project_root = project_root
        self.cache_patterns = [
            '**/*.pyc',
            '**/__pycache__',
            '**/.pytest_cache',
            '**/node_modules/.cache',
            '**/.mypy_cache'
        ]
        self.critical_files = [
            'etl/data_storage.py',
            'etl/time_series_cv.py',
            'ai_llm/ollama_client.py',
            'scripts/run_etl_pipeline.py'
        ]

    def clear_all_caches(self) -> Dict[str, any]:
        """Clear all Python caches in the project"""
        try:
            cleared_items = []
            total_size = 0

            for pattern in self.cache_patterns:
                items = list(self.project_root.glob(pattern))
                for item in items:
                    if item.is_file():
                        size = item.stat().st_size
                        item.unlink()
                        cleared_items.append(str(item))
                        total_size += size
                    elif item.is_dir():
                        size = self._get_dir_size(item)
                        shutil.rmtree(item)
                        cleared_items.append(str(item))
                        total_size += size

            logger.info(f"Cleared {len(cleared_items)} cache items ({total_size / 1024:.1f} KB)")

            return {
                "status": "success",
                "cleared_items": len(cleared_items),
                "total_size_kb": total_size / 1024,
                "items": cleared_items[:10],  # First 10 items
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory"""
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size for {path}: {e}")
        return total_size

    def validate_critical_files(self) -> Dict[str, any]:
        """Validate that critical files are accessible and importable"""
        results = {}

        for file_path in self.critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    # Try to import the module
                    module_name = file_path.replace('/', '.').replace('.py', '')
                    __import__(module_name)
                    results[file_path] = {"status": "ok", "importable": True}
                except Exception as e:
                    results[file_path] = {"status": "error", "error": str(e), "importable": False}
            else:
                results[file_path] = {"status": "missing", "importable": False}

        return results

    def check_cache_health(self) -> Dict[str, any]:
        """Check cache health and identify potential issues"""
        try:
            issues = []
            warnings = []

            # Check for stale .pyc files
            pyc_files = list(self.project_root.glob('**/*.pyc'))
            for pyc_file in pyc_files:
                py_file = pyc_file.with_suffix('')
                if py_file.exists():
                    pyc_mtime = pyc_file.stat().st_mtime
                    py_mtime = py_file.stat().st_mtime
                    if pyc_mtime < py_mtime:
                        issues.append(f"Stale .pyc file: {pyc_file}")

            # Check for large cache directories
            cache_dirs = list(self.project_root.glob('**/__pycache__'))
            for cache_dir in cache_dirs:
                size = self._get_dir_size(cache_dir)
                if size > 10 * 1024 * 1024:  # 10MB
                    warnings.append(f"Large cache directory: {cache_dir} ({size / 1024 / 1024:.1f} MB)")

            # Check for old cache files
            cutoff_time = datetime.now() - timedelta(days=7)
            old_files = []
            for pattern in self.cache_patterns:
                for item in self.project_root.glob(pattern):
                    if item.is_file():
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        if mtime < cutoff_time:
                            old_files.append(str(item))

            if old_files:
                warnings.append(f"Found {len(old_files)} old cache files")

            return {
                "status": "healthy" if not issues else "issues_found",
                "issues": issues,
                "warnings": warnings,
                "total_issues": len(issues),
                "total_warnings": len(warnings),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def optimize_imports(self) -> Dict[str, any]:
        """Optimize imports by clearing caches and reimporting"""
        try:
            # Clear caches first
            clear_result = self.clear_all_caches()

            # Validate critical files
            validation_result = self.validate_critical_files()

            # Check for any import issues
            import_issues = []
            for file_path, result in validation_result.items():
                if not result.get("importable", False):
                    import_issues.append(f"{file_path}: {result.get('error', 'Unknown error')}")

            return {
                "status": "success" if not import_issues else "partial_success",
                "cache_cleared": clear_result["status"] == "success",
                "import_issues": import_issues,
                "validation_results": validation_result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def create_cache_cleanup_schedule(self) -> str:
        """Create a scheduled cache cleanup script"""
        script_content = f'''#!/bin/bash
# Automated Cache Cleanup Script
# Generated on {datetime.now().isoformat()}

echo "Starting automated cache cleanup..."

# Clear Python caches
find {self.project_root} -name "*.pyc" -delete
find {self.project_root} -name "__pycache__" -type d -exec rm -rf {{}} + 2>/dev/null || true
find {self.project_root} -name ".pytest_cache" -type d -exec rm -rf {{}} + 2>/dev/null || true

# Clear other caches
find {self.project_root} -name ".mypy_cache" -type d -exec rm -rf {{}} + 2>/dev/null || true

echo "Cache cleanup completed at $(date)"
'''

        script_path = self.project_root / "scripts" / "cleanup_cache.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)

        return str(script_path)

    def generate_cache_report(self) -> Dict[str, any]:
        """Generate comprehensive cache report"""
        try:
            # Get cache health
            health = self.check_cache_health()

            # Get cache statistics
            cache_stats = self._get_cache_statistics()

            # Get critical file validation
            validation = self.validate_critical_files()

            return {
                "health": health,
                "statistics": cache_stats,
                "validation": validation,
                "recommendations": self._generate_recommendations(health, cache_stats),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_cache_statistics(self) -> Dict[str, any]:
        """Get cache statistics"""
        stats = {
            "pyc_files": 0,
            "cache_dirs": 0,
            "total_cache_size": 0,
            "largest_cache": None,
            "oldest_cache": None
        }

        try:
            # Count .pyc files
            pyc_files = list(self.project_root.glob('**/*.pyc'))
            stats["pyc_files"] = len(pyc_files)

            # Count cache directories
            cache_dirs = list(self.project_root.glob('**/__pycache__'))
            stats["cache_dirs"] = len(cache_dirs)

            # Calculate total size
            total_size = 0
            largest_size = 0
            oldest_time = None

            for pattern in self.cache_patterns:
                for item in self.project_root.glob(pattern):
                    if item.is_file():
                        size = item.stat().st_size
                        total_size += size
                        if size > largest_size:
                            largest_size = size
                            stats["largest_cache"] = str(item)
                    elif item.is_dir():
                        size = self._get_dir_size(item)
                        total_size += size
                        if size > largest_size:
                            largest_size = size
                            stats["largest_cache"] = str(item)

                        # Check age
                        mtime = item.stat().st_mtime
                        if oldest_time is None or mtime < oldest_time:
                            oldest_time = mtime
                            stats["oldest_cache"] = str(item)

            stats["total_cache_size"] = total_size
            stats["total_cache_size_mb"] = total_size / (1024 * 1024)

        except Exception as e:
            stats["error"] = str(e)

        return stats

    def _generate_recommendations(self, health: Dict, stats: Dict) -> List[str]:
        """Generate cache management recommendations"""
        recommendations = []

        if health.get("total_issues", 0) > 0:
            recommendations.append("Clear stale cache files to resolve import issues")

        if stats.get("total_cache_size_mb", 0) > 100:
            recommendations.append("Consider clearing large cache directories to free space")

        if stats.get("pyc_files", 0) > 1000:
            recommendations.append("High number of .pyc files detected - consider cleanup")

        if not recommendations:
            recommendations.append("Cache is healthy - no immediate action needed")

        return recommendations


def main():
    """Main cache management function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    manager = CacheManager()

    print("🧹 Starting Cache Management...")

    # Check cache health
    print("\n🔍 Checking cache health...")
    health = manager.check_cache_health()
    print(f"Status: {health['status']}")
    if health.get('issues'):
        print(f"Issues found: {len(health['issues'])}")
        for issue in health['issues'][:3]:  # Show first 3
            print(f"  - {issue}")

    # Clear caches if needed
    if health.get('total_issues', 0) > 0:
        print("\n🧹 Clearing caches...")
        clear_result = manager.clear_all_caches()
        print(f"Cleared {clear_result.get('cleared_items', 0)} items")

    # Validate critical files
    print("\n✅ Validating critical files...")
    validation = manager.validate_critical_files()
    for file_path, result in validation.items():
        status = "✅" if result.get("importable") else "❌"
        print(f"  {status} {file_path}")

    # Generate report
    print("\n📊 Generating cache report...")
    report = manager.generate_cache_report()

    print(f"\n📈 Cache Statistics:")
    stats = report.get('statistics', {})
    print(f"  - .pyc files: {stats.get('pyc_files', 0)}")
    print(f"  - Cache directories: {stats.get('cache_dirs', 0)}")
    print(f"  - Total size: {stats.get('total_cache_size_mb', 0):.1f} MB")

    print(f"\n💡 Recommendations:")
    for rec in report.get('recommendations', []):
        print(f"  - {rec}")

    print("\n✅ Cache management complete")


if __name__ == "__main__":
    main()
