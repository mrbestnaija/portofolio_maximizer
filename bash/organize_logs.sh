#!/usr/bin/env bash
#
# organize_logs.sh - Automatically organize logs by phase and category
#
# Usage: bash/organize_logs.sh [--dry-run]
#

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[INFO] DRY RUN MODE - No files will be moved"
fi

LOGS_DIR="logs"
cd "$(dirname "$0")/.."

# Create directory structure
echo "[INFO] Creating log directory structure..."
mkdir -p "$LOGS_DIR"/{phase7.5,phase7.6,phase7.7,phase7.8,archive,automation,brutal,cron,errors,events,forecast_audits,forecast_audits_cache,hyperopt,live_runs,performance,signals,stages,warnings}

# Function to move logs
move_log() {
    local pattern="$1"
    local dest_dir="$2"
    local count=0

    for file in "$LOGS_DIR"/$pattern 2>/dev/null; do
        if [[ -f "$file" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "[DRY-RUN] Would move: $file -> $dest_dir/"
            else
                echo "[MOVE] $file -> $dest_dir/"
                mv "$file" "$dest_dir/"
            fi
            ((count++))
        fi
    done

    if [[ $count -gt 0 ]]; then
        echo "[OK] Moved $count file(s) matching '$pattern' to $dest_dir/"
    fi
}

# Organize phase logs
echo ""
echo "=== Organizing Phase Logs ==="
move_log "phase7.5_*.log" "$LOGS_DIR/phase7.5"
move_log "phase7.6_*.log" "$LOGS_DIR/phase7.6"
move_log "phase7.7_*.log" "$LOGS_DIR/phase7.7"
move_log "phase7.8_*.log" "$LOGS_DIR/phase7.8"

# Archive old large logs (>50MB and older than 7 days)
echo ""
echo "=== Archiving Large Logs ==="
if [[ "$DRY_RUN" == true ]]; then
    find "$LOGS_DIR" -maxdepth 1 -name "*.log" -size +50M -mtime +7 2>/dev/null | while read -r file; do
        echo "[DRY-RUN] Would archive: $file -> $LOGS_DIR/archive/"
    done
else
    count=0
    find "$LOGS_DIR" -maxdepth 1 -name "*.log" -size +50M -mtime +7 2>/dev/null | while read -r file; do
        echo "[ARCHIVE] $file -> $LOGS_DIR/archive/"
        mv "$file" "$LOGS_DIR/archive/"
        ((count++))
    done
    if [[ $count -gt 0 ]]; then
        echo "[OK] Archived $count large log file(s)"
    else
        echo "[INFO] No large logs to archive (>50MB and >7 days old)"
    fi
fi

# Compress archived logs
echo ""
echo "=== Compressing Archived Logs ==="
if [[ "$DRY_RUN" == true ]]; then
    find "$LOGS_DIR/archive" -name "*.log" -type f 2>/dev/null | while read -r file; do
        echo "[DRY-RUN] Would compress: $file"
    done
else
    count=0
    find "$LOGS_DIR/archive" -name "*.log" -type f 2>/dev/null | while read -r file; do
        echo "[COMPRESS] $file"
        gzip -9 "$file" 2>/dev/null || {
            echo "[WARN] Failed to compress $file (gzip not available), skipping"
        }
        ((count++))
    done
    if [[ $count -gt 0 ]]; then
        echo "[OK] Compressed $count archived log file(s)"
    else
        echo "[INFO] No uncompressed logs in archive/"
    fi
fi

# Summary
echo ""
echo "=== Summary ==="
echo "Phase 7.5 logs: $(find "$LOGS_DIR/phase7.5" -name "*.log" 2>/dev/null | wc -l)"
echo "Phase 7.6 logs: $(find "$LOGS_DIR/phase7.6" -name "*.log" 2>/dev/null | wc -l)"
echo "Phase 7.7 logs: $(find "$LOGS_DIR/phase7.7" -name "*.log" 2>/dev/null | wc -l)"
echo "Phase 7.8 logs: $(find "$LOGS_DIR/phase7.8" -name "*.log" 2>/dev/null | wc -l)"
echo "Archived logs: $(find "$LOGS_DIR/archive" -type f 2>/dev/null | wc -l)"
echo ""
echo "Total logs directory size: $(du -sh "$LOGS_DIR" 2>/dev/null | cut -f1)"

echo ""
echo "[SUCCESS] Log organization complete!"
if [[ "$DRY_RUN" == true ]]; then
    echo "[INFO] This was a dry run. Re-run without --dry-run to apply changes."
fi
