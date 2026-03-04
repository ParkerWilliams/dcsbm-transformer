#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# runpod_eval.sh -- Fire-and-forget RunPod evaluation wrapper
#
# Runs the full DCSBM transformer evaluation pipeline on a RunPod GPU
# instance, auto-commits and pushes results to git, then shuts down
# the pod to save money.
#
# Usage:
#   bash scripts/runpod_eval.sh                  # uses config.json
#   bash scripts/runpod_eval.sh my_config.json   # custom config
#
# Environment variables:
#   AUTO_SHUTDOWN  (default: true)  -- shut down pod after completion
#   GIT_PUSH       (default: true)  -- push results to remote
#   RUNPOD_POD_ID  -- set by RunPod; used for pod shutdown
#   RUNPOD_API_KEY -- optional; used for API-based shutdown fallback
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-config.json}"
RESULTS_DIR="results"
LOG_FILE="eval_run_$(date +%Y%m%d_%H%M%S).log"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-true}"
GIT_PUSH="${GIT_PUSH:-true}"
RUN_FAILED=0

# ── Logging setup ─────────────────────────────────────────────────────
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  DCSBM Transformer -- RunPod Evaluation"
echo "============================================================"
echo "Timestamp : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Hostname  : $(hostname)"
echo "Config    : $CONFIG_FILE"
echo "GPU       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "Branch    : $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "Commit    : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Auto-shut : $AUTO_SHUTDOWN"
echo "Git-push  : $GIT_PUSH"
echo "Log file  : $LOG_FILE"
echo "============================================================"

# ── Cleanup / shutdown function ───────────────────────────────────────
cleanup() {
    echo ""
    echo "── Cleanup ──────────────────────────────────────────────────"
    if [ "$AUTO_SHUTDOWN" = "true" ]; then
        if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
            echo "Shutting down via runpodctl (pod $RUNPOD_POD_ID)..."
            runpodctl stop pod "$RUNPOD_POD_ID" && echo "Shutdown: runpodctl succeeded" || echo "Shutdown: runpodctl failed"
        elif [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
            echo "Shutting down via RunPod API (pod $RUNPOD_POD_ID)..."
            curl -s -X POST "https://api.runpod.io/v2/${RUNPOD_POD_ID}/stop" \
                -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
                && echo "Shutdown: API call succeeded" \
                || echo "Shutdown: API call failed"
        else
            echo "WARNING: runpodctl not found and RUNPOD_POD_ID/RUNPOD_API_KEY not set."
            echo "Attempting 'shutdown -h +1' (requires root)..."
            shutdown -h +1 2>/dev/null && echo "Shutdown: scheduled in 1 minute" || echo "Shutdown: failed (not root?)"
        fi
    else
        echo "AUTO_SHUTDOWN=false -- skipping pod shutdown."
    fi
}
trap cleanup EXIT

# ── Git push helper ───────────────────────────────────────────────────
git_push_results() {
    cd "$REPO_ROOT"
    git add "${RESULTS_DIR}/" "$LOG_FILE" 2>/dev/null || true
    git add -u 2>/dev/null || true

    if git diff --cached --quiet; then
        echo "No changes to commit."
        return 0
    fi

    git commit -m "results: runpod eval $(date -u +%Y-%m-%dT%H:%M:%SZ) [$(hostname)]"

    if [ "$GIT_PUSH" = "true" ]; then
        local attempt=0
        local max_attempts=3
        while [ $attempt -lt $max_attempts ]; do
            attempt=$((attempt + 1))
            echo "git push attempt $attempt/$max_attempts..."
            if git push origin HEAD; then
                echo "Push succeeded."
                return 0
            fi
            if [ $attempt -lt $max_attempts ]; then
                echo "Push failed, retrying in 5s..."
                sleep 5
            fi
        done
        echo "ERROR: git push failed after $max_attempts attempts."
        return 1
    fi
}

# ── Main pipeline ─────────────────────────────────────────────────────
cd "$REPO_ROOT"

# Validate config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Validate python environment
python -c "import src" 2>/dev/null || {
    echo "ERROR: src package not importable. Install with: pip install -e ."
    exit 1
}

echo ""
echo "── Running pipeline ───────────────────────────────────────────"

set +e
python run_experiment.py --config "$CONFIG_FILE" --verbose
PIPELINE_EXIT=$?
set -e

echo ""
if [ $PIPELINE_EXIT -eq 0 ]; then
    echo "============================================================"
    echo "  Pipeline SUCCEEDED"
    echo "============================================================"
else
    echo "============================================================"
    echo "  Pipeline FAILED (exit code $PIPELINE_EXIT)"
    echo "============================================================"
    RUN_FAILED=1
fi

# Always push results (success or failure) so logs are available
echo ""
echo "── Committing and pushing results ─────────────────────────────"
git_push_results || echo "WARNING: git push failed, results may not be on remote."

if [ $RUN_FAILED -ne 0 ]; then
    echo "Exiting with pipeline failure code."
    exit $PIPELINE_EXIT
fi
