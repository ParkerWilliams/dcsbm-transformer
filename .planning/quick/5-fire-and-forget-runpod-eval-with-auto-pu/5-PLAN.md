---
phase: quick-5
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/runpod_eval.sh
autonomous: true
requirements: [QUICK-5]

must_haves:
  truths:
    - "Script runs the full evaluation pipeline (train + eval + analysis + viz + report)"
    - "All results are git committed and pushed after successful run"
    - "Pod shuts down automatically after completion"
    - "If the pipeline fails, logs and partial outputs are still committed and pushed before shutdown"
  artifacts:
    - path: "scripts/runpod_eval.sh"
      provides: "Fire-and-forget RunPod evaluation wrapper"
      min_lines: 60
  key_links:
    - from: "scripts/runpod_eval.sh"
      to: "run_experiment.py"
      via: "python run_experiment.py --config config.json --verbose"
      pattern: "python.*run_experiment\\.py"
---

<objective>
Create a fire-and-forget bash wrapper script that runs the full DCSBM transformer evaluation pipeline on a RunPod GPU instance, auto-commits and pushes all results to git, then shuts down the pod to save money.

Purpose: Enable unattended GPU evaluation runs. User SSHs into RunPod, launches the script, disconnects, and comes back later to find results pushed to the repo.
Output: `scripts/runpod_eval.sh` -- a single executable bash script.
</objective>

<execution_context>
@/root/.claude/get-shit-done/workflows/execute-plan.md
@/root/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@run_experiment.py
@config.json
@.gitignore
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create fire-and-forget RunPod evaluation script</name>
  <files>scripts/runpod_eval.sh</files>
  <action>
Create `scripts/runpod_eval.sh` with the following structure. Use `#!/usr/bin/env bash` and `set -euo pipefail`.

**Configuration section (top of script):**
- `CONFIG_FILE` variable, default `config.json`, overridable via first CLI argument: `CONFIG_FILE="${1:-config.json}"`
- `RESULTS_DIR` variable, default `results`
- `LOG_FILE` variable: `eval_run_$(date +%Y%m%d_%H%M%S).log`
- `AUTO_SHUTDOWN` variable, default `true`, overridable via env var: `AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-true}"`
- `GIT_PUSH` variable, default `true`, overridable via env var: `GIT_PUSH="${GIT_PUSH:-true}"`

**Logging setup:**
- Use `exec > >(tee -a "$LOG_FILE") 2>&1` to tee all stdout/stderr to both console and log file
- Print a banner with timestamp, hostname, config file, GPU info (`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"`)
- Print git branch and current commit hash

**Cleanup/shutdown function:**
- Define `cleanup()` function that runs on EXIT trap
- Inside cleanup: if `AUTO_SHUTDOWN` is true, schedule shutdown:
  - Try `runpodctl stop pod "$RUNPOD_POD_ID"` first (RunPod's standard env var)
  - If runpodctl not available, try `curl -s -X POST "https://api.runpod.io/v2/${RUNPOD_POD_ID}/stop" -H "Authorization: Bearer ${RUNPOD_API_KEY}"` (only if both env vars exist)
  - If neither works, try `shutdown -h +1` as last resort (with a warning that this requires root)
  - Log which shutdown method was used
- Register: `trap cleanup EXIT`

**Git helper function:**
- Define `git_push_results()` that:
  - `cd` to the repo root (use `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` and `REPO_ROOT="$(dirname "$SCRIPT_DIR")"`)
  - `git add results/ "$LOG_FILE"` (add results dir and the log file)
  - `git add -u` (stage any modified tracked files like updated caches)
  - Check if there are staged changes: `git diff --cached --quiet && echo "No changes to commit" && return 0`
  - `git commit -m "results: runpod eval $(date -u +%Y-%m-%dT%H:%M:%SZ) [$(hostname)]"`
  - If `GIT_PUSH` is true: `git push origin HEAD` with retry logic (3 attempts, 5s sleep between, in case of transient network issues)

**Main pipeline execution:**
- `cd "$REPO_ROOT"`
- Check config file exists, exit 1 if not
- Check python is available and can import src: `python -c "import src" || { echo "ERROR: src package not importable. Install with: pip install -e ."; exit 1; }`
- Run: `python run_experiment.py --config "$CONFIG_FILE" --verbose`
- Capture exit code: use `set +e` before the python call, capture `$?`, then `set -e` again
- If exit code is 0:
  - Print success banner
  - Call `git_push_results`
- If exit code is non-zero:
  - Print failure banner with exit code
  - Still call `git_push_results` (so logs and any partial output get pushed for debugging)
  - Set a flag so cleanup knows the run failed (for the log message)

**Important details:**
- Do NOT hardcode any paths to virtual environments -- the script should work with whatever python is in PATH (user activates venv before running, or RunPod template has it pre-activated)
- Make the script `chmod +x` compatible (the shebang line handles this)
- Keep the script simple and readable -- it's a bash wrapper, not a framework
- Total script should be ~80-120 lines including comments
- Add a usage comment block at the top explaining: usage, env vars (AUTO_SHUTDOWN, GIT_PUSH, RUNPOD_POD_ID, RUNPOD_API_KEY), and examples
  </action>
  <verify>
    <automated>bash -n /root/Repos/dcsbm-transformer/scripts/runpod_eval.sh && echo "Syntax OK"</automated>
  </verify>
  <done>
    - `scripts/runpod_eval.sh` exists and passes bash syntax check
    - Script has executable shebang line
    - Script accepts config path as first argument with default to config.json
    - Script logs to timestamped file and tees to stdout
    - Script runs `python run_experiment.py --config ... --verbose`
    - Script git-adds results/, commits with timestamp, pushes with retry
    - Script calls pod shutdown on EXIT trap (runpodctl > API > shutdown fallback)
    - Script still pushes logs on pipeline failure for remote debugging
  </done>
</task>

</tasks>

<verification>
- `bash -n scripts/runpod_eval.sh` passes (no syntax errors)
- Script contains all key sections: logging, cleanup trap, git push, pipeline execution, error handling
- `grep -c 'run_experiment.py' scripts/runpod_eval.sh` returns >= 1
- `grep -c 'git push' scripts/runpod_eval.sh` returns >= 1
- `grep -c 'runpodctl\|RUNPOD' scripts/runpod_eval.sh` returns >= 1
- `grep -c 'trap cleanup' scripts/runpod_eval.sh` returns >= 1
</verification>

<success_criteria>
A single self-contained bash script at `scripts/runpod_eval.sh` that a user can copy to a RunPod instance, run with `bash scripts/runpod_eval.sh`, disconnect, and return to find results pushed to the git repo and the pod shut down.
</success_criteria>

<output>
After completion, create `.planning/quick/5-fire-and-forget-runpod-eval-with-auto-pu/5-SUMMARY.md`
</output>
