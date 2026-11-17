#!/bin/bash
# Poll the Kaggle CPU validation kernel until it completes, then pull metrics.
PY="/c/Users/krish/AppData/Local/Programs/Python/Python312/python.exe"
REF="krishanbansal/litterguard-cpu-validation"
ROOT="/c/Users/krish/.zcode/workspace/697a15d4-0293-42c9-9dc6-78d11719a678/litter-guard"
cd "$ROOT" || exit 1

for i in $(seq 1 45); do
  S=$("$PY" -m kaggle kernels status "$REF" 2>&1 | tail -1)
  echo "[$(date +%H:%M:%S)] $S"
  if echo "$S" | grep -q COMPLETE; then
    echo "=== DONE ==="
    mkdir -p kaggle/output-cpu2
    "$PY" -m kaggle kernels output "$REF" -p kaggle/output-cpu2 >/dev/null 2>&1
    echo "--- metrics.json ---"
    cat kaggle/output-cpu2/metrics.json 2>/dev/null
    echo "--- log tail ---"
    tail -30 kaggle/output-cpu2/litterguard-cpu-validation.log 2>/dev/null | tr -d '\n' | tail -c 3000
    exit 0
  fi
  sleep 60
done
echo "=== TIMEOUT: still running after 45 min ==="
exit 0
