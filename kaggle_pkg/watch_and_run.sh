#!/usr/bin/env bash
# Push the LitterGuard GPU kernel when a GPU slot frees; track; pull output.
export PATH="$PATH:/c/Users/krish/AppData/Local/Programs/Python/Python312/Scripts"
cd /c/Users/krish/.zcode/workspace/697a15d4-0293-42c9-9dc6-78d11719a678/litter-guard
mkdir -p kaggle_out

echo "waiting for a free GPU session slot..."
PUSHED=0
for i in $(seq 1 40); do
  OUT=$(kaggle kernels push -p kaggle_pkg 2>&1)
  if echo "$OUT" | grep -q "Maximum batch GPU"; then
    echo "slot busy ($i): $(echo "$OUT" | tail -1 | cut -c1-80)"
    sleep 180
  elif echo "$OUT" | grep -qi "error\|409\|400"; then
    echo "push error: $OUT" | head -3; sleep 120
  else
    echo "PUSHED at $(date +%H:%M:%S)"; PUSHED=1; break
  fi
done
[ "$PUSHED" -eq 0 ] && { echo "never got a slot"; exit 1; }

# track to completion (kernel auto-runs after push)
for i in $(seq 1 120); do
  sleep 60
  ST=$(kaggle kernels status krishanbansal/litterguard-gpu-eval 2>/dev/null | head -1)
  echo "[$i min] $ST"
  case "$ST" in
    *COMPLETE*|*CANCEL*|*ERROR*) break ;;
  esac
done

kaggle kernels output krishanbansal/litterguard-gpu-eval -p kaggle_out 2>&1 | tail -2
echo "=== summary.json ==="
cat kaggle_out/summary.json 2>/dev/null || echo "(no summary - check kaggle_out/ log)"
