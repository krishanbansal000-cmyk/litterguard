#!/usr/bin/env bash
# Pull the GPU kernel result the moment it completes (valid pose-on full-set run).
export PATH="$PATH:/c/Users/krish/AppData/Local/Programs/Python/Python312/Scripts"
cd /c/Users/krish/.zcode/workspace/697a15d4-0293-42c9-9dc6-78d11719a678/litter-guard
for i in $(seq 1 120); do
  sleep 60
  ST=$(kaggle kernels status krishanbansal/litterguard-gpu-eval 2>&1 | head -1)
  if echo "$ST" | grep -q COMPLETE; then
    rm -rf kaggle_out_gpu_final
    kaggle kernels output krishanbansal/litterguard-gpu-eval -p kaggle_out_gpu_final >/dev/null 2>&1
    echo "GPU RESULT:"; cat kaggle_out_gpu_final/summary.json 2>/dev/null; break
  fi
  if echo "$ST" | grep -q ERROR; then echo "GPU ERROR"; break; fi
done
