#!/usr/bin/env bash
export PATH="$PATH:/c/Users/krish/AppData/Local/Programs/Python/Python312/Scripts"
cd /c/Users/krish/.zcode/workspace/697a15d4-0293-42c9-9dc6-78d11719a678/litter-guard
for i in $(seq 1 480); do
  sleep 60
  for K in litterguard-gpu-eval litterguard-gpu-eval-nopose; do
    ST=$(kaggle kernels status krishanbansal/$K 2>&1 | head -1)
    if echo "$ST" | grep -q COMPLETE && [ ! -f "kaggle_out_final_$K/summary.json" ]; then
      rm -rf "kaggle_out_final_$K"
      kaggle kernels output krishanbansal/$K -p "kaggle_out_final_$K" >/dev/null 2>&1
      echo "== $K"; cat "kaggle_out_final_$K/summary.json" 2>/dev/null
    fi
  done
  if [ -f kaggle_out_final_litterguard-gpu-eval/summary.json ] && [ -f kaggle_out_final_litterguard-gpu-eval-nopose/summary.json ]; then echo BOTH_DONE; break; fi
done
