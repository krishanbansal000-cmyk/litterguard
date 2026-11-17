#!/usr/bin/env bash
# Autonomous watcher: pull GPU kernel results when they complete; compute the
# local full-set score when all 100 videos are done.
export PATH="$PATH:/c/Users/krish/AppData/Local/Programs/Python/Python312/Scripts"
cd /c/Users/krish/.zcode/workspace/697a15d4-0293-42c9-9dc6-78d11719a678/litter-guard
PY="/c/Users/krish/AppData/Local/Programs/Python/Python312/python.exe"

for i in $(seq 1 200); do
  sleep 60
  for K in litterguard-gpu-eval litterguard-gpu-eval-nopose; do
    ST=$(kaggle kernels status krishanbansal/$K 2>&1 | head -1)
    if echo "$ST" | grep -q COMPLETE; then
      if [ ! -f "kaggle_out_$K/summary.json" ]; then
        kaggle kernels output krishanbansal/$K -p "kaggle_out_$K" >/dev/null 2>&1
        echo "PULLED $K:"; cat "kaggle_out_$K/summary.json" 2>/dev/null
      fi
    fi
  done
  N=$("$PY" -c "
import json, glob
rows=[]
for p in glob.glob('runs/mivia_local/results_f*.json'):
    rows.extend(json.load(open(p)))
print(len({r['video'] for r in rows}))" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] local=$N/100"
  if [ "$N" -ge 100 ] && [ ! -f runs/mivia_final/FINAL_summary.json ]; then
    "$PY" - <<'PYEOF'
import json, glob
rows=[]
for p in glob.glob('runs/mivia_local/results_f*.json'):
    rows.extend(json.load(open(p)))
seen=set(); u=[]
for r in rows:
    if r['video'] not in seen: seen.add(r['video']); u.append(r)
tp=fp=fn=tn=0; atp=afp=0; by={}; lit={}
for r in u:
    if r['label'] is None: continue
    det=r['events']>0; al=(r.get('alerts') or 0)>0
    et=(r.get('meta') or {}).get('event_type') or '?'
    d=by.setdefault(et,{'tp':0,'fp':0,'fn':0,'tn':0})
    L=(r.get('meta') or {}).get('lighting') or '?'
    dd=lit.setdefault(L,{'tp':0,'fp':0,'fn':0,'tn':0})
    if r['label'] and det: tp+=1; d['tp']+=1; dd['tp']+=1
    elif r['label'] and not det: fn+=1; d['fn']+=1; dd['fn']+=1
    elif not r['label'] and det: fp+=1; d['fp']+=1; dd['fp']+=1
    else: tn+=1; d['tn']+=1; dd['tn']+=1
    if r['label'] and al: atp+=1
    elif not r['label'] and al: afp+=1
n=len([r for r in u if r['label'] is not None])
out={"n":n,"detections":{"tp":tp,"fp":fp,"fn":fn,"tn":tn,
     "precision":round(tp/max(tp+fp,1),4),"recall":round(tp/max(tp+fn,1),4),
     "accuracy":round((tp+tn)/max(n,1),4)},
     "alerts":{"tp":atp,"fp":afp,"precision":round(atp/max(atp+afp,1),4)},
     "by_event_type":by,"by_lighting":lit}
json.dump(out, open('runs/mivia_final/FINAL_summary.json','w'), indent=1)
print(json.dumps(out, indent=1))
PYEOF
  fi
  if [ -f runs/mivia_final/FINAL_summary.json ] && [ -f kaggle_out_litterguard-gpu-eval/summary.json ]; then
    echo "BOTH_DONE"; break
  fi
done
