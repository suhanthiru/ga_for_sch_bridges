#!/usr/bin/env bash
# Second pass on a quiet GPU: wait for the other session's chain to finish g5 (its last
# phase), for the first (contended) pass to end, and for three idle minutes; keep the
# contended results as results_shared.json and run the suite again.
cd "$(dirname "$0")/.."
LOG="D:/s_bridges/skill_chains/results_generator/chain_log.txt"
quiet=0
while true; do
  done_other=$(grep -c "generator phase g5 seed" "$LOG" 2>/dev/null)
  done_mine=$(grep -c "BENCH DONE" bench/logs/run_all.txt 2>/dev/null)
  read util mem <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',' )"
  if [ "${done_other:-0}" -ge 1 ] && [ "${done_mine:-0}" -ge 1 ] && [ "$util" -lt 8 ] && [ "$mem" -lt 3500 ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%H:%M:%S) other=$done_other mine=$done_mine util=$util mem=$mem quiet=$quiet"
  if [ "$quiet" -ge 3 ]; then break; fi
  sleep 60
done
cp bench/results.json bench/results_shared.json 2>/dev/null
mv bench/logs/run_all.txt bench/logs/run_all_shared.txt 2>/dev/null
bash bench/run_all.sh
