#!/usr/bin/env bash
# Wait until the replication has all its shards and the GPU has been idle for three
# consecutive minutes, then run the suite. Idle: utilisation < 8% and < 1200 MiB used.
cd "$(dirname "$0")/.."
quiet=0
while true; do
  n=$(ls results/gate/seed9/*.parquet 2>/dev/null | wc -l)
  read util mem <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',' )"
  if [ "$n" -ge 26 ] && [ "$util" -lt 8 ] && [ "$mem" -lt 1200 ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%H:%M:%S) shards=$n util=$util mem=$mem quiet=$quiet"
  if [ "$quiet" -ge 3 ]; then break; fi
  sleep 60
done
bash bench/run_all.sh
