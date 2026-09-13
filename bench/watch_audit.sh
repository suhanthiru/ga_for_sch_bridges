#!/usr/bin/env bash
# After the clean bench pass (run_all_shared.txt exists and the new run_all.txt says done),
# run the oracle audit on the quiet GPU.
cd "$(dirname "$0")/.."
while true; do
  if [ -f bench/logs/run_all_shared.txt ] && grep -q "BENCH DONE" bench/logs/run_all.txt 2>/dev/null; then break; fi
  sleep 60
done
echo "=== audit start $(date) ===" > bench/logs/audit.txt
python -m sb.cli audit --n 64 >> bench/logs/audit.txt 2>&1
echo "=== AUDIT DONE $(date) ===" >> bench/logs/audit.txt
