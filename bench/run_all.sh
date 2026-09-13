#!/usr/bin/env bash
# Run every benchmark in sequence on a quiet GPU; each script merges into bench/results.json.
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
mkdir -p bench/logs
echo "=== bench start $(date) ===" | tee bench/logs/run_all.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tee -a bench/logs/run_all.txt
for s in gate_precision env_step sinkhorn_grid bridge_fit diffusion_train ppo_update determinism; do
  echo "--- $s $(date +%H:%M:%S) ---" | tee -a bench/logs/run_all.txt
  python bench/$s.py > bench/logs/$s.txt 2>&1; echo "exit $?" | tee -a bench/logs/run_all.txt
  tail -n 12 bench/logs/$s.txt | tee -a bench/logs/run_all.txt
done
echo "=== BENCH DONE $(date) ===" | tee -a bench/logs/run_all.txt
