"""Shared timing harness for the throughput pilot.

timeit() synchronises the device around every call and reports the median and the
10th/90th percentiles; record() merges one result into bench/results.json under the
machine it was measured on, so numbers from different boxes never get mixed up.
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"


def machine_info():
    info = dict(host=platform.node(), os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                cpu_count=torch.get_num_threads(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        info["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE.parent, capture_output=True, text=True).stdout.strip()
    except Exception:
        info["git"] = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0), gpu_mem_total_mb=total // 2 ** 20, gpu_mem_free_at_start_mb=free // 2 ** 20)
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    try:                                   # other work on the GPU at start: the plan labels such runs shared_gpu
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        info["gpu_util_at_start"] = int(q[0]); info["gpu_mem_used_at_start_mb"] = int(q[1])
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        info["gpu_compute_apps_at_start"] = len([l for l in apps.splitlines() if l.strip()])
        info["shared_gpu"] = info["gpu_util_at_start"] > 8
    except Exception:
        info["shared_gpu"] = None
    return info


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, warmup=3, iters=10):
    """Median / p10 / p90 seconds per call, with the device synchronised around each call."""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts)
    return dict(median=float(np.median(ts)), p10=float(np.percentile(ts, 10)), p90=float(np.percentile(ts, 90)), iters=iters)


def peak_mb():
    return torch.cuda.max_memory_allocated() // 2 ** 20 if torch.cuda.is_available() else 0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def record(bench, config, metrics):
    """Merge one measurement into results.json: results[machine][bench][config] = metrics."""
    info = machine_info()
    key = f"{info['host']}|{info.get('gpu', 'cpu')}"
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    m = data.setdefault(key, dict(machine=info, runs={}))
    m["machine"] = info
    m["runs"].setdefault(bench, {})[config] = dict(metrics, measured=info["time"], git=info["git"])
    RESULTS.write_text(json.dumps(data, indent=1) + "\n", newline="\n")
    return key
