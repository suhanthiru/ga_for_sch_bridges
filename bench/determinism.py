"""Two-process bit-equality: the same (seed) FastEnv rollout with a PD controller, run in
two fresh processes under the deterministic configuration, must produce identical final
states and trajectory hashes. Also compares against the in-process eager run.

    python bench/determinism.py
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _harness import machine_info, record  # noqa: E402

CHILD = r'''
import os, hashlib, json, sys, torch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask
from sb.envs.fast import FastEnv
from sb.core import se2 as S
from sb.envs import task as TK
dev = torch.device(sys.argv[1]); n = int(sys.argv[2])
tk = GenTask(TR.Layout("L2"), "push", n, 7, dev); fe = FastEnv.from_task(tk)
gen = torch.Generator(device=dev).manual_seed(123)
g = tk.sample(0, n); traj = [g.clone()]
for s in range(300):
    k = min(s // TK.T_SKILL, 2); tau = torch.full((n,), (s % TK.T_SKILL) / TK.T_SKILL + 1.0 / TK.T_SKILL, device=dev).clamp(max=1.0)
    ref = S.SE2.interp(tk.means[k].expand(n, 3), tk.means[k + 1].expand(n, 3), tau)
    u = 6.0 * S.between(g, ref)
    g, hit = fe.step_random(g, u, s, gen); traj.append(g.clone())
T = torch.stack(traj, 1).cpu().numpy()
print(json.dumps(dict(sha=hashlib.sha256(T.tobytes()).hexdigest(), final=T[:, -1].sum().item())))
'''


def run_child(device, n):
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(HERE.parent))
    r = subprocess.run([sys.executable, "-c", CHILD, device, str(n)], capture_output=True, text=True, env=env, cwd=HERE.parent)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    import torch
    print(machine_info())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a, b = run_child(device, 4096), run_child(device, 4096)
    c = run_child("cpu", 4096)
    m = dict(device=device, n=4096, steps=300, two_process_bit_equal=(a["sha"] == b["sha"]), gpu_vs_cpu_bit_equal=(a["sha"] == c["sha"]),
             sha_a=a["sha"][:16], sha_b=b["sha"][:16], sha_cpu=c["sha"][:16])
    record("determinism", device, m)
    print(m)


if __name__ == "__main__":
    main()
