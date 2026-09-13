"""The linear-Gaussian covariance-steering gate at fp64, fp32, bf16 and fp16: max abs
error of one IMF step against the closed-form coupling (tolerance 1e-3). The gate must
pass in fp32; the bf16 number is recorded to show why the solver never runs in bf16.

    python bench/gate_precision.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import machine_info, record  # noqa: E402

from sb.core import sde as SD  # noqa: E402

S0 = np.array([[0.040, 0.012], [0.012, 0.015]])
S1 = np.array([[0.008, -0.004], [-0.004, 0.030]])
EPS = 0.05


def imf_step_torch(S0, S1, C, eps, dtype, device, n_t=4000, t_end=1 - 1e-5):
    S0, S1, C = (torch.tensor(a, dtype=dtype, device=device) for a in (S0, S1, C))
    I = torch.eye(2, dtype=dtype, device=device)

    def A(t):
        St = (1 - t) ** 2 * S0 + t ** 2 * S1 + t * (1 - t) * (C + C.T) + eps * t * (1 - t) * I
        K = ((1 - t) * C.T + t * S1) @ torch.linalg.inv(St.float()).to(dtype)
        return (K - I) / (1 - t)

    J = S0.clone(); ts = torch.linspace(0, t_end, n_t + 1, dtype=torch.float64).tolist()
    for i in range(n_t):
        t, h = ts[i], ts[i + 1] - ts[i]
        k1 = J @ A(t).T; k2 = (J + h / 2 * k1) @ A(t + h / 2).T
        k3 = (J + h / 2 * k2) @ A(t + h / 2).T; k4 = (J + h * k3) @ A(t + h).T
        J = J + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return J.double().cpu().numpy()


def main():
    print(machine_info())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = SD.gaussian_sb_coupling(S0, S1, EPS)
    for name, dtype in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        err = float(np.abs(imf_step_torch(S0, S1, C, EPS, dtype, device) - C).max())
        m = dict(dtype=name, max_abs_err=err, tol=1e-3, passes=bool(err < 1e-3))
        record("gate_precision", name, m)
        print(f"{name}: max abs err {err:.2e}  {'pass' if m['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
