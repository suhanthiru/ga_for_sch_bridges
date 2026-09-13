"""Paths and device. Nothing here touches the filesystem at import time.

Override the results root with the SB_RESULTS environment variable; everything else
hangs off the repo root.
"""
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("SB_RESULTS", ROOT / "results"))
DATA = ROOT / "data"
GATE = RESULTS / "gate"


def device(override=None):
    if override is not None:
        return torch.device(override)
    env = os.environ.get("SB_DEVICE")
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def demo_path(layout, tag=""):
    return DATA / f"demos_{layout}{tag}.npz"
