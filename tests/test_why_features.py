import math

import torch

from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask
from sb.stats import why_features as W


def test_features_have_the_expected_ranges_and_ordering(cpu, small_demos):
    G, U = small_demos
    tk = GenTask(TR.Layout("L1"), "none", 16, 1, cpu)
    slip = GenTask(TR.Layout("L1"), "slip", 16, 2, cpu)
    rough = GenTask(TR.Layout("L1"), "slip", 16, 2, cpu, slip_scale=2.8)
    f = W.features(tk, slip, G)
    assert set(f) == {"w2_demo_to_target", "sigma_condition", "demo_multimodality", "pd_reachability", "shell_fraction", "kl_demo_reference"}
    assert all(math.isfinite(v) for v in f.values())
    assert 0.0 <= f["pd_reachability"] <= 1.0 and 0.0 <= f["shell_fraction"] <= 1.0
    assert f["sigma_condition"] >= 1.0 and f["demo_multimodality"] >= 1 and f["w2_demo_to_target"] >= 0.0
    # rougher terrain is harder for feedback alone, and a bimodal cloud is seen as such
    assert W.pd_reachability(rough, G) <= f["pd_reachability"]
    bim = torch.cat([G, G])
    bim[len(G):, -1, 1] += 0.4
    assert W.demo_multimodality(bim) >= 2 >= W.demo_multimodality(G)
    # demos that end on the target have a small KL against the reference aimed there
    assert f["kl_demo_reference"] >= 0.0
