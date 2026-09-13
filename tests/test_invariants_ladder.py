import torch

from sb.envs import terrain as TR
from sb.envs.base import Caps, Descriptor
from sb.envs.e1_terrain import E1Terrain
from sb.search.elites import EliteMap
from sb.search.invariants import check_episode_set, first_goal_step, segment_hits, tripped
from sb.search.ladder import HOLD, after_rung0, due_for_validation, is_jump


def test_segment_hits_are_geometric():
    geo = dict(wall_x=0.5, gaps=[(0.5, 0.22)], pile=(0.7, 0.66, 0.05))
    a = torch.tensor([[0.45, 0.5], [0.45, 0.2], [0.65, 0.66], [0.5, 0.5]]); b = torch.tensor([[0.55, 0.5], [0.55, 0.2], [0.69, 0.66], [1.01, 0.5]])
    h = segment_hits(a, b, geo)
    assert h.tolist() == [False, True, True, True]


def test_honest_oracle_trips_nothing_and_a_teleport_trips(cpu):
    env = E1Terrain(16, cpu); env.reset(0, Descriptor(dict(layout="L2", disturbance="none")))
    out = env.tk.rollout(env.oracle(Caps()), n=16, record=True)
    T = out["traj"]; geo = env.invariant_geometry()
    acts = torch.zeros(16, T.shape[1] - 1, 3)
    inv = check_episode_set(T, acts, out["success"], geo)
    assert tripped(inv) == [], inv
    T2 = T.clone(); T2[0, 50:, :2] = torch.tensor([0.9, 0.5])                      # teleport through the wall between the gaps
    inv2 = check_episode_set(T2, acts, out["success"], geo)
    assert "wall_penetration" in tripped(inv2)
    acts2 = acts.clone(); acts2[1, 3, 0] = 5.0
    assert "action_bounds" in tripped(check_episode_set(T, acts2, out["success"], geo))
    fg = first_goal_step(T, geo)
    assert fg.shape == (16,) and int(fg.max()) <= T.shape[1] - 1


def test_promotion_and_validation_rules():
    assert after_rung0(0.5, None).rung == 1
    assert after_rung0(0.45, 0.5).rung == 1 and after_rung0(0.3, 0.5).rung == 0
    assert is_jump(0.7, 0.5) and not is_jump(0.6, 0.5)
    m = EliteMap([[0.0], [1.0]])
    m.insert(0, "g0", 0.5, "e0", gen=0); m.insert(1, "g1", 0.5, "e1", gen=HOLD)
    assert due_for_validation(m, HOLD) == [0] and sorted(due_for_validation(m, 0, final=True)) == [0, 1]
