import torch

from sb.envs.base import Descriptor, Env
from sb.envs.family import AXES, BINS, FAMILY, descriptor_vector, env_defaults, level_index


def test_every_family_member_implements_the_interface(cpu):
    for eid, cls in FAMILY.items():
        assert issubclass(cls, Env) and cls.id == eid
        env = cls(4, cpu) if eid != "E5" else cls(2, cpu)
        obs = env.reset(0, Descriptor(dict(disturbance="none")))
        assert obs.shape[1] == env.obs_dim()
        obs2, r, done, info = env.step(torch.zeros(obs.shape[0], 3))
        assert obs2.shape == obs.shape and r.shape[0] == obs.shape[0]


def test_descriptor_vector_is_unit_scaled_and_complete():
    v = descriptor_vector({})
    assert len(v) == len(AXES) and all(0.0 <= x <= 1.0 for x in v)
    v2 = descriptor_vector(dict(env="E8", slip_scale=2.8, n_demo=1))
    assert v2[AXES.index("slip_scale")] == 1.0 and v2[AXES.index("n_demo")] == 0.0
    assert level_index("slip_scale", 0.9) == 1 and level_index("terrain_info", "estimated") == 2
    assert env_defaults("E2")["observability"] == "partial" and env_defaults("E6")["horizon"] == 6
    assert all(len(BINS[a]) >= 2 for a in AXES)
