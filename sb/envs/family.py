"""The environment family index and the descriptor bins of SEARCH_PLAN section 1.

FAMILY maps an id to its class; BINS lists the levels of every descriptor axis; the
descriptor vector for the map is the concatenation of each axis's bin index scaled to
[0, 1] (an axis an environment does not own takes its default level).
"""
from sb.envs.e1_terrain import E1Terrain
from sb.envs.e2_partial import E2Partial, E7Shifted, E8Sparse
from sb.envs.e3_multimodal import E3Multimodal
from sb.envs.e4_contact import E4Contact
from sb.envs.e5_multiagent import E5Cross
from sb.envs.e6_long import E6Long

FAMILY = {"E1": E1Terrain, "E2": E2Partial, "E3": E3Multimodal, "E4": E4Contact, "E5": E5Cross,
          "E6": E6Long, "E7": E7Shifted, "E8": E8Sparse}

BINS = {
    "env": tuple(FAMILY),
    "slip_scale": (0.35, 0.7, 1.4, 2.8),
    "aniso": (1.0, 3.0, 10.0),
    "n_voronoi": (4, 14, 40),
    "push_mult": (0.0, 1.0, 3.0),
    "heading_std": (0.05, 0.2, 0.5, 1.0),
    "n_demo": (1, 5, 20, 100),
    "observability": ("full", "partial"),
    "goal_modality": (1, 2, 3),
    "horizon": (3, 6, 10),
    "terrain_info": ("none", "oracle", "estimated"),
    "dynamics": ("unicycle", "ackermann", "legged"),
    "demo_quality": ("oracle", "oracle_noisy", "scripted"),
}
DEFAULT_LEVEL = {"env": "E1", "slip_scale": 0.7, "aniso": 1.0, "n_voronoi": 14, "push_mult": 1.0, "heading_std": 0.2,
                 "n_demo": 20, "observability": "full", "goal_modality": 1, "horizon": 3, "terrain_info": "oracle",
                 "dynamics": "unicycle", "demo_quality": "oracle"}
AXES = tuple(BINS)


def level_index(axis, value):
    levels = BINS[axis]
    if isinstance(value, str):
        return levels.index(value)
    return min(range(len(levels)), key=lambda i: abs(float(levels[i]) - float(value)))


def descriptor_vector(values):
    """values: dict axis -> value (missing axes take the default). Returns a list of
    len(AXES) floats in [0, 1], each the axis's bin index over its number of bins."""
    out = []
    for ax in AXES:
        v = values.get(ax, DEFAULT_LEVEL[ax])
        n = len(BINS[ax])
        out.append(level_index(ax, v) / max(n - 1, 1))
    return out


def env_defaults(env_id):
    """Descriptor values implied by an environment id (observability, modality, horizon)."""
    d = {"env": env_id}
    if env_id == "E2":
        d["observability"] = "partial"
    if env_id == "E3":
        d["goal_modality"] = 2
    if env_id == "E6":
        d["horizon"] = 6
    if env_id == "E8":
        d["n_demo"] = 5
    return d
