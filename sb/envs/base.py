"""The environment interface every family in SEARCH_PLAN section 1 implements.

    env = E1Terrain(n=256, device)
    obs = env.reset(seed, descriptor)            # descriptor: dict of the axes the env owns
    obs, reward, done, info = env.step(action)   # batched over n episodes, tensor auto-reset
    ctl = env.oracle(caps)                       # a controller with privileged access; train time only
    G, U = env.demos(n_demo, caps)               # demonstrations for the data slot
    env.invariant_geometry()                     # geometry for the exploit checks, independent of step()

`caps` is the capability object handed out at train time; test-time code never holds
one, so a component cannot reach the oracle or the true map when it is being scored.
Every environment ships tests/test_env_<id>.py asserting oracle success >= 0.95
undisturbed.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Caps:
    """What train-time code may touch. Absent at test time."""
    oracle: bool = True
    true_map: bool = True
    note: str = ""


@dataclass
class Descriptor:
    """Values of the axes in SEARCH_PLAN section 1; unspecified axes take the env's default."""
    values: dict = field(default_factory=dict)

    def get(self, k, default=None):
        return self.values.get(k, default)


class Env(ABC):
    id = "E?"
    axes = ()                       # descriptor axes this environment owns

    @abstractmethod
    def reset(self, seed, descriptor):
        ...

    @abstractmethod
    def step(self, action):
        ...

    @abstractmethod
    def oracle(self, caps):
        ...

    @abstractmethod
    def demos(self, n, caps):
        ...

    @abstractmethod
    def invariant_geometry(self):
        ...

    def obs_dim(self):
        raise NotImplementedError
