"""Component registry: what a component declares and how the grammar reads it.

    @component(key="controller.pd", slots=("controller",), params={"kp": P.loguniform(1, 30)},
               tag="none", test="tests/components/test_controllers.py::test_pd")
    class PD: ...

A component with no test is not in the grammar (freeze refuses it). `disabled` is the
only post-freeze change and only ever shrinks the grammar.
"""
import math
from dataclasses import dataclass, field

import numpy as np

REGISTRY = {}


@dataclass(frozen=True)
class ParamSpec:
    kind: str                          # log | lin | int | choice
    lo: float = 0.0
    hi: float = 1.0
    choices: tuple = ()
    sigma: float = 0.3                 # mutation kernel: log10 units (log), fraction of range (lin/int)
    p_flip: float = 0.2                # choice: probability a mutation re-draws

    def sample(self, rng):
        if self.kind == "log":
            return float(10 ** rng.uniform(math.log10(self.lo), math.log10(self.hi)))
        if self.kind == "lin":
            return float(rng.uniform(self.lo, self.hi))
        if self.kind == "int":
            return int(rng.integers(int(self.lo), int(self.hi) + 1))
        return self.choices[int(rng.integers(len(self.choices)))]

    def mutate(self, v, rng):
        if self.kind == "log":
            return float(self.clip(10 ** (math.log10(v) + rng.normal(0, self.sigma))))
        if self.kind == "lin":
            return float(self.clip(v + rng.normal(0, self.sigma * (self.hi - self.lo))))
        if self.kind == "int":
            return int(self.clip(round(v + rng.normal(0, max(1.0, self.sigma * (self.hi - self.lo))))))
        return self.sample(rng) if rng.random() < self.p_flip else v

    def clip(self, v):
        if self.kind == "choice":
            return v if v in self.choices else self.choices[0]
        return min(max(v, self.lo), self.hi)

    def valid(self, v):
        if self.kind == "choice":
            return v in self.choices
        if self.kind == "int":
            return isinstance(v, (int, np.integer)) and self.lo <= v <= self.hi
        return isinstance(v, (int, float, np.floating)) and self.lo <= v <= self.hi


class P:
    @staticmethod
    def loguniform(lo, hi, sigma=0.3):
        return ParamSpec("log", lo, hi, sigma=sigma)

    @staticmethod
    def uniform(lo, hi, sigma=0.1):
        return ParamSpec("lin", lo, hi, sigma=sigma)

    @staticmethod
    def int_uniform(lo, hi, sigma=0.2):
        return ParamSpec("int", lo, hi, sigma=sigma)

    @staticmethod
    def choice(choices, p_flip=0.2):
        return ParamSpec("choice", choices=tuple(choices), p_flip=p_flip)


@dataclass(frozen=True)
class SlotSpec:
    type: str
    optional: bool = True


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    slots: tuple                       # slot types this component can fill
    sub_slots: dict = field(default_factory=dict)     # name -> SlotSpec it exposes
    params: dict = field(default_factory=dict)        # name -> ParamSpec
    tag: str = "none"                  # bridge | rl | none
    oracle: str = "none"               # none | train_data | train_value
    cost: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)          # bridges: reference, coupling, marginal, representation, family
    test: str = ""
    disabled: str = ""
    cls: object = None

    def manifest(self):
        return dict(key=self.key, slots=list(self.slots), sub_slots={k: [v.type, v.optional] for k, v in self.sub_slots.items()},
                    params={k: [p.kind, p.lo, p.hi, list(p.choices)] for k, p in self.params.items()}, tag=self.tag,
                    oracle=self.oracle, axes=dict(self.axes), test=self.test, disabled=self.disabled)


def component(key, slots, sub_slots=None, params=None, tag="none", oracle="none", cost=None, axes=None, test="", disabled=""):
    def deco(cls):
        if key in REGISTRY:
            raise KeyError(f"component {key} registered twice")
        REGISTRY[key] = ComponentSpec(key, tuple(slots), dict(sub_slots or {}), dict(params or {}), tag, oracle,
                                      dict(cost or {}), dict(axes or {}), test, disabled, cls)
        cls.spec = REGISTRY[key]
        return cls
    return deco


def register(spec):
    """Register a spec built by hand (tests, dummy components)."""
    if spec.key in REGISTRY:
        raise KeyError(f"component {spec.key} registered twice")
    REGISTRY[spec.key] = spec
    return spec


def clear():
    REGISTRY.clear()
