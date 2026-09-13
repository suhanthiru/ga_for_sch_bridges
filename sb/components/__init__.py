"""Component library. load_all() imports every module that registers components; nothing
registers at package import so the registry only fills when the grammar asks for it."""
import importlib

MODULES = ("sb.components.core_set",)


def load_all():
    from sb.core.registry import REGISTRY
    if not REGISTRY:
        for m in MODULES:
            importlib.import_module(m)
    return REGISTRY
