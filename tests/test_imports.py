"""Importing any sb module must not touch the filesystem."""
import importlib
import os
import pkgutil
import subprocess
import sys

import sb


def test_imports_have_no_side_effects(tmp_path):
    mods = [m.name for m in pkgutil.walk_packages(sb.__path__, "sb.")]
    code = "import importlib,sys\n" + "\n".join(f"importlib.import_module({m!r})" for m in mods)
    before = set(os.listdir(tmp_path))
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(sb.__file__)))
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert set(os.listdir(tmp_path)) == before
    for m in mods:
        importlib.import_module(m)
