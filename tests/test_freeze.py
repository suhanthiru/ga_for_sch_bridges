import json

from sb.search.freeze import freeze, load_frozen


def test_freeze_writes_manifest_and_seeds(tmp_path):
    G, out = freeze(out_dir=tmp_path / "g", run_tests=False, n_random=20)
    man = json.loads((out / "manifest.json").read_text())
    assert man["hash"] == G.hash and (out / "seeds.jsonl").read_text().count("\n") == 20
    assert (out / "seed_set.json").exists() and (out / "disabled.json").read_text().strip() == "[]"
    G2, man2 = load_frozen(out)
    assert G2.hash == G.hash and man2["filter"] == "none"
    (out / "disabled.json").write_text(json.dumps([dict(key="planner.mppi", reason="test")]))
    G3, _ = load_frozen(out)
    assert G3.spec("planner.mppi").disabled == "test" and not any(o.key == "planner.mppi" for o in G3.options("planner"))


def test_load_frozen_ignores_later_components_and_treats_source_drift_by_strictness(tmp_path, capsys):
    from sb.core import registry as RG
    G, out = freeze(out_dir=tmp_path / "g", run_tests=False, n_random=5)
    RG.REGISTRY["zzz.later"] = RG.ComponentSpec("zzz.later", ("safety",), test="tests/x.py::test_x", cls=object)
    try:
        G2, _ = load_frozen(out)
        assert G2.hash == G.hash and "zzz.later" not in G2.registry
        man = json.loads((out / "manifest.json").read_text()); man["source"] = "0" * 16
        (out / "manifest.json").write_text(json.dumps(man))
        import pytest
        with pytest.raises(SystemExit):
            load_frozen(out, strict=True)
        G3, _ = load_frozen(out, strict=False)
        assert "warning" in capsys.readouterr().err and set(G3.registry) == set(G.registry)
        man["slots"]["extra_slot"] = ["safety", True]
        (out / "manifest.json").write_text(json.dumps(man))
        with pytest.raises(SystemExit):
            load_frozen(out, strict=False)
    finally:
        RG.REGISTRY.pop("zzz.later", None)
