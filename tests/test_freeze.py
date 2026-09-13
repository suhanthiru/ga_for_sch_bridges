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
