import pytest

from sb.envs import audit as A


@pytest.mark.slow
def test_audit_runs_on_one_family(cpu):
    A.BIG.update(K=200)                       # keep the comparator small on CPU
    rows = A.audit(n=16, device=cpu, ids=["E1"], log=lambda m: None)
    assert rows[0]["env"] == "E1" and rows[0]["big"] is not None and "regret" in rows[0]
    assert "| E1 |" in A.render(rows, 16)


def test_render_handles_missing_comparator():
    txt = A.render([dict(env="E4", oracle=0.97, big=None, regret=None, oracle_s=1.0, note="no sampling comparator (rule-based pusher)")], 8)
    assert "| E4 | 0.970 | - | - |" in txt
