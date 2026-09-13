"""The pilot report (SEARCH_PLAN 0.4): every registered target against what bench/
measured, the ratio, and the mechanical decision (pass / prune at < 50 %), plus the
numbers recorded without a target. Reads bench/results.json (and results_shared.json for
the contended pass, reported separately) and writes findings/pilot.md.

    python bench/report.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGETS = {"env_steps_per_s": 50_000, "grid_bridges_per_s": 200, "neural_bridges_per_s": 200, "ppo_genome_updates_per_s": 128}


def best(runs, bench, key, prefer=None):
    rows = runs.get(bench, {})
    if prefer:
        rows = {k: v for k, v in rows.items() if prefer(k, v)}
    vals = [(v.get(key), k) for k, v in rows.items() if v.get(key) is not None]
    return max(vals) if vals else (None, None)


def summarise(data):
    out = []
    for machine, blob in data.items():
        runs = blob["runs"]; info = blob["machine"]
        shared = info.get("shared_gpu")
        r = dict(machine=machine, shared_gpu=shared, git=info.get("git"), measured=info.get("time"))
        v, k = best(runs, "env_step", "steps_per_s", lambda k, v: k.startswith("cudagraph")); r["env_steps_per_s"] = v; r["env_cfg"] = k
        v, k = best(runs, "sinkhorn_grid", "bridges_per_s", lambda k, v: k.startswith("float32")); r["grid_bridges_per_s"] = v; r["grid_cfg"] = k
        v, k = best(runs, "bridge_fit", "bridges_per_s_at_cfg", lambda k, v: k.startswith("P")); r["neural_bridges_per_s"] = v; r["neural_cfg"] = k
        v, k = best(runs, "ppo_update", "genome_updates_per_s"); r["ppo_genome_updates_per_s"] = v; r["ppo_cfg"] = k
        r["gate_fp32_pass"] = runs.get("gate_precision", {}).get("fp32", {}).get("passes")
        r["gate_bf16_err"] = runs.get("gate_precision", {}).get("bf16", {}).get("max_abs_err")
        r["determinism"] = {k: v.get("two_process_bit_equal") for k, v in runs.get("determinism", {}).items()}
        r["diffusion_s_per_8000"] = runs.get("diffusion_train", {}).get("single_b1024", {}).get("s_per_8000_steps")
        r["env_bit_equal"] = {k: v.get("repeat_bit_equal") for k, v in runs.get("env_step", {}).items() if k.startswith("cudagraph")}
        out.append(r)
    return out


def decision(r):
    rows = []
    for key, tgt in TARGETS.items():
        v = r.get(key)
        ratio = (v / tgt) if v is not None else None
        verdict = "not measured" if v is None else ("pass" if ratio >= 1 else ("below target" if ratio >= 0.5 else "prune (< 50 %)"))
        rows.append((key, tgt, v, ratio, verdict))
    return rows


def render(summaries, title):
    lines = [f"# {title}\n"]
    for r in summaries:
        tag = "shared GPU (another CUDA process was active; provisional)" if r["shared_gpu"] else "quiet GPU"
        lines.append(f"## {r['machine']} — {tag}\n\nmeasured {r['measured']}, commit {r['git']}\n")
        lines.append("| target | registered | measured | ratio | decision |\n|---|---|---|---|---|")
        for key, tgt, v, ratio, verdict in decision(r):
            lines.append(f"| {key} | {tgt:,} | {'-' if v is None else f'{v:,.1f}'} | {'-' if ratio is None else f'{ratio:.2f}x'} | {verdict} |")
        lines.append("")
        lines.append(f"Configurations: env {r['env_cfg']}, grid {r['grid_cfg']}, neural {r['neural_cfg']}, ppo {r['ppo_cfg']}.\n")
        lines.append(f"Gate at fp32: {'pass' if r['gate_fp32_pass'] else 'FAIL'}; bf16 max abs err {r['gate_bf16_err']}. "
                     f"Determinism (two processes): {r['determinism']}. Graph replay bit-equal on repeat: {r['env_bit_equal']}. "
                     f"Diffusion: {r['diffusion_s_per_8000'] and round(r['diffusion_s_per_8000'], 1)} s per 8000 steps"
                     + (f" (graph-captured: {round(r['diffusion_graphed_s_per_8000'], 1)} s)" if r.get("diffusion_graphed_s_per_8000") else "") + ".\n")
    return "\n".join(lines)


def main():
    parts = []
    for name, title in (("results.json", "Pilot throughput report"), ("results_shared.json", "Contended pass (kept for the record)")):
        p = HERE / name
        if p.exists():
            parts.append(render(summarise(json.loads(p.read_text())), title))
    if not parts:
        sys.exit("no bench results yet")
    txt = "\n\n".join(parts) + "\nPruning follows SEARCH_PLAN 0.4: any target below 50 % of its value prunes the component that depends on it; the decision goes to PLAN_CHANGES.md.\n"
    out = ROOT / "findings" / "pilot.md"; out.parent.mkdir(exist_ok=True); out.write_text(txt, encoding="utf-8", newline="\n")
    print(txt)


if __name__ == "__main__":
    main()
