# Incidents

Things that did not fit the error classes. One paragraph each: what was seen, three
hypotheses, the cheapest test that separates them, and what the test said.

---

## 2026-09-14 17:36 — pilot tranche 1 stopped at evaluation 994 without an error

What I saw: the search process ended between the rung-0 line of evaluation 993 and its
rung-1 follow-up. No traceback, no CUDA error and no Windows application-error event; the
shell wrapper reported exit 127, which is not an exit code Python produces. The machine had
15.1 GB of 31.9 GB free, had not rebooted (uptime 5 days), and the GPU was idle afterwards.
Because `run()` performs its final validation pass and writes `map.parquet` after the
evaluation loop, neither happened: the last checkpoint is ckpt-0002460 at 900 evaluations,
with rows to 993 in the shards.

My first reading was that every python process on the machine had died together, which
would have implicated something machine-wide. That was wrong and is corrected here: the
other sessions' jobs had ended earlier on their own (skill_chains' chain_log.txt last wrote
at 22:13 the previous evening), so this process died alone.

Three hypotheses:
1. The run was reaped with its launcher. It was started as a background task of the agent
   harness; the wrapper's exit 127 and the absence of any Python-side evidence fit a child
   killed when its parent task was cleaned up.
2. A silent native abort inside the process (a CUDA-graph pool or a worker pool tearing
   down badly), which would print nothing to a redirected stdout.
3. An external kill from outside the session (a policy, a cleanup script, a user action).

Cheapest distinguishing test: relaunch the same command detached from the harness, with
PowerShell Start-Process, and let it run past the point it reached before. If it survives,
hypothesis 1 stands and the operational rule is that multi-hour runs are launched detached.
If it dies again near the same elapsed time or evaluation count, hypothesis 2 is live and
the next step is a faulthandler dump and running the RL path without graph capture.

Action: resume from the 900-evaluation checkpoint, detached, to finish the tranche and get
the final validation pass. Nothing is lost - the archive is append-only and the rows to 993
are on disk.
