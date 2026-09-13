# Oracle audit

Undisturbed, 64 episodes per environment, oracle against the same planner at K = 2000 (ten times its samples). Regret above 0.05 replaces the oracle before the search starts.

| env | oracle success | 10x planner | regret | note |
|---|---|---|---|---|
| E1 | 0.969 | 0.984 | +0.016 | ok |
| E2 | 0.969 | 0.984 | +0.016 | ok |
| E3 | 1.000 | 1.000 | +0.000 | ok |
| E4 | 0.969 | - | - | no sampling comparator (rule-based pusher) |
| E5 | 0.969 | 0.938 | -0.031 | ok |
| E6 | 0.984 | 0.969 | -0.016 | ok |
| E7 | 1.000 | 1.000 | +0.000 | ok |
| E8 | 1.000 | 1.000 | +0.000 | ok |
