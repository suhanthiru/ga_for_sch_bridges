# Errors

Append-only. One entry per failure, in the order they happened. Never edit an old entry;
if it turns out to be wrong, add a new one that says so.

Each entry: date, class (1 env bug / 2 component bug / 3 search infra / 4 resource /
5 statistical anomaly / 6 plan conflict / 7 unexpected), component, what happened,
what was done, what it invalidated.

---
