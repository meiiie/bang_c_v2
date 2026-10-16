#!/usr/bin/env python3
"""Per-question latency/cost/token stats for one or more questions.jsonl logs."""
import json, sys, statistics
def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)
for path in sys.argv[1:]:
    R = [json.loads(l) for l in open(path)]
    s = [r['seconds'] for r in R]; c = [r['cost'] for r in R]
    t = [sum(sum((a.get('usage') or {}).get(k) or 0 for k in ('prompt_tokens', 'completion_tokens')) for a in r['attempts']) for r in R]
    print(f"{path}: n={len(R)} sec p50={pct(s,.5):.1f} p90={pct(s,.9):.1f} max={max(s):.1f} | cost/q mean={statistics.mean(c):.5f} max={max(c):.5f} | tok/q={statistics.mean(t):.0f} | errors={sum(bool(r['error']) for r in R)}")
