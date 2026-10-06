"""Reproduces every accuracy, McNemar test and bootstrap interval reported in the
paper from the per-question result files in results/. CPU only, no database.

    python src/paper_statistics.py
"""
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "results" / "strict_benchmark"
PD = ROOT / "results" / "pd_test"


def mcnemar(b, c):
    """Exact two-sided McNemar test on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cluster_ci(df, col, group="intent", reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {g: d[col].to_numpy() for g, d in df.groupby(group)}
    keys = list(groups)
    vals = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]).mean()
            for _ in range(reps)]
    return np.percentile(vals, [2.5, 97.5]) * 100


def compare(df, a, b):
    x = int((df[a] & ~df[b]).sum())
    y = int((~df[a] & df[b]).sum())
    return f"{a} vs {b}: +{x}/-{y}, exact McNemar p={mcnemar(x, y):.2g}"


def strict_benchmark():
    s = pd.read_csv(STRICT / "all_systems_200.csv")
    systems = ["orig_zero", "orig_few", "orig_rag", "coder_zero", "coder_few", "ft_direct", "ft_rag"]
    print("=" * 70, "\nSTRICT BENCHMARK (correct / 100, 95% intent-cluster CI)\n" + "=" * 70)
    for split in ["validation", "final_test"]:
        d = s[s.split == split]
        print(f"\n[{split}]")
        for c in systems:
            lo, hi = cluster_ci(d, c)
            print(f"  {c:12s} {int(d[c].sum()):3d}   [{lo:.0f}, {hi:.0f}]")
        for a, b in [("ft_direct", "orig_few"), ("ft_direct", "coder_few"),
                     ("ft_rag", "ft_direct"), ("orig_rag", "orig_zero")]:
            print("  " + compare(d, a, b))
    print("\n[pooled 200]  " + compare(s, "ft_direct", "orig_few"))


def pd_test():
    routes = pd.read_csv(PD / "D1_frozen_controller_routes.csv")
    d = pd.read_csv(PD / "D2_pd_finetuned_direct.csv")[["question_id", "execution_correct"]]
    r = pd.read_csv(PD / "D3_pd_finetuned_rag.csv")[["question_id", "execution_correct"]]
    m = (routes.merge(d.rename(columns={"execution_correct": "direct"}), on="question_id")
               .merge(r.rename(columns={"execution_correct": "rag"}), on="question_id"))
    m["intent"] = m["intent_id"]
    for v in ["v1", "v2a", "v2b"]:
        m[f"router_{v}"] = np.where(m[f"route_{v}"], m["rag"], m["direct"])
    m["oracle"] = m["direct"] | m["rag"]
    cols = ["direct", "rag", "router_v1", "router_v2a", "router_v2b", "oracle"]
    print("\n" + "=" * 70, "\nPD-TEST (frozen routers; 64 questions per type)\n" + "=" * 70)
    print(m.groupby("question_type")[cols].sum().T.assign(total=lambda t: t.sum(axis=1)))
    print("\nRAG calls:", {v: int(m[f"route_{v}"].sum()) for v in ["v1", "v2a", "v2b"]})
    print("Routing rate by type:\n", m.groupby("question_type")[["route_v1", "route_v2a", "route_v2b"]].mean().round(3))
    for a, b in [("rag", "direct"), ("router_v2b", "direct"), ("router_v2b", "rag"), ("router_v2a", "rag")]:
        print(compare(m, a, b))
    pol = m[m.question_type == "policy_dependent"]
    print("policy-dependent only: " + compare(pol, "rag", "direct"))
    for c in ["direct", "rag", "router_v2b"]:
        lo, hi = cluster_ci(m, c)
        print(f"CI {c}: [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    strict_benchmark()
    pd_test()
