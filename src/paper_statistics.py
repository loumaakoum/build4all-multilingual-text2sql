"""Recomputes the accuracies, tests, intervals and tables reported in the paper
from the per-question files in results/. CPU only, no database needed
(strict/tolerant scores come from results/strict_benchmark/rescored_all_systems.csv,
which src/rescore.py produces from the stored SQL and the restored database).

    python src/paper_statistics.py

Not covered: training-run values (loss, steps, runtime), which come from the
training logs printed in notebooks/1_main_experiment.ipynb.

Statistical units. Each benchmark intent appears in four languages, so the four
questions of an intent are not independent. All tests below treat the INTENT as
the cluster:
  * Durkalski et al. (2003) cluster-adjusted McNemar statistic,
  * Obuchowski (1998) test for correlated proportions in clustered data,
  * an exact intent-level sign-flip permutation test (two-sided).
A family-level permutation test (intents of one template family are also related)
and the question-level exact McNemar test are printed for reference.
"""
from collections import Counter
from math import comb, erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "results" / "strict_benchmark"
PD = ROOT / "results" / "pd_test"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ----------------------------------------------------------------- tests
def chi2_1df_p(x):
    """Upper tail of chi-square with 1 df."""
    return 1 - erf(sqrt(x / 2)) if x > 0 else 1.0


def mcnemar_exact(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


# The two cluster-adjusted tests follow the implementations in the R package
# clust.bin.pair (Gopstein et al.), which uses Yang et al. (2010)'s notation:
#   Durkalski:  (sum_k (b_k-c_k)/n_k)^2 / sum_k ((b_k-c_k)/n_k)^2
#   Obuchowski: ((K-1)/K) * (sum_k (b_k-c_k))^2 / sum_k (b_k-c_k)^2
# where b_k, c_k are the discordant counts in cluster k, n_k its size, K the number of clusters.
def _cluster_counts(a, b, cluster):
    g = pd.DataFrame({"b": (a & ~b).astype(int).values, "c": (~a & b).astype(int).values,
                      "k": np.asarray(cluster)})
    return g.groupby("k").agg(b=("b", "sum"), c=("c", "sum"), n=("b", "size"))


def durkalski(a, b, cluster):
    s = _cluster_counts(a, b, cluster)
    w = (s.b - s.c) / s.n
    den = (w ** 2).sum()
    return 1.0 if den == 0 else chi2_1df_p(w.sum() ** 2 / den)


def obuchowski(a, b, cluster):
    s = _cluster_counts(a, b, cluster)
    K, d = len(s), s.b - s.c
    den = (d ** 2).sum()
    return 1.0 if den == 0 else chi2_1df_p((K - 1) / K * d.sum() ** 2 / den)


def permutation(a, b, cluster):
    """Exact two-sided sign-flip test on per-intent differences."""
    d = pd.Series(a.astype(int) - b.astype(int)).groupby(np.asarray(cluster)).sum()
    d = [int(v) for v in d if v != 0]
    if not d:
        return 1.0
    dist = Counter({0: 1})
    for v in d:
        nxt = Counter()
        for s, c in dist.items():
            nxt[s + v] += c
            nxt[s - v] += c
        dist = nxt
    obs = abs(sum(d))
    return sum(c for s, c in dist.items() if abs(s) >= obs) / 2 ** len(d)


def compare(df, a, b, label=""):
    A, B = df[a].astype(bool), df[b].astype(bool)
    x, y, p_q = mcnemar_exact(A, B)
    disc = df[A != B]
    return {
        "comparison": f"{a} vs {b}" + (f" [{label}]" if label else ""),
        "A_correct": int(A.sum()), "B_correct": int(B.sum()), "n": len(df),
        "A_only": x, "B_only": y,
        "intents_total": df.intent.nunique(),
        "intents_discordant": disc.intent.nunique(),
        "families_discordant": disc.family.nunique() if "family" in df else None,
        "p_durkalski": durkalski(A, B, df.intent),
        "p_obuchowski": obuchowski(A, B, df.intent),
        "p_intent_permutation": permutation(A, B, df.intent),
        "p_family_permutation": permutation(A, B, df.family),
        "p_mcnemar_question_level": p_q,
    }


def cluster_ci(df, col, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {g: d[col].astype(float).to_numpy() for g, d in df.groupby("intent")}
    keys = list(groups)
    vals = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]).mean() for _ in range(reps)]
    return np.percentile(vals, [2.5, 97.5]) * 100


def show(rows, title):
    print("\n" + title)
    t = pd.DataFrame(rows)
    for c in [c for c in t if c.startswith("p_")]:
        t[c] = t[c].map(lambda v: f"{v:.2g}")
    print(t.to_string(index=False))


# ------------------------------------------------------- strict benchmark
SYSTEMS = ["orig_zero", "orig_few", "orig_zero_conv", "orig_few_conv", "orig_rag", "coder_zero", "coder_few",
           "ft_direct", "ft_rag"]


def strict_benchmark():
    r = pd.read_csv(STRICT / "rescored_all_systems.csv")
    rec = pd.read_csv(STRICT / "all_systems_200.csv").set_index("question_id")
    assert all(rec.loc[q, s] == v for q, s, v in zip(r.question_id, r.system, r.strict)), \
        "rescored strict outcomes differ from the recorded ones"
    w = r.pivot_table(index=["question_id", "split", "intent_id", "template_family", "language"],
                      columns="system", values=["strict", "tolerant"]).reset_index()
    w.columns = ["_".join(c for c in col if c) for col in w.columns]
    w = w.rename(columns={"intent_id": "intent", "template_family": "family"})

    print("=" * 78, "\nSTRICT BENCHMARK  (correct / 100; 95% intent-cluster bootstrap CI)\n" + "=" * 78)
    rows = []
    for split in ["validation", "final_test"]:
        d = w[w.split == split]
        for s in SYSTEMS:
            for k in ["strict", "tolerant"]:
                lo, hi = cluster_ci(d, f"{k}_{s}")
                rows.append({"split": split, "system": s, "metric": k,
                             "correct": int(d[f"{k}_{s}"].sum()), "CI": f"[{lo:.0f}, {hi:.0f}]"})
    t = pd.DataFrame(rows)
    print(t.pivot_table(index="system", columns=["split", "metric"], values="correct", aggfunc="first")
          .reindex(SYSTEMS))
    print(t.pivot_table(index="system", columns=["split", "metric"], values="CI", aggfunc="first")
          .reindex(SYSTEMS))
    print("\nIntents / families per split:",
          w.groupby("split")[["intent", "family"]].nunique().to_dict("index"))

    print("\nStrict failures by type (rounding_only = differs only by ROUND(...,2);"
          " column_only = extra/missing column):")
    print(pd.crosstab([r.system, r.split], r.error_type).reindex(SYSTEMS, level=0))

    print("\nBy language (validation + locked test, out of 50):")
    lang = w.groupby("language")[[f"{k}_{s}" for k in ["strict", "tolerant"]
                                  for s in ["orig_zero", "orig_few", "orig_zero_conv", "orig_few_conv", "coder_few",
                                            "ft_direct", "ft_rag"]]].sum()
    print(lang.astype(int).T)

    comps = []
    for split, d in [("validation", w[w.split == "validation"]), ("final_test", w[w.split == "final_test"]),
                     ("pooled", w)]:
        for k in ["strict", "tolerant"]:
            for a, b in [("ft_direct", "orig_few"), ("ft_direct", "orig_zero"), ("ft_direct", "orig_few_conv"),
                         ("ft_direct", "orig_zero_conv"), ("ft_direct", "coder_few"), ("ft_rag", "ft_direct")]:
                c = compare(d, f"{k}_{a}", f"{k}_{b}", split)
                comps.append(c)
    show(comps, "Paired comparisons (cluster = intent):")

    az = w[w.language == "lebanese_arabizi"]
    show([compare(az, f"{k}_ft_direct", f"{k}_{b}", "Arabizi, pooled")
          for k in ["strict", "tolerant"] for b in ["orig_zero", "orig_few", "orig_zero_conv", "orig_few_conv"]],
         "Lebanese Arabizi only (one question per intent, so the intent-level tests reduce to"
         " question level; the family-level permutation is the conservative check):")

    nt = w[w.family != "events_top_products_by_type"]
    print("\nWithout the top-products family (strict):",
          {s: f"{int(nt[f'strict_{s}'].sum())}/{len(nt)}" for s in ["orig_zero_conv", "orig_few_conv", "ft_direct"]},
          "| by split:", nt.groupby("split")[["strict_orig_few_conv", "strict_ft_direct"]].sum().astype(int).to_dict())
    show([compare(nt, "strict_ft_direct", "strict_orig_few_conv", "pooled, without top-products family")],
         "Fine-tuned vs few-shot +conventions without the top-products family:")
    d = w[w.split == "final_test"]
    disc = d[d.strict_ft_direct.astype(bool) != d.strict_orig_few.astype(bool)]
    print("\nLocked test, fine-tuned vs few-shot discordant pairs by family:",
          disc.family.value_counts().to_dict())

    print("\nWhat retrieval changed for the fine-tuned model (strict):")
    rr = r[r.system.isin(["ft_direct", "ft_rag"])].pivot_table(
        index=["question_id", "split", "template_family", "language"], columns="system",
        values=["strict", "error_type"], aggfunc="first").reset_index()
    rr.columns = ["_".join(c for c in col if c) for col in rr.columns]
    ch = rr[rr.strict_ft_direct != rr.strict_ft_rag].copy()
    ch["change"] = np.where(ch.strict_ft_rag.astype(bool), "correction", "regression")
    ch["failure_type"] = np.where(ch.change == "correction", ch.error_type_ft_direct, ch.error_type_ft_rag)
    print(ch.groupby(["split", "change", "template_family", "failure_type"]).size().to_string())
    print(ch.groupby(["split", "change", "language"]).size().unstack(fill_value=0))
    return w


# ---------------------------------------------------------------- PD-Test
def pd_test():
    routes = pd.read_csv(PD / "D1_frozen_controller_routes.csv")
    d = pd.read_csv(PD / "D2_pd_finetuned_direct.csv")[["question_id", "execution_correct"]]
    r = pd.read_csv(PD / "D3_pd_finetuned_rag.csv")[["question_id", "execution_correct"]]
    m = (routes.merge(d.rename(columns={"execution_correct": "direct"}), on="question_id")
               .merge(r.rename(columns={"execution_correct": "rag"}), on="question_id"))
    m["intent"], m["family"] = m["intent_id"], m["template_family"]
    for v in ["v1", "v2a", "v2b"]:
        m[f"router_{v}"] = np.where(m[f"route_{v}"], m["rag"], m["direct"])
    m["oracle"] = m["direct"] | m["rag"]
    cols = ["direct", "rag", "router_v1", "router_v2a", "router_v2b", "oracle"]
    print("\n" + "=" * 78, "\nPD-TEST (routers frozen before generation; 64 questions per type)\n" + "=" * 78)
    print("Intents:", m.groupby("question_type").intent.nunique().to_dict(),
          "| families:", m.groupby("question_type").family.nunique().to_dict())
    print(m.groupby("question_type")[cols].sum().T.assign(total=lambda t: t.sum(axis=1)))
    for c in ["direct", "rag", "router_v2b"]:
        lo, hi = cluster_ci(m, c)
        print(f"CI {c}: [{lo:.1f}, {hi:.1f}]")

    calls = {v: int(m[f"route_{v}"].sum()) for v in ["v1", "v2a", "v2b"]}
    print("\nRAG calls:", calls)
    print("\nRouting rate to RAG by question type and language:")
    print(m.pivot_table(index=["question_type", "language"], values=["route_v1", "route_v2a", "route_v2b"],
                        aggfunc="sum").astype(int).to_string(), "\n(out of 16 per cell)")

    best = max(m.direct.sum(), m.rag.sum())
    print("\nPre-set criteria: (i) within 2 questions of the better of direct/RAG with fewer RAG calls;"
          " (ii) policy questions routed to RAG at a clearly higher rate than controls.")
    for v in ["v1", "v2a", "v2b"]:
        rate = m.groupby("question_type")[f"route_{v}"].mean()
        c1 = (m[f"router_{v}"].sum() >= best - 2) and calls[v] < len(m)
        print(f"  {v}: accuracy {int(m[f'router_{v}'].sum())} (best fixed {best}), criterion (i) {'met' if c1 else 'NOT met'};"
              f" routing policy {rate['policy_dependent']:.2f} vs control {rate['explicit_control']:.2f}")

    comps = [compare(m, a, b) for a, b in [("rag", "direct"), ("router_v2b", "direct"),
                                           ("router_v2b", "rag"), ("router_v2a", "rag")]]
    pol = m[m.question_type == "policy_dependent"]
    con = m[m.question_type == "explicit_control"]
    comps += [compare(pol, "rag", "direct", "policy-dependent"), compare(con, "rag", "direct", "control")]
    show(comps, "Paired comparisons (cluster = intent):")


if __name__ == "__main__":
    strict_benchmark()
    pd_test()
