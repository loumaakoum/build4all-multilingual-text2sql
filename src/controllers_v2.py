# ============================================================
# FROZEN CONTROLLER DEFINITIONS (pre-registered before any PD-Test result exists)
# Training data: the 200 old held-out questions (validation + locked test of the
# strict benchmark) with the fine-tuned direct / RAG outcomes. No PD-Test labels.
# ============================================================
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RETRIEVAL_FEATURES = ["score_1", "score_2", "score_3", "score_4", "score_5", "score_6",
                      "top_score", "mean_score", "std_score", "gap_1_2", "question_length"]

# Business metrics defined in the four policy documents (names only, copied from the PDFs).
POLICY_METRIC_NAMES = [
    "recognized order revenue", "gross merchandise value", "net merchandise sales",
    "average order value", "order count by status", "operational backlog", "cancellation rate",
    "refunded order count", "daily order volume", "recognized coupon discount",
    "recognized product sales by category", "recognized shipping amount", "effective tax rate",
    "collected paid amount", "pending payment amount", "failed payment rate",
    "refunded payment amount", "paid payment mismatch", "product count by catalog status",
    "active products with fewer than ten units", "active inventory value", "current sale price",
    "coupon validity", "event count by type", "view-to-purchase conversion",
    "purchasing customers by region", "monthly customer registrations",
    "repeat purchasing customers", "order count by shipping method",
]


def select_threshold(scores, direct_ok, rag_ok, grid):
    """Pick the threshold with the highest accuracy; among ties, the MIDDLE of the
    widest tied run (robust to small score shifts, unlike the extreme of a plateau)."""
    acc = np.array([np.where(scores >= t, rag_ok, direct_ok).mean() for t in grid])
    best = acc.max()
    runs, cur = [], []
    for t, a in zip(grid, acc):
        if a == best:
            cur.append(t)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    widest = max(runs, key=len)
    return float(np.median(widest)), float(best)


def fit_controller_v2a(feat200, labels200):
    """v2a: logistic regression on the 11 retrieval features only (no question
    embedding, which memorised families in v1). Threshold chosen on
    leave-one-FAMILY-out out-of-fold probabilities."""
    df = feat200.merge(labels200[["question_id", "template_family", "ft_direct_correct", "ft_rag_correct"]].rename(columns={"template_family": "_fam"}), on="question_id")
    X = df[RETRIEVAL_FEATURES].to_numpy(float)
    y = (df.ft_rag_correct & ~df.ft_direct_correct).astype(int).to_numpy()
    fam = df["_fam"].to_numpy()
    make = lambda: Pipeline([("s", StandardScaler()),
                             ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                                       max_iter=5000, random_state=42))])
    oof = np.zeros(len(df))
    for f in np.unique(fam):
        tr, te = fam != f, fam == f
        if y[tr].sum() == 0:
            oof[te] = 0.0
            continue
        m = make().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    tau, oof_acc = select_threshold(oof, df.ft_direct_correct.to_numpy(), df.ft_rag_correct.to_numpy(),
                                    np.round(np.arange(0.05, 0.951, 0.01), 2))
    model = make().fit(X, y)
    return model, tau, oof_acc


def fit_controller_v2b(term_sim200, labels200):
    """v2b: training-free policy-term router. Route to RAG when the question's best
    E5 similarity to a policy metric name is >= tau. tau chosen on the 200 old
    questions (single parameter, no family-specific learning)."""
    df = term_sim200[["question_id", "term_similarity"]].merge(labels200, on="question_id")
    tau, acc = select_threshold(df.term_similarity.to_numpy(), df.ft_direct_correct.to_numpy(),
                                df.ft_rag_correct.to_numpy(), np.round(np.arange(0.70, 0.951, 0.005), 3))
    return tau, acc
