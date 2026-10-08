"""Re-executes every stored prediction against the restored research database and
scores it two ways:

  strict    the paper's original metric: same number of columns and the same
            multiset of rows (row order ignored), values normalised as in the notebooks.
  tolerant  additionally (a) rounds every numeric value to two decimals and
            (b) accepts a result whose columns are a projection of the other result
            (an extra or missing column, e.g. the product name).

Each strict failure is also classified:
  not_executed | rounding_only | column_only | rounding_and_column | other

Usage (database restored with data/database/restore.sh):
    python src/rescore.py "postgresql://user:pass@host:5432/db"
Writes results/strict_benchmark/rescored_all_systems.csv
"""
import itertools
import json
import math
import sys
from decimal import Decimal
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "results" / "strict_benchmark"

SYSTEMS = {
    # name: list of (file, sql column, split filter or None)
    "ft_direct": [("FT_validation_direct_and_rag.csv", "direct_sql"), ("FT_locked_test_direct_and_rag.csv", "direct_sql")],
    "ft_rag": [("FT_validation_direct_and_rag.csv", "rag_sql"), ("FT_locked_test_direct_and_rag.csv", "rag_sql")],
    "orig_zero": [("B2_original_zeroshot.csv", "sql")],
    "orig_few": [("B3_original_fewshot3.csv", "sql")],
    "orig_rag": [("B4_original_rag.csv", "sql")],
    "coder_zero": [("B6_qwen25coder7b_zeroshot.csv", "sql")],
    "coder_few": [("B6_qwen25coder7b_fewshot3.csv", "sql")],
}


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, Decimal):
        if v == v.to_integral():
            return str(v.quantize(Decimal("1")))
        s = format(v.normalize(), "f")
        return s.rstrip("0").rstrip(".") if "." in s else s
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return format(v, ".10g")
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v)


def round2(v):
    try:
        return format(round(float(v), 2), ".2f")
    except (TypeError, ValueError):
        return v


def sort_rows(rows):
    return sorted(rows, key=lambda r: json.dumps(r, ensure_ascii=False, default=str))


def equal(a, b):
    """a, b: lists of normalised tuples. Same width and same multiset."""
    if not a and not b:
        return True
    if not a or not b or len(a[0]) != len(b[0]):
        return False
    return sort_rows(a) == sort_rows(b)


def projection_equal(a, b):
    """True if the narrower result equals some column projection of the wider one."""
    if equal(a, b):
        return True
    if not a or not b or len(a) != len(b):
        return False
    wide, narrow = (a, b) if len(a[0]) > len(b[0]) else (b, a)
    k = len(narrow[0])
    if k == len(wide[0]) or len(wide[0]) > 6:
        return False
    target = sort_rows(narrow)
    for cols in itertools.permutations(range(len(wide[0])), k):
        if sort_rows([tuple(r[c] for c in cols) for r in wide]) == target:
            return True
    return False


class DB:
    def __init__(self, url):
        self.conn = psycopg2.connect(url)
        self.conn.autocommit = True
        self.cache = {}

    def run(self, sql):
        if not isinstance(sql, str) or not sql.strip():
            return None
        if sql in self.cache:
            return self.cache[sql]
        try:
            with self.conn.cursor() as cur:
                cur.execute("SET statement_timeout TO 30000; SET search_path TO build4all_research;")
                cur.execute(sql)
                rows = [tuple(norm(v) for v in r) for r in cur.fetchall()]
        except Exception:
            rows = None
        self.cache[sql] = rows
        return rows


def score(db, gold_sql, pred_sql):
    g, p = db.run(gold_sql), db.run(pred_sql)
    if p is None:
        return False, False, "not_executed"
    strict = equal(g, p)
    if strict:
        return True, True, "correct"
    gr = [tuple(round2(v) for v in r) for r in g]
    pr = [tuple(round2(v) for v in r) for r in p]
    rnd = equal(gr, pr)
    col = projection_equal(g, p)
    both = projection_equal(gr, pr)
    kind = ("rounding_only" if rnd else "column_only" if col
            else "rounding_and_column" if both else "other")
    return False, both, kind


def load_predictions():
    frames = []
    for system, sources in SYSTEMS.items():
        for fname, col in sources:
            d = pd.read_csv(STRICT / fname)
            d = d[d["split"].isin(["validation", "final_test"])]
            frames.append(pd.DataFrame({
                "system": system, "question_id": d["question_id"], "split": d["split"],
                "intent_id": d["intent_id"], "template_family": d["template_family"],
                "language": d["language"], "gold_sql": d["gold_sql"], "pred_sql": d[col]}))
    return pd.concat(frames, ignore_index=True)


def main(url):
    db = DB(url)
    pred = load_predictions()
    out = pred.apply(lambda r: score(db, r.gold_sql, r.pred_sql), axis=1, result_type="expand")
    pred[["strict", "tolerant", "error_type"]] = out
    pred.drop(columns=["gold_sql"]).to_csv(STRICT / "rescored_all_systems.csv", index=False)
    print(pred.pivot_table(index="system", columns="split", values=["strict", "tolerant"], aggfunc="sum"))
    print(pd.crosstab([pred.system, pred.split], pred.error_type))


if __name__ == "__main__":
    main(sys.argv[1])
