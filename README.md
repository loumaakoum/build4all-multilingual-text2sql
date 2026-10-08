# When Retrieval Helps and When It Hurts — Build4All Multilingual Text-to-SQL

Code, data, and per-question results for the paper

> **When Retrieval Helps and When It Hurts: Fine-Tuning, Policy Retrieval, and Retrieval Routing for Multilingual Text-to-SQL with Lebanese Arabizi**
> Louma Akoum — Faculty of Technology, Lebanese University

The study evaluates QLoRA fine-tuning of Qwen3-8B, retrieval of business-policy documents (RAG), and three retrieval routers on Text-to-SQL questions in **English, French, Arabic, and Lebanese Arabizi**.

## Main results

**Strict benchmark** (no SQL template family shared between training, validation, and locked test; 100 questions per
held-out set). *Strict* is the original execution-accuracy metric. *Tolerant* also rounds numbers to two decimals and
accepts an extra or missing column. The gold queries follow two conventions that no question states (rounding to two
decimals, and a product-name column in "top products" queries), and most strict differences come from them.
With the conventions stated in the prompt, few-shot prompting beats the fine-tuned model on validation; its 24
locked-test failures are all one SQL error (an ambiguous `product_id` in the top-products join). The robust gain of
fine-tuning is over zero-shot prompting, above all in Lebanese Arabizi (48/50 vs 14–27/50).

| System | Validation strict | Validation tolerant | Locked test strict | Locked test tolerant |
|---|---|---|---|---|
| Qwen3-8B zero-shot | 46 | 73 | 56 | 94 |
| Qwen3-8B few-shot (3 retrieved examples) | 85 | 94 | 76 | 100 |
| Qwen3-8B zero-shot, conventions stated in prompt | 77 | 77 | 61 | 61 |
| Qwen3-8B few-shot, conventions stated in prompt | 97 | 97 | 76 | 76 |
| Qwen2.5-Coder-7B few-shot | 58 | 84 | 74 | 95 |
| Qwen3-8B + QLoRA | 92 | 92 | 94 | 100 |
| Qwen3-8B + QLoRA + RAG | 87 | 87 | 97 | 100 |

**Policy-dependent test (PD-Test)**, 128 new questions; all routing decisions were frozen before generation:

| System | Policy-dependent (64) | Explicit control (64) | Total | RAG calls |
|---|---|---|---|---|
| Fine-tuned, direct | 11 | 59 | 70 | 0 |
| Fine-tuned + always-on RAG | 52 | 58 | 110 | 128 |
| Router v1 (embedding) | 11 | 59 | 70 | 0 |
| Router v2a (retrieval scores) | 22 | 60 | 82 | 42 |
| Router v2b (policy-term similarity) | 45 | 59 | 104 | 84 |

No router met both pre-set success criteria. v2b routed only 7 of 16 Arabic policy-dependent questions to RAG.

## Repository layout

```
data/
  build4all_multilingual_text2sql_600.csv          600 template-generated questions (150 intents x 4 languages), gold SQL
  build4all_custom600_strict_template_split.csv    the strict template-family split used in the paper
  build4all_policy_test_v1.csv                     PD-Test: 128 questions (16 policy-dependent + 16 control intents)
  policies/                                        the four policy documents used for retrieval
  database/                                        export of the research database + schema.sql + restore.sh
notebooks/
  1_main_experiment.ipynb       split, QLoRA training, fine-tuned/RAG evaluation, router v1, locked test (Kaggle, 2x T4)
  2_revision_experiments.ipynb  reproduction check, zero-/few-shot baselines, Qwen2.5-Coder, router features (Colab, T4)
  3_policy_test.ipynb           PD-Test: gold audit, router freeze (D1), direct and RAG runs, evaluation fix
  4_conventions_rerun.ipynb     zero-/few-shot baselines with the two output conventions stated in the prompt
results/
  strict_benchmark/
    FT_validation_direct_and_rag.csv    fine-tuned model, direct and RAG SQL, validation (with retrieved passages)
    FT_locked_test_direct_and_rag.csv   fine-tuned model, direct and RAG SQL, locked test
    B1..B6_*.csv, E1/E2_*.csv           baselines, and baselines with conventions stated (SQL and outcome per question)
    rescored_all_systems.csv            strict and tolerant score and failure type of every prediction
    all_systems_200.csv                 recorded strict outcomes of every system (one row per question)
  pd_test/                      frozen routing decisions, freeze manifest, freeze protocol, PD-Test outcomes
src/
  controllers_v2.py             router v2a / v2b definitions (hash recorded in the freeze manifest)
  rescore.py                    re-executes every stored prediction on the restored database (strict + tolerant)
  paper_statistics.py           recomputes the accuracies, clustered tests, intervals and tables in the paper
```

## Reproducing the results

The statistics run on CPU from the files in `results/` (no database needed):

```bash
pip install pandas numpy
python src/paper_statistics.py
```

Paired comparisons treat the **intent** (one question in four languages) as the cluster: exact intent-level
permutation test, plus the cluster-adjusted McNemar statistics of Durkalski et al. (2003) and Obuchowski (1998) in
the form used by the R package `clust.bin.pair`. A family-level permutation test and the question-level McNemar test
are printed for reference. Training-run values (loss, steps, runtime) come from the training log in notebook 1.

To re-score every stored prediction from its SQL (this also checks that all 1,400 recorded strict outcomes
reproduce), restore the database (below) and run:

```bash
pip install pandas psycopg2-binary
python src/rescore.py "postgresql://user:password@host:5432/dbname"
```

## Re-running the experiments

The notebooks need a GPU (one 16 GB T4 is enough for inference), the four policy PDFs, and read access to the
`build4all_research` PostgreSQL schema through a secret named `NEON_DATABASE_URL`. The research database contains
synthetic data only.

### Restoring the research database

`data/database/` holds a full export of the 10 tables (CSV), the table definitions (`schema.sql`), and a restore
script. On any PostgreSQL server (local, Docker, or a free Neon project):

```bash
cd data/database
./restore.sh "postgresql://user:password@host:5432/dbname"
```

This creates the `build4all_research` schema and loads 5,000 orders, 12,500 order items, 5,000 payments,
25,000 product events, 500 customers, 150 products, and the small lookup tables. Then set `NEON_DATABASE_URL` to the
same connection URL in the notebooks. All 182 gold queries (150 benchmark intents and 32 PD-Test intents) execute
on the restored database.

The fine-tuned QLoRA adapter (175 MB) is on Hugging Face:
[loumaakoum/qwen3-8b-build4all-text2sql-qlora](https://huggingface.co/loumaakoum/qwen3-8b-build4all-text2sql-qlora).
Load it on top of `Qwen/Qwen3-8B` with `PeftModel.from_pretrained` (example in the model card).

## Integrity notes

- **Frozen before generation.** The PD-Test routing decisions were computed and saved before any PD-Test
  generation (manifest timestamp 20:38:26 UTC on 6 Oct 2026; first generation file created 22 seconds later).
  `results/pd_test/D1_freeze_manifest.json` records the SHA-256 of `D1_frozen_controller_routes.csv`
  (`09d92cdf…d8e0`) and of the router source code; both match the files here. The protocol and success criteria
  are in `results/pd_test/FREEZE_PROTOCOL.md`. It was recorded in this repository, not in an external registry.
- **Reproducibility.** The frozen fine-tuned model scored 94/100 on the locked test on Kaggle and again on Colab,
  with identical SQL for all 100 questions.
- **Evaluation fix.** In the first PD-Test run, a variable in notebook 3 shadowed a function used by the result
  normaliser, so integer-valued results raised an error. Those records were removed and regenerated (see the last
  cell of notebook 3). Routing decisions were frozen earlier and are unaffected.
- **Policy index.** All reported RAG runs used a 26-chunk index of the four PDFs. Notebook 1 was partly
  re-executed on 3 October with a second copy of the PDFs attached, so some of its printed outputs show 8 PDFs and
  52 chunks; a note at the top of the notebook explains this. (An earlier version of this README and of the paper
  wrongly said that all runs used the 52-entry index.)
- **Router-v1 freeze manifest (25 September).** Written by notebook 1, it contains hand-typed summary values that
  differ from the computed ones (e.g. 0.91 instead of 0.92 direct validation accuracy, and router counts
  that do not correspond to the frozen threshold 0.93, which routes 8 validation questions).
  The paper uses values recomputed from the per-question files.
- **Notebook 1** is the original experiment notebook, trimmed to the Text-to-SQL experiment. Cells for other
  project components and an evidence-packaging cell were removed.

## Data provenance and use of AI

Questions and gold SQL were generated from templates with the assistance of large language models. Every gold
query was validated by the SQL safety checker and executed against the research database. The PD-Test questions
were reviewed by a native speaker of Lebanese Arabic; the 600-question benchmark was not. The policy documents were
written for this study and define metrics in terms of columns and status values. The data are synthetic and
contain no personal information.

## Citation

<!-- NOTE (author): add the BibTeX entry once the paper is published. -->
