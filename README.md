# When Retrieval Helps and When It Hurts — Build4All Multilingual Text-to-SQL

Code, data, and per-question results for the paper

> **When Retrieval Helps and When It Hurts: Fine-Tuning, Policy Retrieval, and Retrieval Routing for Multilingual Text-to-SQL with Lebanese Arabizi**
> Louma Akoum — Faculty of Technology, Lebanese University

The study evaluates QLoRA fine-tuning of Qwen3-8B, retrieval of business-policy documents (RAG), and three retrieval routers on Text-to-SQL questions in **English, French, Arabic, and Lebanese Arabizi**.

## Main results

**Strict benchmark** (no SQL template family shared between training, validation, and locked test; 100 questions per held-out set):

| System | Validation | Locked test |
|---|---|---|
| Qwen3-8B zero-shot | 46 | 56 |
| Qwen3-8B few-shot (3 retrieved examples) | 85 | 76 |
| Qwen2.5-Coder-7B few-shot | 58 | 74 |
| **Qwen3-8B + QLoRA** | **92** | **94** |
| Qwen3-8B + QLoRA + RAG | 87 | 97 |

**Policy-dependent test (PD-Test)**, 128 new questions; all routing decisions were frozen before generation:

| System | Policy-dependent (64) | Explicit control (64) | Total | RAG calls |
|---|---|---|---|---|
| Fine-tuned, direct | 11 | 59 | 70 | 0 |
| Fine-tuned + always-on RAG | 52 | 58 | 110 | 128 |
| Router v1 (embedding) | 11 | 59 | 70 | 0 |
| Router v2a (retrieval scores) | 22 | 60 | 82 | 42 |
| Router v2b (policy-term similarity) | 45 | 59 | 104 | 84 |

## Repository layout

```
data/
  build4all_multilingual_text2sql_600.csv          600 questions (150 intents x 4 languages), gold SQL
  build4all_custom600_strict_template_split.csv    the strict template-family split used in the paper
  build4all_policy_test_v1.csv                     PD-Test: 128 questions (16 policy-dependent + 16 control intents)
  policies/                                        the four policy documents used for retrieval
  database/                                        export of the research database + schema.sql + restore.sh
notebooks/
  1_main_experiment.ipynb       split, QLoRA training, fine-tuned/RAG evaluation, router v1, locked test (Kaggle, 2x T4)
  2_revision_experiments.ipynb  reproduction check, zero-/few-shot baselines, Qwen2.5-Coder, router features (Colab, T4)
  3_policy_test.ipynb           PD-Test: gold audit, router freeze (D1), direct and RAG runs, evaluation fix
results/
  strict_benchmark/             per-question outcomes of every system on the 200 held-out questions
  pd_test/                      frozen routing decisions, freeze manifest, pre-registration, PD-Test outcomes
src/
  controllers_v2.py             router v2a / v2b definitions (hash recorded in the freeze manifest)
  paper_statistics.py           recomputes every accuracy, McNemar test, and confidence interval in the paper
```

## Reproducing the statistics (CPU, no database needed)

```bash
pip install pandas numpy
python src/paper_statistics.py
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

The fine-tuned QLoRA adapter (175 MB) is not stored in this repository.
<!-- NOTE (author): add the Hugging Face / Zenodo link of the adapter here. -->

## Integrity notes

- **Pre-registration.** The PD-Test routing decisions were computed and saved before any PD-Test generation.
  `results/pd_test/D1_freeze_manifest.json` records the SHA-256 of `D1_frozen_controller_routes.csv`
  (`09d92cdf…d8e0`) and of the router source code. Both match the files in this repository.
- **Reproducibility.** The frozen fine-tuned model scored 94/100 on the locked test on Kaggle and again on Colab,
  with identical per-question outcomes.
- **Evaluation fix.** In the first PD-Test run, a variable in notebook 3 shadowed a function used by the result
  normaliser, so integer-valued results raised an error. Those records were removed and regenerated (see the last
  cell of notebook 3). Routing decisions were frozen earlier and are unaffected.
- **Policy index.** Each policy PDF was present twice in the input data, so the retrieval index holds every chunk
  twice (52 entries, 26 unique chunks). All reported RAG results use this index.
- **Notebook 1** is the original experiment notebook, trimmed to the Text-to-SQL experiment. Cells for other
  project components and an evidence-packaging cell were removed. The paper reports only values computed by the
  remaining cells and by notebooks 2 and 3.

## Data provenance and use of AI

Questions and gold SQL were drafted with the assistance of large language models. Every gold query was validated
by the SQL safety checker and executed against the research database. The PD-Test questions were reviewed by a
native speaker of Lebanese Arabic. The data are synthetic and contain no personal information.

## Citation

<!-- NOTE (author): add the BibTeX entry once the paper is published. -->
