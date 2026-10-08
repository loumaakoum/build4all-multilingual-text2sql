
### Frozen-before-generation evaluation protocol (fixed on 6 Oct 2026, before any PD-Test generation)

> This protocol was recorded in this repository, not in an external registry. Evidence of the order:
> `D1_frozen_controller_routes.csv` and `D1_freeze_manifest.json` were written at 20:38:26 UTC on 6 Oct 2026
> (timestamp inside the manifest); the first PD-Test generation file (`D2_pd_finetuned_direct.jsonl`) was
> created at 20:38:48 UTC. Outcome: no router met both criteria below (v2b met criterion 2 only).

**Systems compared on the 128 PD-Test questions:** fine-tuned direct; fine-tuned + always-on RAG;
controller v1 (frozen, tau = 0.93); controller v2a (retrieval-feature logistic regression, tau chosen by
leave-one-family-out on the 200 old held-out questions); controller v2b (policy-term similarity rule,
tau chosen on the same 200 questions); oracle (upper bound). Controllers only *select* between the two
generated answers, so no extra generation is needed for them.

**A controller is called useful if both hold:**
1. its accuracy is no more than 2 questions below the better of direct and always-on RAG, **and** it calls RAG
   on fewer questions than always-on RAG;
2. it routes policy-dependent questions to RAG at a clearly higher rate than control questions.

All results will be reported whatever they are. Nothing (prompts, thresholds, features, data) changes after D2/D3 start.
Run **D0 → D1 (freeze) → D2 → D3 → (D4) → D5**.
