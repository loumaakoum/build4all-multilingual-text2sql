
### Pre-registered evaluation protocol (fixed on 6 Oct 2026, before any PD-Test generation)

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
