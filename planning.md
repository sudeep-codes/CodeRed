# CodeRed

**Project Plan & Team Playbook — MM26ML03: Healthcare Triage Classification (Modelling Minds 2.0)**

---

## 1. Project Overview

Build a cost-sensitive multi-class classifier that sorts emergency patients into RED / YELLOW / GREEN / BLACK triage categories per the START protocol, explicitly optimizing against the organizer-provided cost matrix (not accuracy alone). Team size: 2.

---

## 2. Role Split

### Sudeep Kumar Sahu — Pipeline, Evaluation & Narrative Owner

- **EDA + `data-report.md`** — class balance across RED/YELLOW/GREEN/BLACK, missing values, feature distributions, correlations.
- **Preprocessing pipeline** (`src/preprocessing.py`) — encoding, scaling, imputation, kept consistent across train/test.
- **Cost matrix + evaluation harness** (`src/cost_matrix.py`, `src/evaluate.py`) — the given cost matrix hardcoded, scoring function, and the full metrics reporter (accuracy, per-class P/R/F1, macro/weighted-F1, confusion matrix, total cost).
- **`experiment-log.md`** — logs every model run your friend produces, so the final model is picked on numbers, not gut feel.
- **Visualizations** — confusion matrix heatmap, per-class F1 bars, feature importance plot (if attempting the explainability bonus).
- **Submission packaging** — final CSV formatting (`team_ID_MM26ML03.csv`, correct columns, patient order preserved), README (max 1 page), repo/zip structure.

### Shrijay Sinha — Modeling Owner

- Model selection and training: baseline (e.g. logistic regression) → cost-sensitive variants (class weights, custom loss, threshold shifting) → tree ensembles / boosting.
- Hyperparameter tuning, evaluated strictly against your shared evaluation harness (so scores are apples-to-apples).
- **`modeling-plan.md`** — documents what was tried, what worked, what didn't, and why.

**Why this split:** you're not duplicating effort, you can explain *why* the model was chosen in judge Q&A, and because evaluation/cost code lives separately from modeling code, your friend can swap models freely without breaking anything.

---

## 3. Which LLM to Use For What (Antigravity + Gemini Pro)

Antigravity supports model switching per task — Gemini 3 Pro, Gemini 3.5 Flash, Claude Sonnet 4.6, and GPT-OSS 120B are all available in the same IDE. Use the model suited to the task rather than defaulting to one model for everything.

| Task | Recommended Model | Why |
|---|---|---|
| Planning docs (PRD, data-report, modeling-plan, rules, phases) | Gemini 3 Pro | Best for structured reasoning and long-context planning; strong at synthesizing the problem statement into clear docs. |
| EDA + preprocessing code (pandas, sklearn pipelines) | Claude Sonnet 4.6 | Precise, low-hallucination code generation — important since preprocessing bugs silently corrupt every downstream metric. |
| Cost matrix + evaluation harness (the scoring logic itself) | Claude Sonnet 4.6 (Thinking) | This code must be exactly correct — a wrong cost function invalidates every model comparison. Use the more careful/thinking variant here, and review the output by hand. |
| Model training loops, hyperparameter sweeps, boilerplate iteration | Gemini 3.5 Flash | Optimized for fast agentic/coding loops — cheap and quick for repetitive experiment iterations your friend will run many times. |
| Quick lookups, boilerplate snippets, low-stakes fixes | Gemini 3.5 Flash (Fast mode) | Lower latency, good enough for small non-critical edits so you're not burning Pro/Sonnet quota on trivial tasks. |
| README + final write-up (the 1-page submission doc) | Gemini 3 Pro or Claude Sonnet 4.6 | Either works well for polished prose; pick whichever has spare quota at submission time. |
| Debugging a confusing model result or unexpected metric | Claude Sonnet 4.6 (Thinking) | Reasoning-heavy debugging benefits from the slower, more deliberate thinking mode over fast agentic loops. |

**Rule of thumb:** reasoning/correctness-critical work (cost matrix, evaluation harness, planning docs) → Gemini 3 Pro or Claude Sonnet 4.6 (Thinking). High-volume/iterative work (training loops, sweeps, small fixes) → Gemini 3.5 Flash. Always hand-verify anything that touches the cost matrix or final metrics regardless of which model wrote it.

---

## 4. Project Structure

Repository name: **CodeRed**

```
CodeRed/
├── data/
│   ├── raw/                       # untouched originals — never edit
│   │   ├── train_ml03.csv
│   │   └── test_pred_ml03.csv
│   └── processed/                 # cleaned/encoded versions you generate
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── docs/                          # the 6 planning files
│   ├── PRD.md
│   ├── data-report.md
│   ├── modeling-plan.md
│   ├── rules.md
│   ├── phases.md
│   └── experiment-log.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_final_predictions.ipynb   # the notebook you submit
│
├── src/
│   ├── preprocessing.py           # encoding, scaling, imputation
│   ├── cost_matrix.py             # cost matrix + cost-scoring function
│   ├── train.py                   # model training / CV logic
│   └── evaluate.py                # F1 / macro-F1 / confusion matrix / cost
│
├── models/
│   └── best_model.pkl             # saved final model
│
├── outputs/
│   ├── MM2645_MM26ML03.csv     # final submission — exact naming required
│   ├── confusion_matrix.png
│   └── feature_importance.png     # if attempting explainability bonus
│
├── README.md                      # max 1 page, required by problem statement
└── requirements.txt
```

**Notes:** `data/raw` is read-only for the whole team — all transformations write to `data/processed`. `src/cost_matrix.py` and `src/evaluate.py` are shared, single-source-of-truth files so every model your friend trains is scored identically. `notebooks/04_final_predictions.ipynb` should read cleanly top-to-bottom — assume a judge opens only this one.

---
*CodeRed · MM26ML03 · Modelling Minds 2.0*
