
CodeRed/
├── data/
│   ├── raw/                    # untouched original files as given
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/              # cleaned/encoded versions you generate
│       ├── train_clean.csv
│       └── test_clean.csv
│
├── docs/                       # your 6 planning files
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
│   └── 04_final_predictions.ipynb   # the one you actually submit
│
├── src/                         # reusable code, pulled out of notebooks
│   ├── preprocessing.py         # encoding, scaling, imputation functions
│   ├── cost_matrix.py           # the cost matrix + cost-scoring function
│   ├── train.py                 # model training/CV logic
│   └── evaluate.py              # metrics: F1/macro-F1/confusion matrix/cost
│
├── models/
│   └── best_model.pkl           # saved final model (optional but good practice)
│
├── outputs/
│   ├── MM2645_MM26ML03.csv   # final submission file — exact naming
│   ├── confusion_matrix.png
│   └── feature_importance.png   # if you do the explainability bonus
│
├── README.md                    # max 1 page, required by the problem statement
└── requirements.txt