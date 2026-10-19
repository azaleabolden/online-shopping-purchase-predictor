# Online Shopping Purchase Predictor

## Project Goal
Predict whether an online shopping session will result in a purchase using supervised machine learning.

## Current Project Stage
This package contains the completed technical work through the **Streamlit demo stage**.

It continues the work from Practical Checkpoints 1 and 2.

## Main Technical Contributions
- Investigated and removed 125 exact duplicate rows before train/test splitting.
- Treated numeric category codes as categorical variables.
- Used a shared 80/20 stratified split for fair model comparison.
- Compared:
  - Majority-Class Baseline
  - Logistic Regression
  - Logistic Regression + Balanced
  - Decision Tree
  - Decision Tree + Balanced
  - Random Forest
  - Random Forest + Balanced
- Evaluated Accuracy, Precision, Recall, F1-score, FP, FN, and TP.
- Tested class weighting and analyzed the precision–recall tradeoff.
- Selected standard Random Forest as the final demo model based on F1-score.
- Built a Streamlit interface for interactive predictions.

## Files
- `START_HERE.md` — exact next steps
- `Online_Shopping_Purchase_Predictor_Final.ipynb` — final analysis notebook
- `src/app.py` — Streamlit interface
- `src/train_and_evaluate.py` — reproducible training/evaluation script
- `requirements.txt` — dependencies
- `results/model_comparison.csv` — model results
- `results/RESULTS_SUMMARY.md` — concise interpretation
- `data/README.md` — dataset access information
- `artifacts/README.md` — model artifact instructions

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the notebook from top to bottom, then launch the app:

```powershell
streamlit run src/app.py
```

See `START_HERE.md` for the full workflow.

## Dataset
UCI Online Shoppers Purchasing Intention Dataset:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## AI Assistance Disclosure
ChatGPT was used for planning, debugging assistance, code organization, and cleanup of reproducibility materials.
All reported model results should come from executed project code.
