# Online Shopping Purchase Predictor

## Project Goal
Predict whether an online shopping session will result in a purchase using supervised machine learning.

## Project Stage
This repository contains the completed final project. It continues the work from Practical Checkpoints 1 and 2 and includes the final analysis notebook, the saved model, and the Streamlit demo.

## Links
- Streamlit app: https://onlinepurchasepredictor.streamlit.app/
- Demo video: https://youtu.be/0sFBjmopcNY
- The research paper is submitted separately as a PDF with these links on its supplemental page.

## Research Question
How accurately can machine learning predict whether an online shopping session will result in a purchase based on customer browsing behavior?

## Main Technical Contributions
- Investigated and removed 125 exact duplicate rows before train/test splitting.
- Treated numeric category codes (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) as categorical variables.
- Used a shared 80/20 stratified split with `random_state=42`, giving 9,764 training sessions and 2,441 test sessions.
- Compared seven models: Majority-Class Baseline, Logistic Regression, Logistic Regression + Balanced, Decision Tree, Decision Tree + Balanced, Random Forest, and Random Forest + Balanced.
- Selected the final model with 5-fold stratified cross-validation on the training set only, then tuned the Random Forest settings with a grid search.
- Evaluated the selected model once on the held-out test set.
- Ran a sensitivity check without `PageValues`, a feature that may not be available before a purchase occurs.
- Examined why class weighting helped Logistic Regression but not the tree models.
- Built a Streamlit interface that uses the final tuned Random Forest for interactive predictions.

## Final Results
Held-out test set of 2,441 sessions.

| Model | Accuracy | Precision | Recall | F1 | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| Random Forest, tuned (selected by cross-validation) | 0.8906 | 0.6316 | 0.7225 | 0.6740 | 161 | 106 | 276 |
| Random Forest, default settings | 0.9029 | 0.7617 | 0.5524 | 0.6404 | 66 | 171 | 211 |
| Random Forest, tuned, without PageValues | 0.8202 | 0.4139 | 0.3586 | 0.3843 | 194 | 245 | 137 |

Cross-validation selected Random Forest (mean F1 0.6284). Tuning raised its cross-validation F1 to 0.6839 using balanced class weights, no depth limit, and a minimum of 3 samples per leaf. Removing `PageValues` sharply lowered performance, so the results describe a retrospective setting and not a live prediction during a session.

## Files
| File | Purpose |
|---|---|
| `START_HERE.md` | Step by step workflow |
| `Online_Shopping_Purchase_Predictor_Final.ipynb` | Final analysis notebook, including cross-validation selection, tuning, and the PageValues check |
| `src/app.py` | Streamlit interface |
| `src/train_and_evaluate.py` | Training and evaluation script |
| `requirements.txt` | Pinned dependencies |
| `results/model_comparison.csv` | Model results |
| `results/RESULTS_SUMMARY.md` | Concise interpretation |
| `data/README.md` | Dataset access information |
| `artifacts/README.md` | Model artifact instructions |

## Quick Start
Environment used: Python 3.10.11, scikit-learn 1.7.2, pandas 2.3.3, NumPy 2.2.6.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the notebook from top to bottom, then launch the app:

```powershell
streamlit run src/app.py
```

The notebook downloads the dataset automatically, so the first run needs an internet connection. See `START_HERE.md` for the full workflow.

## Dataset
UCI Online Shoppers Purchasing Intention Dataset:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## AI Assistance Disclosure
ChatGPT was used to assist with planning, code organization, and explanations. Claude was also briefly used mainly to help troubleshoot code and install packages, and it also helped revise parts of the paper text, create figures from the reported results, and check reference details. All reported model results come from executed project code.