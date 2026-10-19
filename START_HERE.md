# START HERE — Final Project Through Streamlit Demo

This folder continues the same Online Shopping Purchase Predictor completed in Practical Checkpoints 1 and 2.

## What is already complete

The project currently:
- loads the UCI Online Shoppers Purchasing Intention Dataset,
- investigates and removes 125 exact duplicate rows before splitting,
- treats `OperatingSystems`, `Browser`, `Region`, and `TrafficType` as categorical variables,
- uses an 80/20 stratified train/test split,
- compares a majority-class baseline with Logistic Regression, Decision Tree, and Random Forest,
- tests balanced class weighting,
- reports Accuracy, Precision, Recall, F1-score, FP, FN, and TP,
- identifies Random Forest as the strongest overall model by F1-score,
- shows the precision–recall tradeoff,
- includes a weak/failure-case workflow,
- includes sample predictions and feature importance.

## Reported model results

All models were tested on the same **2,441** held-out cases.

### Random Forest
- Accuracy: 0.9029
- Precision: 0.7617
- Recall: 0.5524
- F1-score: 0.6404

### Logistic Regression + Balanced
- Accuracy: 0.8460
- Precision: 0.5050
- Recall: 0.7958
- F1-score: 0.6179

The Random Forest is used for the Streamlit demo because it had the strongest overall F1-score.

---

# Step 1 — Open the project folder in VS Code

Open:

`Online_Shopping_Predictor_Final_Project_Through_Streamlit`

Use the same Python/Jupyter environment that successfully ran Checkpoint 2.

---

# Step 2 — Install dependencies

Open a VS Code terminal in the project folder and run:

```powershell
python -m pip install -r requirements.txt
```

If your current environment already contains these packages, pip will keep the installed versions or update as needed.

---

# Step 3 — Run the final notebook

Open:

`Online_Shopping_Purchase_Predictor_Final.ipynb`

Then:

1. Restart the kernel.
2. Click **Run All**.
3. Confirm there are no red errors.
4. Scroll to the final section:
   **Save the Final Model for Streamlit**
5. Confirm the notebook prints that the model was saved.
6. Confirm the final readiness cell says:

`Model ready for Streamlit: True`

Then save the notebook.

The following file should now exist:

`artifacts/random_forest_pipeline.joblib`

---

# Step 4 — Confirm the final results

In the notebook, confirm the full comparison still shows:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Majority-Class Baseline | 0.8435 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.8869 | 0.7524 | 0.4136 | 0.5338 |
| Logistic Regression + Balanced | 0.8460 | 0.5050 | 0.7958 | 0.6179 |
| Decision Tree | 0.8566 | 0.5400 | 0.5654 | 0.5524 |
| Decision Tree + Balanced | 0.8558 | 0.5412 | 0.5157 | 0.5282 |
| Random Forest | 0.9029 | 0.7617 | 0.5524 | 0.6404 |
| Random Forest + Balanced | 0.8931 | 0.7266 | 0.5079 | 0.5978 |

If the numbers change because the code/data changed, use the **new executed results** going forward.

---

# Step 5 — Launch the Streamlit demo

In the VS Code terminal, from the project root, run:

```powershell
streamlit run src/app.py
```

A browser tab should open automatically.

If it does not, copy the Local URL printed in the terminal into your browser.

---

# Step 6 — Test the app

The app should show fields for:
- Administrative pages and duration
- Informational pages and duration
- Product-related pages and duration
- Bounce rate
- Exit rate
- Page value
- Special-day score
- Month
- Visitor type
- Operating system code
- Browser code
- Region code
- Traffic type code
- Weekend status

Enter a sample session and click **Predict**.

A successful output should display:
- `Purchase` or `No Purchase`
- Purchase probability

Test at least **three different input combinations**.

For each test, confirm:
1. the app does not crash,
2. a prediction appears,
3. a probability appears,
4. categorical fields are accepted correctly.

---

# Step 7 — Stop here and review progress

At this point you will have completed the technical project through the Streamlit demo:

- final executed notebook,
- reproducible model comparison,
- saved trained model,
- working Streamlit interface,
- README / dependency information,
- dataset instructions,
- results files.

Do not move to the paper/video/final-link stage until the Streamlit app has been tested successfully.
