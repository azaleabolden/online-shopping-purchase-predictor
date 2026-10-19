# START HERE

This folder holds the final Online Shopping Purchase Predictor project. It continues the work from Practical Checkpoints 1 and 2.

## What is already complete

The project:
- loads the UCI Online Shoppers Purchasing Intention Dataset,
- investigates and removes 125 exact duplicate rows before splitting,
- treats `OperatingSystems`, `Browser`, `Region`, and `TrafficType` as categorical variables,
- uses an 80/20 stratified train/test split with `random_state=42`,
- compares a majority-class baseline with Logistic Regression, Decision Tree, and Random Forest,
- tests balanced class weighting,
- reports Accuracy, Precision, Recall, F1-score, FP, FN, and TP,
- selects the final model with 5-fold stratified cross-validation on the training set and tunes its settings (tuned Random Forest),
- checks how much the results depend on `PageValues`,
- includes a weak case example, sample predictions, and feature importance,
- includes a Streamlit demo that uses the final model.

## Final results

All models were tested on the same 2,441 held-out cases.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest, tuned (selected by cross-validation) | 0.8906 | 0.6316 | 0.7225 | 0.6740 |
| Random Forest, default settings | 0.9029 | 0.7617 | 0.5524 | 0.6404 |
| Logistic Regression + Balanced | 0.8460 | 0.5050 | 0.7958 | 0.6179 |
| Random Forest, tuned, without PageValues | 0.8202 | 0.4139 | 0.3586 | 0.3843 |

Cross-validation selected Random Forest (mean F1 0.6284). Tuning raised its cross-validation F1 to 0.6839 using balanced class weights, no depth limit, and a minimum of 3 samples per leaf. The tuned Random Forest is used for the Streamlit demo.

---

# Step 1. Open the project folder in VS Code

Use File, then Open Folder, and choose the project folder. Use the same Python environment that ran the notebook.

---

# Step 2. Install dependencies

Open a terminal in the project folder and run:

```powershell
python -m pip install -r requirements.txt
```

The project was developed with Python 3.10.11, scikit-learn 1.7.2, pandas 2.3.3, and NumPy 2.2.6.

---

# Step 3. Run the final notebook

Open `Online_Shopping_Purchase_Predictor_Final.ipynb`, then:

1. Restart the kernel.
2. Click **Run All**. The tuning step in Section 24A takes a few minutes.
3. Confirm there are no red errors.
4. Confirm Section 25 prints that the final model was saved and shows the package versions.
5. Confirm the last cell says `Model ready for Streamlit: True`.
6. Save the notebook.

This creates `artifacts/random_forest_pipeline.joblib`, which holds the tuned Random Forest and its preprocessing. The first run needs an internet connection because the notebook downloads the dataset.

---

# Step 4. Confirm the final results

In the notebook, confirm that Section 24A shows the cross-validation table with Random Forest selected, the best tuned settings, and the final test row for the tuned Random Forest (F1 0.6740). Confirm the full comparison in Section 12 still matches the default model results in the paper.

If the numbers change because the code or data changed, use the new executed results going forward.

---

# Step 5. Launch the Streamlit demo

From the project root, run:

```powershell
streamlit run src/app.py
```

A browser tab should open. If it does not, copy the Local URL from the terminal into your browser.

---

# Step 6. Test the app

The app should show fields for administrative, informational, and product-related pages and durations, bounce rate, exit rate, page value, special-day score, month, visitor type, operating system, browser, region, traffic type, and weekend status.

Enter a sample session and click **Predict**. A successful run shows `Purchase` or `No Purchase` and a purchase probability.

Test at least three different input combinations and confirm that the app does not crash, a prediction appears, a probability appears, and categorical fields are accepted.

---

# Step 7. Note about page value

Page value is tied to completed transactions, so it may not be available before a purchase occurs. The demo accepts it as a user-entered input, so it shows the model on completed session data and not on a live visitor. Removing page value lowered the test F1-score from 0.6740 to 0.3843.