# Results Summary

All configurations were evaluated on the same **2,441 held-out test cases**.

## Main Result
**Random Forest** had the strongest overall F1-score:

- Accuracy: **0.9029**
- Precision: **0.7617**
- Recall: **0.5524**
- F1-score: **0.6404**
- False positives: **66**
- False negatives: **171**
- True positives: **211**

## Recall-Focused Result
**Logistic Regression + Balanced** produced the highest Purchase recall:

- Accuracy: **0.8460**
- Precision: **0.5050**
- Recall: **0.7958**
- F1-score: **0.6179**
- False positives: **298**
- False negatives: **78**
- True positives: **304**

## Interpretation
The results show a clear precision–recall tradeoff. The balanced Logistic Regression identifies substantially more actual purchases than the standard Logistic Regression, but its precision falls because many more No Purchase sessions are labeled as Purchase.

Random Forest provides the strongest balance of the reported metrics and the highest F1-score, while balanced Logistic Regression is preferable when missing a potential purchase is considered more costly than generating a false positive.

## Weak Result to Show in the Paper/Demo
The selected Random Forest still misses **171 actual purchases** on the held-out test set. For the final paper/demo, use one concrete false-negative row from the executed notebook's weak-case output and explain that the available browsing features did not give the model enough evidence to classify that session as a Purchase.
