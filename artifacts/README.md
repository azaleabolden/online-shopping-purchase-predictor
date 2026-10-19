# Model Artifact

After running the final notebook from top to bottom, this folder should contain:

`random_forest_pipeline.joblib`

The file stores the fitted preprocessing pipeline and final Random Forest model together.

The Streamlit application at `src/app.py` loads this file.

If the file is missing, run the final notebook through the **Save the Final Model for Streamlit** section,
or run:

```bash
python src/train_and_evaluate.py
```
