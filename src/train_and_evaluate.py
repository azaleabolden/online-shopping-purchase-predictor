from pathlib import Path
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# 1. Load data
shopping_data = fetch_ucirepo(id=468)
X_raw = shopping_data.data.features.copy()
y_raw = shopping_data.data.targets.copy()
if isinstance(y_raw, pd.DataFrame):
    y_raw = y_raw.iloc[:, 0]

df = X_raw.copy()
df["Revenue"] = y_raw

# 2. Remove exact duplicates before split
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["Revenue"]).copy()
y = df["Revenue"].copy()

if y.dtype == bool:
    y = y.astype(int)
else:
    y = (
        y.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
    )

if y.isnull().any():
    raise ValueError("Unexpected Revenue values.")

# 3. Correct categorical handling
forced_categorical = ["OperatingSystems", "Browser", "Region", "TrafficType"]
auto_categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
categorical_features = list(dict.fromkeys(auto_categorical + forced_categorical))
numeric_features = [c for c in X.columns if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 4. Shared split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "Model": name,
        "Test Cases": len(y_test),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

models = {
    "Majority-Class Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "Logistic Regression + Balanced": Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ]),
    "Decision Tree": Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]),
    "Decision Tree + Balanced": Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )),
    ]),
    "Random Forest": Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ]),
    "Random Forest + Balanced": Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    ]),
}

results = []
for name, model in models.items():
    results.append(evaluate(name, model))

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "reproduced_model_comparison.csv", index=False)
print(results_df.round(4).to_string(index=False))

# Refit and save the selected final model: standard Random Forest.
final_model = models["Random Forest"]
final_model.fit(X_train, y_train)
joblib.dump(final_model, ARTIFACTS_DIR / "random_forest_pipeline.joblib")

print("\nSaved:")
print(RESULTS_DIR / "reproduced_model_comparison.csv")
print(ARTIFACTS_DIR / "random_forest_pipeline.joblib")
