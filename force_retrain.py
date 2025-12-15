# force_retrain.py
import os, sys, json, pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Try to locate symptom CSV (the proper symptom dataset you generated earlier)
candidates = [
    os.path.join(ROOT, "symptom_dataset.csv"),
    os.path.join(ROOT, "dataset", "symptom_dataset.csv"),
    os.path.join(ROOT, "dataset", "disease_symptoms.csv"),
    os.path.join(ROOT, "symptom_dataset.csv"),
]
csv_path = None
for c in candidates:
    if os.path.isfile(c):
        csv_path = c
        break

if csv_path is None:
    print("ERROR: Cannot find symptom CSV. Place 'symptom_dataset.csv' in medical_ai/ or medical_ai/dataset/ and re-run.")
    sys.exit(1)

print("Using CSV:", csv_path)

# Load dataset
df = pd.read_csv(csv_path)
# ensure column name is 'disease'
if "disease" not in df.columns and "Disease" in df.columns:
    df = df.rename(columns={"Disease": "disease"})
if "disease" not in df.columns:
    print("ERROR: CSV must contain 'disease' column. Columns found:", df.columns.tolist())
    sys.exit(1)

# Infer symptom columns (all except 'disease')
symptom_cols = [c for c in df.columns if c not in ("disease","Disease")]
if len(symptom_cols) < 5:
    print("WARNING: Found only", len(symptom_cols), "symptom columns. Are you using the correct symptom dataset?")
print("Detected symptom columns:", len(symptom_cols))

# Prepare X, y
X = df[symptom_cols].fillna(0)
# convert Yes/No/bools to 0/1 if necessary
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].replace({"Yes":1,"No":0,"yes":1,"no":0, True:1, False:0})
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)

y = df["disease"].astype(str)

# Filter out diseases with only 1 sample (optional)
vc = y.value_counts()
multi = vc[vc > 1].index
if len(multi) < len(vc):
    removed = len(y) - y.isin(multi).sum()
    print(f"Note: Removing {removed} rows where disease occurs only once to avoid stratify errors.")
    mask = y.isin(multi)
    X = X[mask]
    y = y[mask]

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.20, random_state=42)

# Train RandomForest
print("Training RandomForest on", X_train.shape[0], "samples and", X_train.shape[1], "features ...")
clf = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# quick eval
try:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
except Exception as e:
    print("Evaluation skipped:", e)

# Save artifacts (overwrite)
model_path = os.path.join(MODEL_DIR, "model.pkl")
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
symptom_json_path = os.path.join(MODEL_DIR, "symptom_columns.json")

with open(model_path, "wb") as f:
    pickle.dump(clf, f)
with open(le_path, "wb") as f:
    pickle.dump(le, f)
with open(symptom_json_path, "w", encoding="utf-8") as f:
    json.dump(symptom_cols, f, indent=2)

print("Saved model ->", model_path)
print("Saved label encoder ->", le_path)
print("Saved symptom columns ->", symptom_json_path)

# Verify shapes: model expects n_features_in_
expected = getattr(clf, "n_features_in_", None)
if expected is not None:
    print("Model n_features_in_:", expected)
    print("Symptom columns count:", len(symptom_cols))
    if expected != len(symptom_cols):
        print("WARNING: feature count mismatch AFTER training. Something is wrong.")
    else:
        print("Feature counts match. Good to run app.py now.")
else:
    print("Could not read model.n_features_in_. The model may not support this attribute.")

# create default precautions.json if missing
precautions_path = os.path.join(MODEL_DIR, "precautions.json")
if not os.path.isfile(precautions_path):
    diseases = sorted(list(set(y.astype(str).tolist())))
    prec_map = {d: [] for d in diseases}
    with open(precautions_path, "w", encoding="utf-8") as f:
        json.dump(prec_map, f, indent=2)
    print("Saved default precautions.json (empty lists) at", precautions_path)

print("\nDONE. Now run: python app.py")
