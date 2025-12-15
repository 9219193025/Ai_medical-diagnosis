# create_model_artifacts.py
import os
import json
import pickle
import glob
import sys

try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except Exception as e:
    print("Missing packages in your venv. Run: pip install pandas numpy scikit-learn")
    raise

ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Find symptom CSV
candidates = [
    os.path.join(ROOT, "symptom_dataset.csv"),
    os.path.join(ROOT, "dataset", "symptom_dataset.csv"),
    os.path.join(ROOT, "dataset", "disease_symptoms.csv"),
    os.path.join(ROOT, "symptom_dataset.csv"),
    os.path.join(ROOT, "..", "symptom_dataset.csv"),
]
csv_path = None
for c in candidates:
    if os.path.isfile(c):
        csv_path = c
        break

if csv_path is None:
    print("ERROR: Could not find symptom CSV. Place 'symptom_dataset.csv' in the medical_ai folder or dataset/ and re-run.")
    sys.exit(1)

print("Found CSV:", csv_path)

# 2) Load dataset
df = pd.read_csv(csv_path)
if "disease" not in df.columns and "Disease" in df.columns:
    df = df.rename(columns={"Disease": "disease"})
if "disease" not in df.columns:
    print("ERROR: CSV must have a 'disease' column. Columns found:", df.columns.tolist())
    sys.exit(1)

# 3) Determine symptom columns (all except 'disease')
symptom_cols = [c for c in df.columns if c.lower() not in ("disease", "diseases")]
if len(symptom_cols) == 0:
    print("ERROR: No symptom columns detected. Make sure CSV has many symptom columns (0/1) and a 'disease' column.")
    sys.exit(1)

# Save symptom columns JSON
symptom_json_path = os.path.join(MODEL_DIR, "symptom_columns.json")
with open(symptom_json_path, "w", encoding="utf-8") as f:
    json.dump(symptom_cols, f, indent=2)
print("Saved symptom columns to:", symptom_json_path, " (count =", len(symptom_cols), ")")

# 4) Load or train model
model_path = os.path.join(MODEL_DIR, "model.pkl")
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

if os.path.isfile(model_path) and os.path.isfile(le_path):
    print("Model and label encoder already exist. Skipping training.")
else:
    print("Training model (this may take ~30-120s depending on your machine)...")
    X = df[symptom_cols].fillna(0)
    # Ensure features numeric 0/1
    for col in X.columns:
        X[col] = pd.to_numeric(X[col].replace({"Yes":1,"No":0,"yes":1,"no":0, True:1, False:0}), errors='coerce').fillna(0).astype(int)

    y = df["disease"].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # If some classes have only 1 sample, remove them to avoid stratify error
    vc = pd.Series(y).value_counts()
    valid = vc[vc > 1].index.tolist()
    if len(valid) < len(vc):
        keep_mask = df["disease"].isin(valid)
        print(f"Removing {len(df) - keep_mask.sum()} rows with diseases that appear only once.")
        X = X[keep_mask]
        y_enc = le.transform(df["disease"][keep_mask])
        # Recreate encoder on filtered labels
        le = LabelEncoder()
        y_enc = le.fit_transform(df["disease"][keep_mask].astype(str))

    # train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.20, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # evaluate quickly
    try:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Quick eval accuracy: {acc:.4f}")
    except Exception:
        print("Model trained. (No eval performed.)")

    # save model and encoder
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print("Model and label encoder saved to model/")

# 5) Prepare precautions mapping (precautions.json)
precautions_path = os.path.join(MODEL_DIR, "precautions.json")
# If medical_data.json exists in project, extract mapping
medical_data_candidates = [
    os.path.join(ROOT, "medical_data.json"),
    os.path.join(ROOT, "model", "medical_data.json"),
    os.path.join(ROOT, "..", "medical_data.json"),
]
medical_data_file = None
for m in medical_data_candidates:
    if os.path.isfile(m):
        medical_data_file = m
        break

if medical_data_file:
    print("Found medical_data.json:", medical_data_file)
    try:
        med = json.load(open(medical_data_file, "r", encoding="utf-8"))
        # build precautions mapping (disease -> list)
        prec_map = {}
        for k, v in med.items():
            prec_map[k] = v.get("precautions", []) if isinstance(v, dict) else []
        with open(precautions_path, "w", encoding="utf-8") as f:
            json.dump(prec_map, f, indent=2)
        print("Saved precautions.json from medical_data.json")
    except Exception as e:
        print("Failed to extract medical_data.json:", e)
else:
    # create default empty precautions for all diseases in dataset/label encoder
    print("No medical_data.json found. Creating default precautions.json.")
    # load diseases from CSV
    diseases = sorted(df["disease"].astype(str).unique().tolist())
    prec_map = {d: [] for d in diseases}
    with open(precautions_path, "w", encoding="utf-8") as f:
        json.dump(prec_map, f, indent=2)
    print("Saved default precautions.json with", len(diseases), "diseases.")

print("\nALL DONE — artifacts are ready in the 'model/' folder:")
for fname in os.listdir(MODEL_DIR):
    print(" -", fname)
print("\nNow run: python app.py")
