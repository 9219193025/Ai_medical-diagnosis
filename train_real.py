# train_real.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# --- 1. Load dataset (adjust filename if different) ---
df = pd.read_csv("dataset/disease_symptoms.csv")

# --- 2. Inspect columns quickly (print to console) ---
print("Columns:", df.columns.tolist()[:30])

# --- 3. Find disease/label column ---
# Common column names: 'disease', 'prognosis', 'Diagnosis'
label_col_candidates = ["disease","Disease","prognosis","Diagnosis","prognosis.1"]
label_col = None
for c in label_col_candidates:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    raise Exception("Could not find label column. Open CSV and set label_col variable.")

print("Using label column:", label_col)

# --- 4. Build feature matrix X and labels y ---
# Many Kaggle datasets already have symptoms as binary columns (0/1).
# If dataset has a 'symptoms' text field rather than columns, you'll need text -> one-hot preprocessing.
# Here we assume symptom columns are columns other than the label.
X = df.drop(columns=[label_col])
y = df[label_col].astype(str)

# If there are non-feature columns (like 'precautions', 'description') drop them:
non_feature_cols = [c for c in X.columns if X[c].dtype == 'O' and X[c].nunique() > 50]
# keep this simple: drop obvious non-binary text columns if any
for c in non_feature_cols:
    print("Dropping non-feature/text column:", c)
    X = X.drop(columns=[c])

# Ensure features are numeric (0/1). If not, try to convert booleans/true-false/yes-no.
X = X.fillna(0)
for col in X.columns:
    if X[col].dtype == object:
        # try map common values
        X[col] = X[col].replace({"Yes":1,"No":0,"yes":1,"no":0, True:1, False:0})
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)

print("Feature columns count:", X.shape[1])

# --- 5. Encode labels ---
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --- 6. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.15, random_state=42
)

# --- 7. Model training ---
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- 8. Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy (top-1): {acc:.4f}")

# Top-3 accuracy
try:
    top3 = top_k_accuracy_score(y_test, model.predict_proba(X_test), k=3)
    print(f"Top-3 accuracy: {top3:.4f}")
except Exception as e:
    print("Top-k accuracy not computed:", e)

# --- 9. Save model, encoder, and symptom columns order ---
import os
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("model/symptom_cols.pkl", "wb"))

print("Saved model, label encoder, and symptom columns to ./model/")
