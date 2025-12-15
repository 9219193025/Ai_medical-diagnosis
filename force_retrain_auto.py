# force_retrain_auto.py
import os, glob, json, pickle, sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.dirname(__file__))
candidates = glob.glob(os.path.join(ROOT, "*.csv")) + glob.glob(os.path.join(ROOT, "dataset", "*.csv"))
print("CSV candidates:", candidates)

best = None
best_sym_count = -1
for c in candidates:
    try:
        df = pd.read_csv(c, nrows=5)  # just read header
        cols = df.columns.tolist()
        # guess: symptom dataset will have many columns and a 'disease' or 'Disease' column
        if any(x.lower() == 'disease' for x in cols) and len(cols) >= 20:
            sym_count = len(cols) - 1
            if sym_count > best_sym_count:
                best_sym_count = sym_count
                best = c
    except Exception:
        continue

if best is None:
    print("ERROR: Could not find a symptom CSV (one with a 'disease' column and many symptom columns).")
    print("CSV candidates and their columns:")
    for c in candidates:
        try:
            df = pd.read_csv(c, nrows=2)
            print(c, "->", df.columns.tolist())
        except Exception as e:
            print(c, "-> cannot read:", e)
    sys.exit(1)

print("Using CSV for training:", best)
df = pd.read_csv(best)
# normalize 'Disease' column name
if 'disease' not in df.columns and 'Disease' in df.columns:
    df = df.rename(columns={'Disease':'disease'})
if 'disease' not in df.columns:
    print("ERROR: chosen CSV does not have 'disease' column.")
    sys.exit(1)

# symptom columns = all except 'disease'
symptom_cols = [c for c in df.columns if c.lower() != 'disease']
print("Detected symptom columns count:", len(symptom_cols))

X = df[symptom_cols].fillna(0)
# coerce to numeric 0/1
for col in X.columns:
    if X[col].dtype == object:
        X[col] = X[col].replace({"Yes":1,"No":0,"yes":1,"no":0, True:1, False:0})
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)

y = df['disease'].astype(str)

# filter single-sample classes
vc = y.value_counts()
multi = vc[vc > 1].index
if len(multi) < len(vc):
    print("Removing classes with only 1 sample:", set(vc[vc == 1].index.tolist()))
    mask = y.isin(multi)
    X = X[mask]
    y = y[mask]

le = LabelEncoder()
y_enc = le.fit_transform(y)

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

print("Training on:", X_train.shape)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# save
os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)
with open(os.path.join(ROOT, "model", "model.pkl"), "wb") as f:
    pickle.dump(clf, f)
with open(os.path.join(ROOT, "model", "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(ROOT, "model", "symptom_columns.json"), "w", encoding="utf-8") as f:
    json.dump(symptom_cols, f, indent=2)

print("Saved model and artifacts to model/ .")
print("Model n_features_in_:", getattr(clf, "n_features_in_", "unknown"))
print("Symptom columns count:", len(symptom_cols))
print("DONE. Now run: python app.py")
