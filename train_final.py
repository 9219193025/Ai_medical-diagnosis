import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
df = pd.read_csv("symptom_dataset.csv")
print("Dataset Loaded:", df.shape)

# --------------------------------------------------
# SEPARATE FEATURES AND LABEL
# --------------------------------------------------
X = df.drop("disease", axis=1)
y = df["disease"]

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --------------------------------------------------
# SAVE MODEL + ENCODER + SYMPTOMS
# --------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("symptom_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\nModel saved successfully!")
