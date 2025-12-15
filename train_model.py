import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("Disease_Symptom_Dataset.csv")

# Extract symptoms and disease columns
symptom_columns = [col for col in df.columns if col != "Disease"]

# Prepare data
X = df[symptom_columns]
y = df["Disease"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("./model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save symptom columns
with open("./model/symptom_columns.json", "w") as f:
    json.dump(symptom_columns, f)

print("Training complete! Model saved.")
