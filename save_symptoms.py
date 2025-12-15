import json
import pandas as pd

df = pd.read_csv("cleaned_symptoms.csv")

symptom_columns = [col for col in df.columns if col != "Disease"]
json.dump(symptom_columns, open("./model/symptom_columns.json", "w"))

print("symptom_columns.json saved!")
