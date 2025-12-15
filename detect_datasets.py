# detect_datasets.py
import os, glob, pandas as pd, json

ROOT = os.path.abspath(os.path.dirname(__file__))
print("Project folder:", ROOT)
csvs = glob.glob(os.path.join(ROOT, "*.csv")) + glob.glob(os.path.join(ROOT, "dataset", "*.csv"))
if not csvs:
    print("No CSV files found in project root or dataset/ directory.")
else:
    print("Found CSV files:")
    for p in csvs:
        print(" -", p)
        try:
            df = pd.read_csv(p, nrows=5)
            print("   Columns (first 20 shown):", df.columns.tolist()[:20])
            print("   First rows:")
            print(df.head(3).to_string(index=False))
        except Exception as e:
            print("   Could not read preview:", e)
print("\nAlso listing model/* files:")
model_dir = os.path.join(ROOT, "model")
if os.path.isdir(model_dir):
    print(os.listdir(model_dir))
else:
    print("model/ folder missing")
