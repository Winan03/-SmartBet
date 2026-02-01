from datasets import load_dataset
import pandas as pd

try:
    print("Attempting to load 'JulienDelavande/soccer-stats'...")
    ds = load_dataset("JulienDelavande/soccer-stats", split="train")
except Exception as e:
    print(f"Failed with dash, trying underscore: {e}")
    try:
        ds = load_dataset("JulienDelavande/soccer_stats", split="train")
    except Exception as e2:
        print(f"Failed with underscore: {e2}")
        exit(1)

df = ds.to_pandas()
print("Columns:", df.columns.tolist())
print("First row:", df.iloc[0].to_dict())
