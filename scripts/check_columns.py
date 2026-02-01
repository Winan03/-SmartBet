import pandas as pd
try:
    df = pd.read_csv('https://huggingface.co/datasets/JulienDelavande/soccer_stats/resolve/main/fbref_results.csv', encoding='latin-1')
    print("Columns:", df.columns.tolist())
    if 'home_sat' in df.columns:
        print("\nSample 'home_sat':")
        print(df['home_sat'].head(10))
        print("\nSample 'away_sat':")
        print(df['away_sat'].head(10))
    else:
        print("'home_sat' not found.")
except Exception as e:
    print(e)
