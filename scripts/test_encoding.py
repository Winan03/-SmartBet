import pandas as pd
import sys

URL_BASE = "https://huggingface.co/datasets/JulienDelavande/soccer_stats/resolve/main/"
FILES_TO_TRY = ["soccer_odds.csv", "fbref_results.csv", "sofifa_teams_stats.csv"] # Guesses based on search results

def test_encoding(file_name):
    url = URL_BASE + file_name
    print(f"Testing {url}...")
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    
    for enc in encodings:
        try:
            print(f"  Trying encoding: {enc}")
            df = pd.read_csv(url, encoding=enc, nrows=5)
            print(f"  SUCCESS with {enc}!")
            return enc
        except Exception as e:
            print(f"  Failed with {enc}: {e}")
    return None

if __name__ == "__main__":
    # We will accept file names from args or use defaults
    pass
