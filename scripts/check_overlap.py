import pandas as pd
import sys

# URLs
RESULTS_URL = "https://huggingface.co/datasets/JulienDelavande/soccer_stats/resolve/main/fbref_results.csv"
ODDS_URL = "https://huggingface.co/datasets/JulienDelavande/soccer_stats/resolve/main/soccer_odds.csv"

def inspect_overlap():
    print("Loading datasets...")
    try:
        results = pd.read_csv(RESULTS_URL, encoding='utf-8')
    except:
        results = pd.read_csv(RESULTS_URL, encoding='latin-1')
        
    try:
        odds = pd.read_csv(ODDS_URL, encoding='utf-8')
    except:
        odds = pd.read_csv(ODDS_URL, encoding='latin-1')

    print(f"Results Shape: {results.shape}")
    print(f"Odds Shape: {odds.shape}")
    
    # Columns
    print("\nResults Columns:", results.columns.tolist())
    print("Odds Columns:", odds.columns.tolist())
    
    # Check ID Overlap
    # Results usually has 'game_id' or 'id' or 'match_id'
    # Odds has 'match_id'
    
    r_id_col = 'game_id' if 'game_id' in results.columns else 'id'
    if r_id_col not in results.columns:
        # User renamed it in notebook but CSV has original names
        # User said: "game_id, league..." in snippet
        # Let's find columns that look like IDs
        possible = [c for c in results.columns if 'id' in c.lower()]
        print(f"Possible ID columns in Results: {possible}")
        if possible:
            r_id_col = possible[0]
            
    if r_id_col in results.columns and 'match_id' in odds.columns:
        len_r = len(set(results[r_id_col].dropna().astype(str)))
        len_o = len(set(odds['match_id'].dropna().astype(str)))
        overlap = len(set(results[r_id_col].dropna().astype(str)).intersection(set(odds['match_id'].dropna().astype(str))))
        print(f"\nID Overlap ({r_id_col} vs match_id): {overlap} (Results unique: {len_r}, Odds unique: {len_o})")
    else:
        print("\nCould not check ID overlap (columns missing).")
        
    # Check Team Overlap
    # Results: 'home_team'
    # Odds: 'home_team'
    if 'home_team' in results.columns and 'home_team' in odds.columns:
        r_teams = set(results['home_team'].unique())
        o_teams = set(odds['home_team'].unique())
        team_overlap = len(r_teams.intersection(o_teams))
        print(f"Team Name Overlap: {team_overlap} (Results unique: {len(r_teams)}, Odds unique: {len(o_teams)})")
        print("Sample Results Teams:", list(r_teams)[:5])
        print("Sample Odds Teams:", list(o_teams)[:5])
        
    # Check Date Overlap
    # Results: 'date'
    # Odds: 'commence_time' -> date
    if 'date' in results.columns and 'commence_time' in odds.columns:
        results['date_dt'] = pd.to_datetime(results['date'], errors='coerce')
        odds['date_dt'] = pd.to_datetime(odds['commence_time'], errors='coerce')
        
        r_min, r_max = results['date_dt'].min(), results['date_dt'].max()
        o_min, o_max = odds['date_dt'].min(), odds['date_dt'].max()
        
        print(f"\nResults Date Range: {r_min} to {r_max}")
        print(f"Odds Date Range:    {o_min} to {o_max}")

if __name__ == "__main__":
    inspect_overlap()
