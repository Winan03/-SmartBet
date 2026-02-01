import pandas as pd

def load_soccer_data(split="train"):
    """
    Loads the JulienDelavande/soccer_stats dataset (fbref_results.csv).
    Handles encoding issues by trying latin-1 if utf-8 fails.
    Returns a pandas DataFrame.
    """
    base_url = "https://huggingface.co/datasets/JulienDelavande/soccer_stats/resolve/main/"
    files = {
        "results": "fbref_results.csv",
        "odds": "soccer_odds.csv"
    }
    
    target_url = base_url + files["results"]
    print(f"Loading main dataset from: {target_url}")
    
    try:
        return pd.read_csv(target_url, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, trying latin-1...")
        return pd.read_csv(target_url, encoding='latin-1')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raise e

def load_football_data(seasons=['2324', '2425'], leagues=['E0', 'SP1', 'I1', 'D1', 'F1']):
    """
    Loads data from football-data.co.uk which includes Corners (HC/AC) and Shots (HST/AST).
    
    Args:
        seasons: List of two-digit season codes (e.g., ['2324', '2425']).
        leagues: List of league codes (E0=EPL, SP1=LaLiga, I1=SerieA, D1=Bundesliga, F1=Ligue1).
    """
    base_url = "https://www.football-data.co.uk/mmz4281/"
    all_data = []
    
    for season in seasons:
        for league in leagues:
            url = f"{base_url}{season}/{league}.csv"
            try:
                print(f"Loading {league} {season} from {url}...")
                df = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
                # Add metadata
                df['season_code'] = season
                df['league_code'] = league
                all_data.append(df)
            except Exception as e:
                print(f"Failed to load {league} {season}: {e}")
                
    if not all_data:
        raise ValueError("No data could be loaded from football-data.co.uk")
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Standardize Column Names to match our pipeline expectations
    # Map football-data.co.uk columns to our internal names
    # Date -> date (needs parsing), HomeTeam -> home_team, AwayTeam -> away_team
    # FTHG -> home_g, FTAG -> away_g
    # HC -> home_corners, AC -> away_corners
    # HST -> home_shots_target, AST -> away_shots_target
    # HS -> home_shots, AS -> away_shots
    
    col_map = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_g',
        'FTAG': 'away_g',
        'HC': 'home_corners',
        'AC': 'away_corners',
        'HST': 'home_shots_ontarget',
        'AST': 'away_shots_ontarget',
        'HS': 'home_shots',
        'AS': 'away_shots',
        'B365H': 'odd_1',
        'B365D': 'odd_x',
        'B365A': 'odd_2'
    }
    
    full_df = full_df.rename(columns=col_map)
    
    # Parse Dates (DD/MM/YYYY)
    full_df['date'] = pd.to_datetime(full_df['date'], dayfirst=True, errors='coerce')
    
    # Drop rows without basic info
    full_df = full_df.dropna(subset=['home_team', 'away_team', 'home_g', 'away_g'])
    
    print(f"Loaded Enriched Data: {full_df.shape}")
    print("Columns available:", full_df.columns.tolist())
    
    return full_df
