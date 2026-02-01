import pandas as pd
import numpy as np
from datetime import timedelta

def compute_rolling_averages(df, team_col='team', date_col='date', stats_cols=['goals'], window=5):
    """
    Calculates rolling averages for specified stats.
    FIX: Uses shift(1) to avoid data leakage (excludes current match).
    """
    df = df.sort_values([team_col, date_col])
    
    for col in stats_cols:
        # Shift by 1 to exclude current match (prevent leakage)
        df[f'rolling_{col}'] = df.groupby(team_col)[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        
    return df

def convert_to_team_centric(df, match_id_col='match_id', date_col='date', 
                           home_team='home_team', away_team='away_team',
                           home_goals='home_goals', away_goals='away_goals',
                           home_corners='home_corners', away_corners='away_corners',
                           home_shots='home_shots', away_shots='away_shots',
                           home_shots_ot='home_shots_ontarget', away_shots_ot='away_shots_ontarget'):
    """
    Converts match-centric DF to team-centric DF (2 rows per match).
    """
    # Home perspective
    home = df.copy()
    home['team'] = home[home_team]
    home['opponent'] = home[away_team]
    home['goals_for'] = home[home_goals]
    home['goals_against'] = home[away_goals]
    home['is_home'] = 1
    
    if home_corners in df.columns:
        home['corners_for'] = home[home_corners]
        home['corners_against'] = home[away_corners]
    
    if home_shots in df.columns:
        home['shots_for'] = home[home_shots]
        home['shots_against'] = home[away_shots]
        
    if home_shots_ot in df.columns:
        home['shots_ot_for'] = home[home_shots_ot]
        home['shots_ot_against'] = home[away_shots_ot]
    
    # Away perspective
    away = df.copy()
    away['team'] = away[away_team]
    away['opponent'] = away[home_team]
    away['goals_for'] = away[away_goals]
    away['goals_against'] = away[home_goals]
    away['is_home'] = 0
    
    if away_corners in df.columns:
        away['corners_for'] = away[away_corners]
        away['corners_against'] = away[home_corners]
        
    if away_shots in df.columns:
        away['shots_for'] = away[away_shots]
        away['shots_against'] = away[home_shots]
        
    if away_shots_ot in df.columns:
        away['shots_ot_for'] = away[away_shots_ot]
        away['shots_ot_against'] = away[home_shots_ot]
    
    combined = pd.concat([home, away], axis=0).sort_values([date_col, match_id_col])
    return combined

def calculate_recent_form(team_df, team_col='team', date_col='date'):
    """
    Calculates rolling points (Form) for the last 5 games.
    Win=3, Draw=1, Loss=0.
    FIX: Uses shift(1) to prevent leakage.
    """
    def get_points(row):
        if row['goals_for'] > row['goals_against']: 
            return 3
        elif row['goals_for'] == row['goals_against']: 
            return 1
        else: 
            return 0
        
    team_df['match_points'] = team_df.apply(get_points, axis=1)
    team_df = team_df.sort_values([team_col, date_col])
    
    # Shift by 1 to use only PRE-MATCH form
    team_df['rolling_form_5'] = team_df.groupby(team_col)['match_points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()
    )
    
    return team_df

def calculate_rolling_strength(team_df, team_col='team', date_col='date'):
    """
    Calculates efficiency metrics.
    """
    # Shot efficiency (Goals / Shots)
    team_df['rolling_shot_efficiency'] = (
        team_df['rolling_goals_for'] / 
        team_df['rolling_shots_for'].replace(0, np.nan)
    ).fillna(0)
    
    # Defense efficiency (lower is better)
    team_df['rolling_defense_efficiency'] = (
        team_df['rolling_goals_against'] / 
        team_df['rolling_shots_against'].replace(0, np.nan)
    ).fillna(0)
    
    return team_df

# ================================================================================
# NEW ADVANCED FEATURES
# ================================================================================

def calculate_head_to_head_stats(df, team_col='team', opponent_col='opponent', 
                                 date_col='date', lookback_matches=5):
    """
    Calculates head-to-head statistics between teams.
    """
    df = df.sort_values([team_col, opponent_col, date_col])
    
    # H2H wins in last N meetings
    df['h2h_wins'] = df.groupby([team_col, opponent_col])['match_points'].transform(
        lambda x: (x.shift(1).rolling(lookback_matches, min_periods=1).apply(
            lambda pts: (pts == 3).sum()
        ))
    )
    
    # H2H goals scored
    df['h2h_goals_scored'] = df.groupby([team_col, opponent_col])['goals_for'].transform(
        lambda x: x.shift(1).rolling(lookback_matches, min_periods=1).mean()
    )
    
    return df

def calculate_momentum_features(team_df, team_col='team', date_col='date'):
    """
    Calculates momentum: recent form trend (improving/declining).
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Last 3 games form
    team_df['form_last3'] = team_df.groupby(team_col)['match_points'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
    )
    
    # Previous 3 games form (games 4-6)
    team_df['form_prev3'] = team_df.groupby(team_col)['match_points'].transform(
        lambda x: x.shift(4).rolling(window=3, min_periods=1).sum()
    )
    
    # Momentum = recent form - previous form (positive = improving)
    team_df['momentum'] = team_df['form_last3'] - team_df['form_prev3'].fillna(0)
    
    return team_df

def calculate_rest_days(df, team_col='team', date_col='date'):
    """
    Calculates days of rest since last match.
    """
    df = df.sort_values([team_col, date_col])
    
    df['last_match_date'] = df.groupby(team_col)[date_col].shift(1)
    df['rest_days'] = (df[date_col] - df['last_match_date']).dt.days
    df['rest_days'] = df['rest_days'].fillna(7)  # Default to 1 week
    
    # Flag congested schedule (<3 days rest)
    df['is_congested'] = (df['rest_days'] < 3).astype(int)
    
    return df

def calculate_home_away_split_stats(team_df, team_col='team', date_col='date', window=10):
    """
    Separate home and away performance statistics.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Home performance
    home_mask = team_df['is_home'] == 1
    team_df.loc[home_mask, 'home_goals_avg'] = team_df.loc[home_mask].groupby(team_col)['goals_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    # Away performance
    away_mask = team_df['is_home'] == 0
    team_df.loc[away_mask, 'away_goals_avg'] = team_df.loc[away_mask].groupby(team_col)['goals_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    # Fill missing values with overall average
    team_df['home_goals_avg'] = team_df.groupby(team_col)['home_goals_avg'].transform(
        lambda x: x.ffill().bfill()
    )
    team_df['away_goals_avg'] = team_df.groupby(team_col)['away_goals_avg'].transform(
        lambda x: x.ffill().bfill()
    )
    
    return team_df

def calculate_season_phase(df, date_col='date'):
    """
    Adds season phase indicator (early, mid, late season).
    """
    # Extract month
    df['month'] = df[date_col].dt.month
    
    # Season phases (European calendar: Aug-May)
    # Early: Aug-Oct (8-10)
    # Mid: Nov-Feb (11-2)
    # Late: Mar-May (3-5)
    def get_phase(month):
        if month in [8, 9, 10]:
            return 'early'
        elif month in [11, 12, 1, 2]:
            return 'mid'
        else:
            return 'late'
    
    df['season_phase'] = df['month'].apply(get_phase)
    
    # One-hot encode
    df = pd.get_dummies(df, columns=['season_phase'], prefix='phase', drop_first=True)
    
    return df

def calculate_scoring_patterns(team_df, team_col='team', date_col='date', window=10):
    """
    Calculates scoring pattern metrics.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Clean sheets (no goals conceded)
    team_df['clean_sheet'] = (team_df['goals_against'] == 0).astype(int)
    team_df['clean_sheet_rate'] = team_df.groupby(team_col)['clean_sheet'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    # Failed to score
    team_df['failed_to_score'] = (team_df['goals_for'] == 0).astype(int)
    team_df['scoring_drought_rate'] = team_df.groupby(team_col)['failed_to_score'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    # High-scoring games (3+ goals)
    team_df['high_scoring'] = (team_df['goals_for'] >= 3).astype(int)
    team_df['high_scoring_rate'] = team_df.groupby(team_col)['high_scoring'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    
    return team_df

def calculate_consistency_metrics(team_df, team_col='team', date_col='date', window=10):
    """
    Calculates performance consistency (variance in recent results).
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Variance in goals scored
    team_df['goals_variance'] = team_df.groupby(team_col)['goals_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).std()
    ).fillna(0)
    
    # Variance in points
    team_df['points_variance'] = team_df.groupby(team_col)['match_points'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).std()
    ).fillna(0)
    
    return team_df

class EloTracker:
    """Enhanced Elo with K-factor adjustments"""
    def __init__(self, k_factor=20, init_rating=1500):
        self.k_factor = k_factor
        self.ratings = {}
        self.init_rating = init_rating
        self.match_count = {}

    def get_rating(self, team):
        return self.ratings.get(team, self.init_rating)
    
    def get_k_factor(self, team):
        """Reduce K-factor for experienced teams (more stable ratings)"""
        matches = self.match_count.get(team, 0)
        if matches < 10:
            return self.k_factor * 1.5  # New teams: higher volatility
        elif matches < 30:
            return self.k_factor
        else:
            return self.k_factor * 0.75  # Established teams: lower volatility

    def update_ratings(self, home_team, away_team, result, margin=0):
        """
        Updates Elo ratings.
        result: 1 (Home Win), 0.5 (Draw), 0 (Away Win)
        margin: Goal difference (optional bonus)
        """
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        
        # Track match count
        self.match_count[home_team] = self.match_count.get(home_team, 0) + 1
        self.match_count[away_team] = self.match_count.get(away_team, 0) + 1
        
        # Expected score
        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        
        # K-factor adjustments
        k_home = self.get_k_factor(home_team)
        k_away = self.get_k_factor(away_team)
        
        # Margin multiplier (big wins matter more)
        margin_multiplier = 1 + abs(margin) * 0.1 if abs(margin) > 1 else 1
        
        new_home = r_home + k_home * margin_multiplier * (result - expected_home)
        new_away = r_away + k_away * margin_multiplier * ((1 - result) - (1 - expected_home))
        
        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away
        
        return new_home, new_away

def apply_all_advanced_features(team_df, team_col='team', opponent_col='opponent', 
                                date_col='date', window=10):
    """
    Applies all advanced feature engineering in one pipeline.
    """
    print("Applying advanced features...")
    
    # 1. Head-to-head
    print("  - Head-to-head stats")
    team_df = calculate_head_to_head_stats(team_df, team_col, opponent_col, date_col)
    
    # 2. Momentum
    print("  - Momentum features")
    team_df = calculate_momentum_features(team_df, team_col, date_col)
    
    # 3. Rest days
    print("  - Rest days calculation")
    team_df = calculate_rest_days(team_df, team_col, date_col)
    
    # 4. Home/Away splits
    print("  - Home/Away performance splits")
    team_df = calculate_home_away_split_stats(team_df, team_col, date_col, window)
    
    # 5. Scoring patterns
    print("  - Scoring patterns")
    team_df = calculate_scoring_patterns(team_df, team_col, date_col, window)
    
    # 6. Consistency
    print("  - Consistency metrics")
    team_df = calculate_consistency_metrics(team_df, team_col, date_col, window)
    
    return team_df
