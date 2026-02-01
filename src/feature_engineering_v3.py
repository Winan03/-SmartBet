"""
SmartBet Feature Engineering V3
-------------------------------
Features avanzadas para predicción multi-mercado con alta precisión.
Incluye: corners, remates, xG, defensiva, contexto táctico.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


# =============================================================================
# CORNERS FEATURES
# =============================================================================

def compute_corners_features(team_df, team_col='team', date_col='date', window=5):
    """
    Calcula features avanzadas de corners para predicción de mercados +6.5, +7.5, etc.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Rolling corners for/against
    team_df['rolling_corners_for'] = team_df.groupby(team_col)['corners_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    team_df['rolling_corners_against'] = team_df.groupby(team_col)['corners_against'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # Corner dominance (ratio propio vs concedido)
    team_df['corner_dominance'] = team_df['rolling_corners_for'] / (team_df['rolling_corners_against'] + 0.5)
    
    # Total corners tendency (para over/under)
    team_df['rolling_total_corners'] = team_df['rolling_corners_for'] + team_df['rolling_corners_against']
    
    # Corners en casa vs visita
    team_df['corners_home_flag'] = team_df['is_home']
    
    # Varianza de corners (consistencia)
    team_df['corners_variance'] = team_df.groupby(team_col)['corners_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).std()
    )
    
    return team_df


def compute_corners_first_half_estimate(team_df, first_half_ratio=0.48):
    """
    Estima corners del primer tiempo basado en ratio histórico.
    Por defecto, ~48% de los corners ocurren en el primer tiempo.
    """
    team_df['estimated_corners_1h_for'] = team_df['rolling_corners_for'] * first_half_ratio
    team_df['estimated_corners_1h_against'] = team_df['rolling_corners_against'] * first_half_ratio
    team_df['estimated_total_corners_1h'] = team_df['estimated_corners_1h_for'] + team_df['estimated_corners_1h_against']
    
    return team_df


# =============================================================================
# SHOTS FEATURES
# =============================================================================

def compute_shots_features(team_df, team_col='team', date_col='date', window=5):
    """
    Calcula features de remates para mercados de shots on target.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Rolling shots totales
    team_df['rolling_shots_for'] = team_df.groupby(team_col)['shots_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    team_df['rolling_shots_against'] = team_df.groupby(team_col)['shots_against'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # Rolling shots on target
    if 'shots_ot_for' in team_df.columns:
        team_df['rolling_shots_ot_for'] = team_df.groupby(team_col)['shots_ot_for'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        team_df['rolling_shots_ot_against'] = team_df.groupby(team_col)['shots_ot_against'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        
        # Shot accuracy (remates al arco / remates totales)
        team_df['shot_accuracy'] = team_df['rolling_shots_ot_for'] / (team_df['rolling_shots_for'] + 0.5)
    
    # Total shots tendency
    team_df['rolling_total_shots'] = team_df['rolling_shots_for'] + team_df['rolling_shots_against']
    
    # Shot conversion rate (goles / remates)
    team_df['shot_conversion'] = team_df['rolling_goals_for'] / (team_df['rolling_shots_for'] + 0.5)
    
    return team_df


# =============================================================================
# XG FEATURES MEJORADOS
# =============================================================================

def compute_xg_advanced(team_df, team_col='team', date_col='date', window=5):
    """
    Features de xG mejorados incluyendo sobrerendimiento y eficiencia.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    if 'xg_for' in team_df.columns:
        # Rolling xG
        team_df['rolling_xg_for'] = team_df.groupby(team_col)['xg_for'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        team_df['rolling_xg_against'] = team_df.groupby(team_col)['xg_against'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        
        # xG per shot
        team_df['xg_per_shot'] = team_df['rolling_xg_for'] / (team_df['rolling_shots_for'] + 0.5)
        
        # Overperformance (goles reales vs esperados)
        team_df['xg_overperformance'] = team_df['rolling_goals_for'] - team_df['rolling_xg_for']
        
        # xG differential
        team_df['xg_differential'] = team_df['rolling_xg_for'] - team_df['rolling_xg_against']
    
    return team_df


# =============================================================================
# DEFENSIVE WEAKNESS FEATURES
# =============================================================================

def compute_defensive_features(team_df, team_col='team', date_col='date', window=5):
    """
    Features de vulnerabilidad defensiva para predecir goles concedidos.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Goals conceded per shot (vulnerabilidad)
    team_df['goals_conceded_per_shot'] = team_df['rolling_goals_against'] / (team_df['rolling_shots_against'] + 0.5)
    
    # Clean sheet rate
    team_df['clean_sheet'] = (team_df['goals_against'] == 0).astype(int)
    team_df['clean_sheet_rate'] = team_df.groupby(team_col)['clean_sheet'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # Goals conceded streak (partidos consecutivos recibiendo goles)
    def calc_conceding_streak(s):
        result = []
        streak = 0
        for val in s:
            if val > 0:
                streak += 1
            else:
                streak = 0
            result.append(streak)
        return result
    
    team_df['conceding_streak'] = team_df.groupby(team_col)['goals_against'].transform(
        lambda x: pd.Series(calc_conceding_streak(x.values), index=x.index)
    ).shift(1)
    
    return team_df


# =============================================================================
# SCORING PATTERNS & BTTS FEATURES
# =============================================================================

def compute_scoring_patterns(team_df, team_col='team', date_col='date', window=5):
    """
    Features de patrones de anotación para BTTS y over/under.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Scoring streak (partidos consecutivos anotando)
    def calc_scoring_streak(s):
        result = []
        streak = 0
        for val in s:
            if val > 0:
                streak += 1
            else:
                streak = 0
            result.append(streak)
        return result
    
    team_df['scoring_streak'] = team_df.groupby(team_col)['goals_for'].transform(
        lambda x: pd.Series(calc_scoring_streak(x.values), index=x.index)
    ).shift(1)
    
    # Failed to score rate
    team_df['failed_to_score'] = (team_df['goals_for'] == 0).astype(int)
    team_df['failed_to_score_rate'] = team_df.groupby(team_col)['failed_to_score'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # BTTS history
    team_df['btts_occurred'] = ((team_df['goals_for'] > 0) & (team_df['goals_against'] > 0)).astype(int)
    team_df['btts_rate'] = team_df.groupby(team_col)['btts_occurred'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # Over 2.5 history
    team_df['over_25_occurred'] = (team_df['goals_for'] + team_df['goals_against'] > 2.5).astype(int)
    team_df['over_25_rate'] = team_df.groupby(team_col)['over_25_occurred'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # Over 1.5 history
    team_df['over_15_occurred'] = (team_df['goals_for'] + team_df['goals_against'] > 1.5).astype(int)
    team_df['over_15_rate'] = team_df.groupby(team_col)['over_15_occurred'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    return team_df


# =============================================================================
# FORM & MOMENTUM AVANZADOS
# =============================================================================

def compute_advanced_form(team_df, team_col='team', date_col='date'):
    """
    Features de forma y momentum mejorados.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Points calculation
    def get_points(row):
        if row['goals_for'] > row['goals_against']:
            return 3
        elif row['goals_for'] == row['goals_against']:
            return 1
        return 0
    
    team_df['points'] = team_df.apply(get_points, axis=1)
    
    # Points last 5 and 10
    team_df['points_last_5'] = team_df.groupby(team_col)['points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).sum()
    )
    team_df['points_last_10'] = team_df.groupby(team_col)['points'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).sum()
    )
    
    # Goal trend (creciente vs decreciente)
    team_df['goals_last_3'] = team_df.groupby(team_col)['goals_for'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )
    team_df['goals_last_6'] = team_df.groupby(team_col)['goals_for'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=3).mean()
    )
    team_df['goals_trend'] = team_df['goals_last_3'] - team_df['goals_last_6']
    
    # Momentum (diferencia entre forma reciente y antigua)
    team_df['form_momentum'] = (team_df['points_last_5'] / 5) - (team_df['points_last_10'] / 10)
    
    return team_df


# =============================================================================
# HOME/AWAY SPLITS
# =============================================================================

def compute_home_away_splits(team_df, team_col='team', date_col='date', window=5):
    """
    Estadísticas separadas para partidos en casa vs visita.
    """
    team_df = team_df.sort_values([team_col, date_col])
    
    # Separate home and away
    home_games = team_df[team_df['is_home'] == 1].copy()
    away_games = team_df[team_df['is_home'] == 0].copy()
    
    # Home stats
    home_games['home_goals_avg'] = home_games.groupby(team_col)['goals_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    home_games['home_conceded_avg'] = home_games.groupby(team_col)['goals_against'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    home_games['home_corners_avg'] = home_games.groupby(team_col)['corners_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # Away stats
    away_games['away_goals_avg'] = away_games.groupby(team_col)['goals_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    away_games['away_conceded_avg'] = away_games.groupby(team_col)['goals_against'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    away_games['away_corners_avg'] = away_games.groupby(team_col)['corners_for'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    
    # Merge back
    home_cols = ['home_goals_avg', 'home_conceded_avg', 'home_corners_avg']
    away_cols = ['away_goals_avg', 'away_conceded_avg', 'away_corners_avg']
    
    for col in home_cols:
        team_df[col] = np.nan
        team_df.loc[home_games.index, col] = home_games[col]
        
    for col in away_cols:
        team_df[col] = np.nan
        team_df.loc[away_games.index, col] = away_games[col]
    
    # Forward fill for missing values
    for col in home_cols + away_cols:
        team_df[col] = team_df.groupby(team_col)[col].transform(lambda x: x.ffill())
    
    return team_df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def apply_all_v3_features(team_df, team_col='team', date_col='date', window=5):
    """
    Aplica todas las features V3 en un pipeline unificado.
    """
    print("Applying V3 Advanced Features...")
    
    # 1. Corners
    print("  - Corners features")
    team_df = compute_corners_features(team_df, team_col, date_col, window)
    team_df = compute_corners_first_half_estimate(team_df)
    
    # 2. Shots
    print("  - Shots features")
    team_df = compute_shots_features(team_df, team_col, date_col, window)
    
    # 3. xG
    print("  - xG advanced features")
    team_df = compute_xg_advanced(team_df, team_col, date_col, window)
    
    # 4. Defensive
    print("  - Defensive weakness features")
    team_df = compute_defensive_features(team_df, team_col, date_col, window)
    
    # 5. Scoring patterns
    print("  - Scoring pattern features")
    team_df = compute_scoring_patterns(team_df, team_col, date_col, window)
    
    # 6. Form
    print("  - Advanced form features")
    team_df = compute_advanced_form(team_df, team_col, date_col)
    
    # 7. Home/Away splits
    print("  - Home/Away split features")
    team_df = compute_home_away_splits(team_df, team_col, date_col, window)
    
    print("V3 Features Applied Successfully!")
    
    return team_df


# =============================================================================
# API DATA INTEGRATION
# =============================================================================

class APIFootballIntegration:
    """
    Clase para integrar datos de API-Football (lesiones, lineups, etc.)
    Requiere API key de RapidAPI.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        } if api_key else None
        
    def get_injuries(self, fixture_id):
        """
        Obtiene lesiones para un partido específico.
        """
        if not self.api_key:
            return None
            
        import requests
        url = f"{self.base_url}/injuries"
        params = {"fixture": fixture_id}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            return data.get('response', [])
        except Exception as e:
            print(f"Error fetching injuries: {e}")
            return None
    
    def get_fixture_statistics(self, fixture_id):
        """
        Obtiene estadísticas detalladas por mitad (corners, shots, posesión).
        """
        if not self.api_key:
            return None
            
        import requests
        url = f"{self.base_url}/fixtures/statistics"
        params = {"fixture": fixture_id}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            return data.get('response', [])
        except Exception as e:
            print(f"Error fetching statistics: {e}")
            return None
    
    def get_h2h(self, team1_id, team2_id, last=5):
        """
        Obtiene historial head-to-head entre dos equipos.
        """
        if not self.api_key:
            return None
            
        import requests
        url = f"{self.base_url}/fixtures/headtohead"
        params = {"h2h": f"{team1_id}-{team2_id}", "last": last}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            return data.get('response', [])
        except Exception as e:
            print(f"Error fetching H2H: {e}")
            return None


def enrich_with_api_data(df, api_integration, fixture_column='fixture_id'):
    """
    Enriquece el DataFrame con datos de API-Football.
    Solo funciona si hay API key configurada.
    """
    if api_integration.api_key is None:
        print("No API key configured. Using only base features.")
        return df
    
    # Implementación placeholder - se expande según necesidad
    print("Enriching with API-Football data...")
    
    # Add injury count (placeholder logic)
    df['home_injuries_count'] = 0
    df['away_injuries_count'] = 0
    
    # Add key player missing flag (placeholder)
    df['home_key_player_missing'] = 0
    df['away_key_player_missing'] = 0
    
    return df


# =============================================================================
# FEATURE LISTS FOR MODELS
# =============================================================================

def get_corners_features():
    """Features óptimas para predicción de corners."""
    return [
        'rolling_corners_for', 'rolling_corners_against', 'corner_dominance',
        'rolling_total_corners', 'corners_variance',
        'estimated_corners_1h_for', 'estimated_corners_1h_against',
        'home_corners_avg', 'away_corners_avg'
    ]


def get_shots_features():
    """Features óptimas para predicción de remates."""
    return [
        'rolling_shots_for', 'rolling_shots_against', 'rolling_shots_ot_for',
        'rolling_shots_ot_against', 'shot_accuracy', 'rolling_total_shots',
        'shot_conversion'
    ]


def get_goals_features():
    """Features óptimas para predicción de goles (O/U)."""
    return [
        'rolling_goals_for', 'rolling_goals_against', 'rolling_xg_for',
        'rolling_xg_against', 'xg_per_shot', 'xg_overperformance',
        'xg_differential', 'shot_conversion', 'over_25_rate', 'over_15_rate',
        'scoring_streak', 'failed_to_score_rate', 'goals_trend'
    ]


def get_btts_features():
    """Features óptimas para BTTS."""
    return [
        'rolling_goals_for', 'rolling_goals_against', 'btts_rate',
        'scoring_streak', 'failed_to_score_rate', 'clean_sheet_rate',
        'conceding_streak', 'goals_conceded_per_shot'
    ]


def get_defensive_features():
    """Features de vulnerabilidad defensiva."""
    return [
        'rolling_goals_against', 'goals_conceded_per_shot', 'clean_sheet_rate',
        'conceding_streak', 'rolling_shots_against', 'rolling_xg_against'
    ]


def get_form_features():
    """Features de forma y momentum."""
    return [
        'points_last_5', 'points_last_10', 'goals_trend', 'form_momentum',
        'home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg'
    ]
