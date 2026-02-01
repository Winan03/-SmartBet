"""
SmartBet Specialized Models V2
-------------------------------
Sistema multi-mercado con modelos especializados para cada tipo de apuesta.
Objetivo: Precisión y Recall 70-90% con alta confianza.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MARKET-SPECIFIC FEATURE CREATORS
# =============================================================================

class MarketFeaturesV2:
    """
    Crea features específicas optimizadas para cada mercado de apuestas.
    """
    
    @staticmethod
    def create_ou_features(df, line=2.5):
        """Features para Over/Under en cualquier línea."""
        features = df.copy()
        
        # Combined attack strength
        features['total_attack'] = features['home_rolling_goals_for'] + features['away_rolling_goals_for']
        features['total_defense_weakness'] = features['home_rolling_goals_against'] + features['away_rolling_goals_against']
        
        # xG if available
        if 'home_rolling_xg_for' in features.columns:
            features['total_xg'] = features['home_rolling_xg_for'] + features['away_rolling_xg_for']
            features['xg_diff'] = features['total_xg'] - line
        
        # Historical over rates
        if 'home_over_25_rate' in features.columns:
            features['combined_over_rate'] = (features['home_over_25_rate'] + features['away_over_25_rate']) / 2
        
        # Scoring momentum
        if 'home_scoring_streak' in features.columns:
            features['combined_scoring_streak'] = features['home_scoring_streak'] + features['away_scoring_streak']
        
        # Failed to score (negative signal for over)
        if 'home_failed_to_score_rate' in features.columns:
            features['max_fail_rate'] = features[['home_failed_to_score_rate', 'away_failed_to_score_rate']].max(axis=1)
        
        # Shot conversion synergy
        if 'home_shot_conversion' in features.columns:
            features['combined_conversion'] = (features['home_shot_conversion'] + features['away_shot_conversion']) / 2
        
        # Poisson probability for over
        features['poisson_over_prob'] = features.apply(
            lambda row: MarketFeaturesV2._poisson_over(
                row.get('home_rolling_goals_for', 1.3),
                row.get('away_rolling_goals_for', 1.0),
                line
            ), axis=1
        )
        
        return features
    
    @staticmethod
    def create_btts_features(df):
        """Features para Both Teams To Score."""
        features = df.copy()
        
        # Attack product (both need to score)
        features['attack_product'] = features['home_rolling_goals_for'] * features['away_rolling_goals_for']
        
        # Minimum attack (weakest link)
        features['min_attack'] = features[['home_rolling_goals_for', 'away_rolling_goals_for']].min(axis=1)
        
        # Maximum failed to score rate
        if 'home_failed_to_score_rate' in features.columns:
            features['max_fail_rate'] = features[['home_failed_to_score_rate', 'away_failed_to_score_rate']].max(axis=1)
        
        # BTTS historical rate
        if 'home_btts_rate' in features.columns:
            features['combined_btts_rate'] = (features['home_btts_rate'] + features['away_btts_rate']) / 2
        
        # Clean sheet rates (negative signal)
        if 'home_clean_sheet_rate' in features.columns:
            features['max_clean_sheet'] = features[['home_clean_sheet_rate', 'away_clean_sheet_rate']].max(axis=1)
        
        # Scoring streaks
        if 'home_scoring_streak' in features.columns:
            features['min_scoring_streak'] = features[['home_scoring_streak', 'away_scoring_streak']].min(axis=1)
        
        # Poisson BTTS probability
        features['poisson_btts_prob'] = features.apply(
            lambda row: MarketFeaturesV2._poisson_btts(
                row.get('home_rolling_goals_for', 1.3),
                row.get('away_rolling_goals_for', 1.0)
            ), axis=1
        )
        
        return features
    
    @staticmethod
    def create_corners_features(df, line=9.5):
        """Features para mercado de corners totales."""
        features = df.copy()
        
        # Total corners tendency
        features['total_corners_tendency'] = features['home_rolling_corners_for'] + features['away_rolling_corners_for']
        features['total_corners_against'] = features['home_rolling_corners_against'] + features['away_rolling_corners_against']
        
        # Estimated match total
        features['estimated_total_corners'] = (
            features['home_rolling_corners_for'] + features['home_rolling_corners_against'] +
            features['away_rolling_corners_for'] + features['away_rolling_corners_against']
        ) / 2
        
        # Corner dominance
        if 'home_corner_dominance' in features.columns:
            features['combined_corner_dominance'] = features['home_corner_dominance'] + features['away_corner_dominance']
        
        # Home/Away splits
        if 'home_home_corners_avg' in features.columns:
            features['home_specific_corners'] = features['home_home_corners_avg']
            features['away_specific_corners'] = features['away_away_corners_avg']
        
        # Variance (consistency)
        if 'home_corners_variance' in features.columns:
            features['combined_corners_variance'] = features['home_corners_variance'] + features['away_corners_variance']
        
        # Difference from line
        features['corners_diff_from_line'] = features['estimated_total_corners'] - line
        
        return features
    
    @staticmethod
    def create_corners_1h_features(df, line=4.5):
        """Features para corners primer tiempo."""
        features = df.copy()
        
        # Estimated 1H corners
        if 'home_estimated_corners_1h_for' in features.columns:
            features['total_estimated_1h'] = (
                features['home_estimated_corners_1h_for'] + features['away_estimated_corners_1h_for']
            )
            features['corners_1h_diff'] = features['total_estimated_1h'] - line
        
        return features
    
    @staticmethod
    def create_shots_features(df, line=9.5, on_target=True):
        """Features para mercado de remates."""
        features = df.copy()
        
        if on_target:
            # Shots on target
            if 'home_rolling_shots_ot_for' in features.columns:
                features['total_shots_ot'] = features['home_rolling_shots_ot_for'] + features['away_rolling_shots_ot_for']
                features['shots_ot_conceded'] = features['home_rolling_shots_ot_against'] + features['away_rolling_shots_ot_against']
                features['estimated_match_sot'] = (features['total_shots_ot'] + features['shots_ot_conceded']) / 2
        else:
            # Total shots
            features['total_shots'] = features['home_rolling_shots_for'] + features['away_rolling_shots_for']
            features['total_shots_conceded'] = features['home_rolling_shots_against'] + features['away_rolling_shots_against']
            features['estimated_match_shots'] = (features['total_shots'] + features['total_shots_conceded']) / 2
        
        # Shot accuracy
        if 'home_shot_accuracy' in features.columns:
            features['combined_shot_accuracy'] = (features['home_shot_accuracy'] + features['away_shot_accuracy']) / 2
        
        return features
    
    @staticmethod
    def create_1x2_features(df):
        """Features para resultado 1X2."""
        features = df.copy()
        
        # Strength differentials
        features['goal_diff'] = features['home_rolling_goals_for'] - features['away_rolling_goals_for']
        features['defense_diff'] = features['away_rolling_goals_against'] - features['home_rolling_goals_against']
        
        # Elo difference
        if 'home_elo' in features.columns:
            features['elo_diff'] = features['home_elo'] - features['away_elo']
        
        # Form difference
        if 'home_points_last_5' in features.columns:
            features['form_diff'] = features['home_points_last_5'] - features['away_points_last_5']
        
        # Home/Away specific
        if 'home_home_goals_avg' in features.columns:
            features['home_advantage'] = features['home_home_goals_avg'] - features['away_away_goals_avg']
        
        # xG differential
        if 'home_xg_differential' in features.columns:
            features['combined_xg_diff'] = features['home_xg_differential'] - features['away_xg_differential']
        
        # Momentum
        if 'home_form_momentum' in features.columns:
            features['momentum_diff'] = features['home_form_momentum'] - features['away_form_momentum']
        
        return features
    
    @staticmethod
    def _poisson_over(home_att, away_att, line):
        """Calcula probabilidad de over usando Poisson."""
        lambda_h = home_att * 1.1  # Home advantage
        lambda_a = away_att * 0.95
        
        prob_under = 0
        target = int(line)
        for total in range(target + 1):
            for h in range(total + 1):
                prob_under += poisson.pmf(h, lambda_h) * poisson.pmf(total - h, lambda_a)
        
        return 1 - prob_under
    
    @staticmethod
    def _poisson_btts(home_att, away_att):
        """Calcula probabilidad de BTTS usando Poisson."""
        lambda_h = home_att * 1.1
        lambda_a = away_att * 0.95
        
        p_h_0 = poisson.pmf(0, lambda_h)
        p_a_0 = poisson.pmf(0, lambda_a)
        
        return (1 - p_h_0) * (1 - p_a_0)


# =============================================================================
# MULTI-MARKET MODEL SYSTEM
# =============================================================================

class MultiMarketSystem:
    """
    Sistema unificado de modelos para múltiples mercados de apuestas.
    """
    
    def __init__(self):
        self.models = {}
        self.feature_sets = {}
        self.thresholds = {}
        self.metrics = {}
        self.scaler = StandardScaler()
    
    def _train_binary_ensemble(self, X_train, y_train, X_val, y_val, market_name):
        """
        Entrena un ensemble calibrado para mercados binarios.
        Optimizado para alta precisión y recall.
        """
        # Guardar columnas originales para alineación en predicción
        train_columns = X_train.columns.tolist()
        
        # Crear nuevo scaler para este mercado específico
        market_scaler = StandardScaler()
        
        # Preparar features
        X_train_scaled = pd.DataFrame(
            market_scaler.fit_transform(X_train.fillna(0)),
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            market_scaler.transform(X_val.fillna(0)),
            columns=X_val.columns
        )
        
        # Modelo 1: XGBoost calibrado
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
        
        # Modelo 2: LightGBM calibrado
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            verbose=-1
        )
        calibrated_lgb = CalibratedClassifierCV(lgb_model, method='isotonic', cv=5)
        
        # Modelo 3: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
        
        # Entrenar
        print(f"  Training {market_name} XGBoost...")
        calibrated_xgb.fit(X_train_scaled, y_train)
        print(f"  Training {market_name} LightGBM...")
        calibrated_lgb.fit(X_train_scaled, y_train)
        print(f"  Training {market_name} RandomForest...")
        calibrated_rf.fit(X_train_scaled, y_train)
        
        # Obtener probabilidades
        p_xgb = calibrated_xgb.predict_proba(X_val_scaled)[:, 1]
        p_lgb = calibrated_lgb.predict_proba(X_val_scaled)[:, 1]
        p_rf = calibrated_rf.predict_proba(X_val_scaled)[:, 1]
        
        # Pesos basados en performance
        acc_xgb = accuracy_score(y_val, (p_xgb > 0.5).astype(int))
        acc_lgb = accuracy_score(y_val, (p_lgb > 0.5).astype(int))
        acc_rf = accuracy_score(y_val, (p_rf > 0.5).astype(int))
        
        total_acc = acc_xgb + acc_lgb + acc_rf
        w_xgb = acc_xgb / total_acc
        w_lgb = acc_lgb / total_acc
        w_rf = acc_rf / total_acc
        
        # Ensemble probability
        ensemble_probs = (w_xgb * p_xgb) + (w_lgb * p_lgb) + (w_rf * p_rf)
        
        # Find optimal threshold for precision/recall balance
        best_threshold = 0.5
        best_f1 = 0
        for t in np.linspace(0.45, 0.75, 31):
            preds = (ensemble_probs > t).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        # Calculate metrics
        final_preds = (ensemble_probs > best_threshold).astype(int)
        _precision = precision_score(y_val, final_preds)
        _recall = recall_score(y_val, final_preds)
        _accuracy = accuracy_score(y_val, final_preds)
        
        print(f"  {market_name} Results: Acc={_accuracy:.2%}, Prec={_precision:.2%}, Recall={_recall:.2%}, F1={best_f1:.2%}")
        
        return {
            'xgb': calibrated_xgb,
            'lgb': calibrated_lgb,
            'rf': calibrated_rf,
            'weights': (w_xgb, w_lgb, w_rf),
            'scaler': market_scaler,
            'feature_columns': train_columns,  # Guardar columnas para alineación
            'threshold': best_threshold,
            'metrics': {
                'accuracy': _accuracy,
                'precision': _precision,
                'recall': _recall,
                'f1': best_f1
            }
        }
    
    def _predict_proba_ensemble(self, model_dict, X):
        """Obtiene probabilidades del ensemble con alineación de features."""
        scaler = model_dict['scaler']
        expected_columns = model_dict.get('feature_columns', X.columns.tolist())
        
        # Alinear features: usar solo las columnas que el modelo espera
        X_aligned = X.copy()
        
        # Añadir columnas faltantes con ceros
        for col in expected_columns:
            if col not in X_aligned.columns:
                X_aligned[col] = 0
        
        # Seleccionar solo las columnas esperadas en el orden correcto
        X_aligned = X_aligned[expected_columns]
        
        X_scaled = pd.DataFrame(
            scaler.transform(X_aligned.fillna(0)),
            columns=expected_columns
        )
        
        p_xgb = model_dict['xgb'].predict_proba(X_scaled)[:, 1]
        p_lgb = model_dict['lgb'].predict_proba(X_scaled)[:, 1]
        p_rf = model_dict['rf'].predict_proba(X_scaled)[:, 1]
        
        w_xgb, w_lgb, w_rf = model_dict['weights']
        
        return (w_xgb * p_xgb) + (w_lgb * p_lgb) + (w_rf * p_rf)
    
    # =========================================================================
    # GOALS MARKETS
    # =========================================================================
    
    def train_ou_model(self, X_train, y_train, X_val, y_val, line=2.5):
        """Entrena modelo Over/Under para cualquier línea."""
        market_name = f'ou_{str(line).replace(".", "")}'
        print(f"\n=== Training Over/Under {line} Model ===")
        
        f_creator = MarketFeaturesV2()
        X_t = f_creator.create_ou_features(X_train, line)
        X_v = f_creator.create_ou_features(X_val, line)
        
        model_dict = self._train_binary_ensemble(X_t, y_train, X_v, y_val, f'O/U {line}')
        
        self.models[market_name] = model_dict
        self.feature_sets[market_name] = X_t.columns.tolist()
        self.thresholds[market_name] = model_dict['threshold']
        self.metrics[market_name] = model_dict['metrics']
        
        return model_dict['metrics']
    
    def train_btts_model(self, X_train, y_train, X_val, y_val):
        """Entrena modelo BTTS mejorado."""
        print("\n=== Training BTTS Model ===")
        
        f_creator = MarketFeaturesV2()
        X_t = f_creator.create_btts_features(X_train)
        X_v = f_creator.create_btts_features(X_val)
        
        model_dict = self._train_binary_ensemble(X_t, y_train, X_v, y_val, 'BTTS')
        
        self.models['btts'] = model_dict
        self.feature_sets['btts'] = X_t.columns.tolist()
        self.thresholds['btts'] = model_dict['threshold']
        self.metrics['btts'] = model_dict['metrics']
        
        return model_dict['metrics']
    
    # =========================================================================
    # CORNERS MARKETS
    # =========================================================================
    
    def train_corners_model(self, X_train, y_train, X_val, y_val, line=9.5):
        """Entrena modelo de corners totales para cualquier línea."""
        market_name = f'corners_{str(line).replace(".", "")}'
        print(f"\n=== Training Corners O/U {line} Model ===")
        
        f_creator = MarketFeaturesV2()
        X_t = f_creator.create_corners_features(X_train, line)
        X_v = f_creator.create_corners_features(X_val, line)
        
        model_dict = self._train_binary_ensemble(X_t, y_train, X_v, y_val, f'Corners {line}')
        
        self.models[market_name] = model_dict
        self.feature_sets[market_name] = X_t.columns.tolist()
        self.thresholds[market_name] = model_dict['threshold']
        self.metrics[market_name] = model_dict['metrics']
        
        return model_dict['metrics']
    
    def train_corners_1h_model(self, X_train, y_train, X_val, y_val, line=4.5):
        """Entrena modelo de corners primer tiempo."""
        market_name = f'corners_1h_{str(line).replace(".", "")}'
        print(f"\n=== Training Corners 1H O/U {line} Model ===")
        
        f_creator = MarketFeaturesV2()
        X_t = f_creator.create_corners_1h_features(X_train, line)
        X_v = f_creator.create_corners_1h_features(X_val, line)
        
        model_dict = self._train_binary_ensemble(X_t, y_train, X_v, y_val, f'Corners 1H {line}')
        
        self.models[market_name] = model_dict
        self.feature_sets[market_name] = X_t.columns.tolist()
        self.thresholds[market_name] = model_dict['threshold']
        self.metrics[market_name] = model_dict['metrics']
        
        return model_dict['metrics']
    
    # =========================================================================
    # SHOTS MARKETS
    # =========================================================================
    
    def train_shots_on_target_model(self, X_train, y_train, X_val, y_val, line=8.5):
        """Entrena modelo de remates al arco."""
        market_name = f'sot_{str(line).replace(".", "")}'
        print(f"\n=== Training Shots on Target O/U {line} Model ===")
        
        f_creator = MarketFeaturesV2()
        X_t = f_creator.create_shots_features(X_train, line, on_target=True)
        X_v = f_creator.create_shots_features(X_val, line, on_target=True)
        
        model_dict = self._train_binary_ensemble(X_t, y_train, X_v, y_val, f'SoT {line}')
        
        self.models[market_name] = model_dict
        self.feature_sets[market_name] = X_t.columns.tolist()
        self.thresholds[market_name] = model_dict['threshold']
        self.metrics[market_name] = model_dict['metrics']
        
        return model_dict['metrics']
    
    # =========================================================================
    # 1X2 MARKET
    # =========================================================================
    
    def train_1x2_model(self, X_train, y_train, X_val, y_val):
        """Entrena modelo 1X2 (resultado del partido)."""
        print("\n=== Training 1X2 Model ===")
        
        f_creator = MarketFeaturesV2()
        X_t = f_creator.create_1x2_features(X_train)
        X_v = f_creator.create_1x2_features(X_val)
        
        # Para 1X2 usamos multiclase
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_t.fillna(0)),
            columns=X_t.columns
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_v.fillna(0)),
            columns=X_v.columns
        )
        
        # XGBoost multiclase
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
        
        # LightGBM multiclase
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        calibrated_lgb = CalibratedClassifierCV(lgb_model, method='isotonic', cv=5)
        
        print("  Training 1X2 XGBoost...")
        calibrated_xgb.fit(X_train_scaled, y_train)
        print("  Training 1X2 LightGBM...")
        calibrated_lgb.fit(X_train_scaled, y_train)
        
        # Evaluate
        acc_xgb = accuracy_score(y_val, calibrated_xgb.predict(X_val_scaled))
        acc_lgb = accuracy_score(y_val, calibrated_lgb.predict(X_val_scaled))
        
        total = acc_xgb + acc_lgb
        w_xgb, w_lgb = acc_xgb / total, acc_lgb / total
        
        # Ensemble predictions
        p_xgb = calibrated_xgb.predict_proba(X_val_scaled)
        p_lgb = calibrated_lgb.predict_proba(X_val_scaled)
        ensemble_probs = (w_xgb * p_xgb) + (w_lgb * p_lgb)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        _accuracy = accuracy_score(y_val, ensemble_preds)
        print(f"  1X2 Results: Accuracy={_accuracy:.2%}")
        
        self.models['1x2'] = {
            'xgb': calibrated_xgb,
            'lgb': calibrated_lgb,
            'weights': (w_xgb, w_lgb),
            'scaler': self.scaler
        }
        self.feature_sets['1x2'] = X_t.columns.tolist()
        self.metrics['1x2'] = {'accuracy': _accuracy}
        
        return {'accuracy': _accuracy}
    
    # =========================================================================
    # FRANCOTIRADOR MULTI-MARKET
    # =========================================================================
    
    def francotirador_multi_market(self, val_df, X_val, min_edge=0.10, min_conf=0.65):
        """
        Sistema Francotirador: selecciona las mejores apuestas de TODOS los mercados.
        Solo apuesta donde hay edge significativo y alta confianza.
        
        Returns: DataFrame con las apuestas seleccionadas ordenadas por edge.
        """
        print("\n" + "="*60)
        print("FRANCOTIRADOR MULTI-MARKET SELECTOR")
        print("="*60)
        print(f"Criterios: Edge mínimo={min_edge:.0%}, Confianza mínima={min_conf:.0%}")
        
        all_bets = []
        
        # ====== O/U MARKETS ======
        for line in [1.5, 2.5, 3.5]:
            market_name = f'ou_{str(line).replace(".", "")}'
            if market_name not in self.models:
                continue
            
            target_col = f'target_ou{str(line).replace(".", "")}'
            if target_col not in val_df.columns:
                continue
            
            f_creator = MarketFeaturesV2()
            X_features = f_creator.create_ou_features(X_val, line)
            probs = self._predict_proba_ensemble(self.models[market_name], X_features)
            
            odd_col = f'Avg>{line}' if f'Avg>{line}' in val_df.columns else None
            
            for i in range(len(probs)):
                conf = probs[i]
                if conf > min_conf:
                    if odd_col and odd_col in val_df.columns:
                        implied_prob = 1 / val_df.iloc[i][odd_col] if val_df.iloc[i][odd_col] > 0 else 0.5
                        edge = conf - implied_prob
                        if edge > min_edge:
                            all_bets.append({
                                'index': i,
                                'market': f'Over {line}',
                                'confidence': conf,
                                'edge': edge,
                                'odds': val_df.iloc[i][odd_col],
                                'target': val_df.iloc[i][target_col],
                                'home': val_df.iloc[i].get('home_team', 'N/A'),
                                'away': val_df.iloc[i].get('away_team', 'N/A')
                            })
        
        # ====== BTTS ======
        if 'btts' in self.models and 'target_btts' in val_df.columns:
            f_creator = MarketFeaturesV2()
            X_features = f_creator.create_btts_features(X_val)
            probs = self._predict_proba_ensemble(self.models['btts'], X_features)
            
            for i in range(len(probs)):
                conf = probs[i]
                if conf > min_conf:
                    all_bets.append({
                        'index': i,
                        'market': 'BTTS Yes',
                        'confidence': conf,
                        'edge': conf - 0.5,  # Sin cuotas específicas
                        'odds': None,
                        'target': val_df.iloc[i]['target_btts'],
                        'home': val_df.iloc[i].get('home_team', 'N/A'),
                        'away': val_df.iloc[i].get('away_team', 'N/A')
                    })
        
        # ====== CORNERS ======
        for line in [7.5, 8.5, 9.5, 10.5]:
            market_name = f'corners_{str(line).replace(".", "")}'
            if market_name not in self.models:
                continue
            
            target_col = f'target_corners_{str(line).replace(".", "")}'
            if target_col not in val_df.columns:
                continue
            
            f_creator = MarketFeaturesV2()
            X_features = f_creator.create_corners_features(X_val, line)
            probs = self._predict_proba_ensemble(self.models[market_name], X_features)
            
            for i in range(len(probs)):
                conf = probs[i]
                if conf > min_conf:
                    all_bets.append({
                        'index': i,
                        'market': f'Corners O{line}',
                        'confidence': conf,
                        'edge': conf - 0.5,
                        'odds': None,
                        'target': val_df.iloc[i][target_col],
                        'home': val_df.iloc[i].get('home_team', 'N/A'),
                        'away': val_df.iloc[i].get('away_team', 'N/A')
                    })
        
        # ====== RESULTS ======
        if not all_bets:
            print("\n❌ No high-value bets found with current criteria.")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(all_bets)
        results_df = results_df.sort_values('edge', ascending=False)
        
        # Calculate ROI
        results_df['win'] = results_df['target'] == 1
        
        print(f"\n📊 Selected Bets: {len(results_df)}")
        print(f"✅ Win Rate: {results_df['win'].mean():.2%}")
        
        # Group by market
        print("\n--- Performance by Market ---")
        for market in results_df['market'].unique():
            market_df = results_df[results_df['market'] == market]
            wr = market_df['win'].mean()
            print(f"  {market}: {len(market_df)} bets, WR={wr:.2%}")
        
        return results_df
    
    def summary(self):
        """Muestra resumen de todos los modelos entrenados."""
        print("\n" + "="*60)
        print("MULTI-MARKET SYSTEM SUMMARY")
        print("="*60)
        
        for market, metrics in self.metrics.items():
            print(f"\n{market.upper()}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.2%}")
