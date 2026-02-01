import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from scipy.stats import poisson

class MarketSpecificFeatures:
    """
    Generates optimal features for each specific betting market.
    """
    
    @staticmethod
    def create_1x2_features(df):
        features = df.copy()
        
        # 1. Absolute Differences (Relative Strength)
        features['goal_diff'] = features['home_rolling_goals_for'] - features['away_rolling_goals_for']
        features['defense_diff'] = features['away_rolling_goals_against'] - features['home_rolling_goals_against']
        
        if 'home_rolling_form_5' in features.columns:
            features['form_diff'] = features['home_rolling_form_5'] - features['away_rolling_form_5']
            
        features['elo_diff'] = features['home_elo'] - features['away_elo']
        
        # 2. Ratios
        features['attack_ratio'] = features['home_rolling_goals_for'] / (features['away_rolling_goals_for'] + 0.1)
        features['defense_ratio'] = features['away_rolling_goals_against'] / (features['home_rolling_goals_against'] + 0.1)
        
        # 3. Context (H2H, Momentum)
        if 'home_h2h_wins' in features.columns:
            features['h2h_dominance'] = features['home_h2h_wins'] - features.get('away_h2h_wins', 0)
        
        if 'home_momentum' in features.columns:
            features['momentum_diff'] = features['home_momentum'] - features['away_momentum']
            
        return features
    
    @staticmethod
    def create_ou_features(df):
        features = df.copy()
        
        # 1. Combined expectancy
        features['total_attack_strength'] = features['home_rolling_goals_for'] + features['away_rolling_goals_for']
        features['total_defense_weakness'] = features['home_rolling_goals_against'] + features['away_rolling_goals_against']
        
        # 2. Shot Accuracy
        if 'home_rolling_shot_efficiency' in features.columns:
            features['combined_efficiency'] = (features['home_rolling_shot_efficiency'] + features['away_rolling_shot_efficiency']) / 2
        
        # 3. Poisson Prediction
        features['poisson_ou_prob'] = features.apply(
            lambda row: MarketSpecificFeatures._poisson_over_25(
                row.get('home_rolling_goals_for', 1.5),
                row.get('away_rolling_goals_for', 1.5)
            ), axis=1
        )
        
        return features
    
    @staticmethod
    def create_btts_features(df):
        features = df.copy()
        
        # 1. Synergy (Both teams need to contribute)
        features['attack_product'] = features['home_rolling_goals_for'] * features['away_rolling_goals_for']
        features['defense_product'] = features['home_rolling_goals_against'] * features['away_rolling_goals_against']
        
        # 2. Minimum criteria (weakest link)
        features['min_attack'] = features[['home_rolling_goals_for', 'away_rolling_goals_for']].min(axis=1)
        features['max_defense'] = features[['home_rolling_goals_against', 'away_rolling_goals_against']].max(axis=1)
        
        # 3. Clean sheet tendencies
        if 'home_scoring_drought_rate' in features.columns and 'away_scoring_drought_rate' in features.columns:
            features['max_drought'] = features[['home_scoring_drought_rate', 'away_scoring_drought_rate']].max(axis=1)
        
        # 4. Poisson
        features['poisson_btts_prob'] = features.apply(
            lambda row: MarketSpecificFeatures._poisson_btts(
                row.get('home_rolling_goals_for', 1.5),
                row.get('away_rolling_goals_for', 1.5)
            ), axis=1
        )
        
        return features
    
    @staticmethod
    def _poisson_over_25(home_att, away_att):
        lambda_h, lambda_a = home_att * 1.15, away_att * 0.95
        prob_under = 0
        for total in range(3):
            for h in range(total + 1):
                prob_under += (poisson.pmf(h, lambda_h) * poisson.pmf(total - h, lambda_a))
        return 1 - prob_under

    @staticmethod
    def _poisson_btts(home_att, away_att):
        lambda_h, lambda_a = home_att * 1.15, away_att * 0.95
        p_h0, p_a0 = poisson.pmf(0, lambda_h), poisson.pmf(0, lambda_a)
        return 1 - p_h0 - p_a0 + (p_h0 * p_a0)

class SpecializedMarketModels:
    def __init__(self):
        self.models = {}
        self.feature_sets = {}
        self.thresholds = {}

    def train_1x2_model(self, X_train, y_train, X_val, y_val):
        print("Training Calibrated 1X2 Ensemble...")
        f_creator = MarketSpecificFeatures()
        X_t, X_v = f_creator.create_1x2_features(X_train), f_creator.create_1x2_features(X_val)
        
        # XGB with Calibration
        base_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=6, random_state=42, eval_metric='mlogloss')
        calibrated_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=3)
        
        # LGB with Calibration
        base_lgb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1)
        calibrated_lgb = CalibratedClassifierCV(base_lgb, method='sigmoid', cv=3)
        
        calibrated_xgb.fit(X_t, y_train)
        calibrated_lgb.fit(X_t, y_train)
        
        acc_x = accuracy_score(y_val, calibrated_xgb.predict(X_v))
        acc_l = accuracy_score(y_val, calibrated_lgb.predict(X_v))
        
        w_x, w_l = acc_x / (acc_x + acc_l), acc_l / (acc_x + acc_l)
        self.models['1x2'] = {'xgb': calibrated_xgb, 'lgb': calibrated_lgb, 'weights': (w_x, w_l)}
        self.feature_sets['1x2'] = X_t.columns.tolist()
        return (acc_x * w_x + acc_l * w_l)

    def train_ou_model(self, X_train, y_train, X_val, y_val):
        print("Training Calibrated O/U 2.5 Model...")
        f_creator = MarketSpecificFeatures()
        X_t, X_v = f_creator.create_ou_features(X_train), f_creator.create_ou_features(X_val)
        
        base_model = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, random_state=42, eval_metric='logloss')
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        calibrated_model.fit(X_t, y_train)
        
        probs = calibrated_model.predict_proba(X_v)[:, 1]
        best_t, best_a = 0.5, accuracy_score(y_val, (probs > 0.5).astype(int))
        
        for t in np.linspace(0.4, 0.6, 21):
            acc = accuracy_score(y_val, (probs > t).astype(int))
            if acc > best_a: best_a, best_t = acc, t
            
        self.models['ou'] = calibrated_model
        self.feature_sets['ou'] = X_t.columns.tolist()
        self.thresholds['ou'] = best_t
        return best_a

    def train_btts_model(self, X_train, y_train, X_val, y_val):
        print("Training Calibrated BTTS Ensemble...")
        f_creator = MarketSpecificFeatures()
        X_t, X_v = f_creator.create_btts_features(X_train), f_creator.create_btts_features(X_val)
        
        # Pipeline for Scaling + LR
        m_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=2000, random_state=42))
        ])
        
        base_xgb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42, eval_metric='logloss')
        calibrated_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=3)
        
        m_lr.fit(X_t, y_train)
        calibrated_xgb.fit(X_t, y_train)
        
        acc_r = accuracy_score(y_val, m_lr.predict(X_v))
        acc_x = accuracy_score(y_val, calibrated_xgb.predict(X_v))
        
        w_r, w_x = acc_r / (acc_r + acc_x), acc_x / (acc_r + acc_x)
        self.models['btts'] = {'lr': m_lr, 'xgb': calibrated_xgb, 'weights': (w_r, w_x)}
        self.feature_sets['btts'] = X_t.columns.tolist()
        return (acc_r * w_r + acc_x * w_x)
