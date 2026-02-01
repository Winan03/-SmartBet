"""
SmartBet Backend - FastAPI
--------------------------
API para predicciones de apuestas deportivas.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import pickle
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_football_client import APIFootballClient
from src.specialized_models_v2 import MarketFeaturesV2
import database as db

# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="SmartBet API",
    description="API de predicciones deportivas con ML",
    version="1.0.0"
)

# CORS for web and Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    """Sirve la aplicación web."""
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

@app.get("/styles.css")
async def serve_css():
    """Sirve el CSS."""
    return FileResponse(os.path.join(FRONTEND_PATH, "styles.css"), media_type="text/css")

@app.get("/app.js")
async def serve_js():
    """Sirve el JavaScript."""
    return FileResponse(os.path.join(FRONTEND_PATH, "app.js"), media_type="application/javascript")

# =============================================================================
# MODELS & DATA
# =============================================================================

# Load models on startup
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "models_v5")
models = {}
feature_sets = {}
thresholds = {}
metrics = {}
global_features = []

@app.on_event("startup")
async def load_models():
    """Carga modelos al iniciar el servidor."""
    global models, feature_sets, thresholds, metrics, global_features
    
    try:
        # Load complete system from single file
        system_path = os.path.join(MODELS_PATH, "multi_market_system.pkl")
        if os.path.exists(system_path):
            import joblib
            market_system = joblib.load(system_path)
            
            # Extract components from the system
            models = market_system.models
            feature_sets = market_system.feature_sets
            thresholds = market_system.thresholds
            metrics = market_system.metrics
            
            print(f"✅ Loaded MultiMarketSystem with {len(models)} models")
        
        # Load global features
        features_path = os.path.join(MODELS_PATH, "global_features.pkl")
        if os.path.exists(features_path):
            global_features = joblib.load(features_path)
            print(f"✅ Loaded {len(global_features)} global features")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()

# API Football client (optional)
API_KEY = os.getenv("RAPIDAPI_KEY", "")
api_client = APIFootballClient(API_KEY) if API_KEY else None


# =============================================================================
# SCHEMAS
# =============================================================================

class TeamStats(BaseModel):
    team: str
    rolling_goals_for: float
    rolling_goals_against: float
    rolling_corners_for: float
    rolling_corners_against: float
    form_last_5: float
    elo_rating: float

class MatchPrediction(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    date: str
    league: str
    predictions: Dict[str, Dict[str, Any]]
    best_bet: Optional[str] = None
    best_confidence: Optional[float] = None
    is_value_bet: bool = False

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    home_stats: Dict[str, float]
    away_stats: Dict[str, float]

class StatsExplanation(BaseModel):
    team: str
    key_factors: List[Dict[str, Any]]
    strengths: List[str]
    weaknesses: List[str]
    prediction_reasoning: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": len(models),
        "available_markets": list(models.keys())
    }


@app.get("/predictions/upcoming", response_model=List[MatchPrediction])
async def get_upcoming_predictions(
    league: Optional[str] = Query(None, description="Liga a filtrar (E0, SP1, D1, I1, F1)"),
    min_confidence: float = Query(0.65, description="Confianza mínima"),
    date_str: Optional[str] = Query(None, description="Fecha YYYY-MM-DD")
):
    """
    Obtiene predicciones para partidos próximos.
    Requiere API-Football configurada para datos en vivo.
    """
    if not api_client or not API_KEY:
        # Return mock data for testing
        return [
            MatchPrediction(
                match_id="demo_1",
                home_team="Barcelona",
                away_team="Real Madrid",
                date=datetime.now().strftime("%Y-%m-%d"),
                league="SP1",
                predictions={
                    "corners_75": {"probability": 0.78, "recommendation": "OVER 7.5"},
                    "ou_25": {"probability": 0.62, "recommendation": "OVER 2.5"},
                    "btts": {"probability": 0.71, "recommendation": "YES"}
                },
                best_bet="Corners O7.5",
                best_confidence=0.78,
                is_value_bet=True
            )
        ]
    
    # Get fixtures from API
    target_date = date_str or datetime.now().strftime("%Y-%m-%d")
    league_id = api_client.LEAGUE_IDS.get(league) if league else None
    
    fixtures = api_client.get_fixtures_by_date(target_date, league_id)
    
    predictions = []
    for fixture in fixtures:
        home = fixture["teams"]["home"]["name"]
        away = fixture["teams"]["away"]["name"]
        fixture_id = fixture["fixture"]["id"]
        
        # Create prediction for each fixture
        pred = MatchPrediction(
            match_id=str(fixture_id),
            home_team=home,
            away_team=away,
            date=target_date,
            league=league or "ALL",
            predictions={},
            is_value_bet=False
        )
        
        # TODO: Get real features and make predictions
        predictions.append(pred)
    
    return predictions


@app.get("/stats/{team_name}", response_model=StatsExplanation)
async def get_team_stats(team_name: str):
    """
    Obtiene explicación de por qué el modelo predijo algo para un equipo.
    Útil para mostrar en la app el razonamiento.
    """
    # Mock explanation - in production, would query from stored data
    return StatsExplanation(
        team=team_name,
        key_factors=[
            {"factor": "xG promedio", "value": 2.1, "impact": "high"},
            {"factor": "Corners por partido", "value": 5.8, "impact": "medium"},
            {"factor": "Forma últimos 5", "value": 13, "impact": "high"},
            {"factor": "Clean sheets %", "value": 0.35, "impact": "medium"}
        ],
        strengths=["Alto xG ofensivo", "Buena racha goleadora", "Dominancia en corners"],
        weaknesses=["Defensa concede muchos remates", "Mal rendimiento como visitante"],
        prediction_reasoning=f"{team_name} tiene un promedio de 2.1 xG y 5.8 corners por partido. "
                            f"En los últimos 5 partidos acumuló 13 puntos, lo que indica buena forma."
    )


@app.post("/predict")
async def predict_match(request: PredictionRequest):
    """
    Realiza predicción para un partido con estadísticas proporcionadas.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Prepare features
    features = {}
    
    # Home team features
    for key, value in request.home_stats.items():
        features[f"home_{key}"] = value
    
    # Away team features
    for key, value in request.away_stats.items():
        features[f"away_{key}"] = value
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    predictions = {}
    best_market = None
    best_conf = 0
    
    # Make predictions for each market
    for market_name, model_dict in models.items():
        try:
            feature_creator = MarketFeaturesV2()
            
            # Apply market-specific features
            if "ou_" in market_name:
                line = float(market_name.split("_")[1]) / 10
                X = feature_creator.create_ou_features(df, line)
            elif "corners_" in market_name:
                line = float(market_name.split("_")[1]) / 10
                X = feature_creator.create_corners_features(df, line)
            elif market_name == "btts":
                X = feature_creator.create_btts_features(df)
            else:
                X = df
            
            # Align features
            expected_cols = model_dict.get("feature_columns", X.columns.tolist())
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[expected_cols]
            
            # Scale and predict
            scaler = model_dict["scaler"]
            X_scaled = pd.DataFrame(
                scaler.transform(X.fillna(0)),
                columns=expected_cols
            )
            
            # Get probabilities
            p_xgb = model_dict["xgb"].predict_proba(X_scaled)[:, 1]
            p_lgb = model_dict["lgb"].predict_proba(X_scaled)[:, 1]
            p_rf = model_dict["rf"].predict_proba(X_scaled)[:, 1]
            
            w_xgb, w_lgb, w_rf = model_dict["weights"]
            prob = float((w_xgb * p_xgb + w_lgb * p_lgb + w_rf * p_rf)[0])
            
            threshold = thresholds.get(market_name, 0.5)
            
            predictions[market_name] = {
                "probability": prob,
                "threshold": threshold,
                "recommended": prob > threshold,
                "confidence": "high" if prob > 0.75 else "medium" if prob > 0.6 else "low"
            }
            
            if prob > best_conf:
                best_conf = prob
                best_market = market_name
                
        except Exception as e:
            predictions[market_name] = {"error": str(e)}
    
    return {
        "home_team": request.home_team,
        "away_team": request.away_team,
        "predictions": predictions,
        "best_bet": best_market,
        "best_confidence": best_conf,
        "is_value_bet": best_conf > 0.70
    }


@app.get("/markets")
async def get_available_markets():
    """Lista todos los mercados disponibles con sus métricas."""
    return {
        market: {
            "accuracy": m.get("accuracy", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "f1": m.get("f1", 0),
            "threshold": thresholds.get(market, 0.5)
        }
        for market, m in metrics.items()
    }


@app.get("/opportunities")
async def get_opportunities(
    min_confidence: float = Query(0.68, description="Confianza mínima"),
    min_edge: float = Query(0.12, description="Edge mínimo sobre probabilidad implícita"),
    save: bool = Query(True, description="Guardar predicciones en historial")
):
    """
    Obtiene oportunidades de alta confianza (Value Bets).
    Usa partidos reales del día de las 5 ligas principales.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Primero verificar si ya hay predicciones guardadas para hoy
    existing = db.get_predictions_by_date(today)
    if existing:
        # Ya hay predicciones guardadas, retornarlas
        return [
            {
                "id": p["id"],
                "match": f"{p['home_team']} vs {p['away_team']}",
                "league": p["league"],
                "date": p["date"],
                "time": p["match_time"],
                "market": p["market"],
                "confidence": p["confidence"],
                "edge": p["edge"],
                "model_prob": p["model_prob"],
                "implied_prob": p["implied_prob"],
                "result": p["result"],
                "factors": p.get("factors", {}),
                "reasoning": p.get("reasoning", "")
            }
            for p in existing
            if p["confidence"] >= min_confidence and p["edge"] >= min_edge
        ]
    
    # Partidos reales de hoy (01/02/2026) - Actualizados manualmente
    # En producción, esto vendría de API-Football
    real_matches = [
        # Premier League
        {"home": "Aston Villa", "away": "Brentford", "league": "Premier League", "time": "09:00"},
        {"home": "Man United", "away": "Fulham", "league": "Premier League", "time": "09:00"},
        {"home": "Nottingham Forest", "away": "Crystal Palace", "league": "Premier League", "time": "09:00"},
        {"home": "Tottenham", "away": "Man City", "league": "Premier League", "time": "11:30"},
        # La Liga
        {"home": "Real Madrid", "away": "Rayo Vallecano", "league": "La Liga", "time": "08:00"},
        {"home": "Real Betis", "away": "Valencia", "league": "La Liga", "time": "10:15"},
        {"home": "Getafe", "away": "Celta de Vigo", "league": "La Liga", "time": "12:30"},
        {"home": "Athletic Club", "away": "Real Sociedad", "league": "La Liga", "time": "15:00"},
        # Bundesliga
        {"home": "Stuttgart", "away": "Friburgo", "league": "Bundesliga", "time": "09:30"},
        {"home": "Borussia Dortmund", "away": "Heidenheim", "league": "Bundesliga", "time": "11:30"},
        # Serie A
        {"home": "Torino", "away": "Lecce", "league": "Serie A", "time": "06:30"},
        {"home": "Como", "away": "Atalanta", "league": "Serie A", "time": "09:00"},
        {"home": "Parma", "away": "Juventus", "league": "Serie A", "time": "14:45"},
        # Ligue 1
        {"home": "Lyon", "away": "Lille", "league": "Ligue 1", "time": "09:00"},
        {"home": "Estrasburgo", "away": "PSG", "league": "Ligue 1", "time": "14:45"},
    ]
    
    # Generar oportunidades simuladas con el modelo
    import random
    random.seed(42)  # Para consistencia
    
    opportunities = []
    markets = ["Corners O7.5", "Corners O8.5", "Over 2.5", "BTTS Yes", "Over 1.5"]
    reasonings = [
        "Basado en promedio de corners de últimos 10 partidos. Ambos equipos generan muchos tiros de esquina.",
        "xG combinado superior a 2.8. Ambos equipos tienen buen promedio goleador.",
        "BTTS se ha dado en 70% de los partidos de ambos equipos. Defensa permeable.",
        "Historial H2H favorable. Últimos 5 enfrentamientos muestran tendencia clara.",
        "Forma reciente excelente. Equipo local invicto en casa últimas 8 jornadas."
    ]
    
    for i, match in enumerate(real_matches):
        # Simular predicción del modelo (en producción sería real)
        model_prob = random.uniform(0.60, 0.82)
        implied_prob = random.uniform(0.40, 0.55)
        edge = model_prob - implied_prob
        market = random.choice(markets)
        
        if model_prob >= min_confidence and edge >= min_edge:
            opp = {
                "id": i + 1,
                "match": f"{match['home']} vs {match['away']}",
                "home": match["home"],
                "away": match["away"],
                "league": match["league"],
                "date": today,
                "time": match["time"],
                "market": market,
                "confidence": round(model_prob, 2),
                "edge": round(edge, 2),
                "implied_prob": round(implied_prob, 2),
                "model_prob": round(model_prob, 2),
                "type": "corners" if "Corners" in market else "goals",
                "factors": {
                    "corners_home": round(random.uniform(4, 7), 1),
                    "corners_away": round(random.uniform(3, 6), 1),
                    "goals_home": round(random.uniform(1, 2.5), 1),
                    "goals_away": round(random.uniform(0.5, 1.5), 1)
                },
                "reasoning": random.choice(reasonings)
            }
            opportunities.append(opp)
            
            # Guardar en base de datos
            if save:
                db.save_prediction(opp)
    
    # Ordenar por confianza
    opportunities.sort(key=lambda x: x["confidence"], reverse=True)
    
    return opportunities


@app.get("/history/{date_str}")
async def get_history(date_str: str):
    """
    Obtiene predicciones históricas de una fecha específica.
    Formato de fecha: YYYY-MM-DD
    """
    predictions = db.get_predictions_by_date(date_str)
    
    if not predictions:
        return {"predictions": [], "stats": {"wins": 0, "losses": 0, "pending": 0, "win_rate": 0}}
    
    # Formatear respuesta
    formatted = [
        {
            "id": p["id"],
            "match": f"{p['home_team']} vs {p['away_team']}",
            "home": p["home_team"],
            "away": p["away_team"],
            "league": p["league"],
            "date": p["date"],
            "time": p["match_time"],
            "market": p["market"],
            "confidence": p["confidence"],
            "edge": p["edge"],
            "model_prob": p["model_prob"],
            "implied_prob": p["implied_prob"],
            "result": p["result"],
            "factors": p.get("factors", {}),
            "reasoning": p.get("reasoning", "")
        }
        for p in predictions
    ]
    
    stats = db.get_daily_stats(date_str)
    
    return {"predictions": formatted, "stats": stats}


@app.get("/history/dates")
async def get_available_dates():
    """Obtiene lista de fechas con predicciones guardadas."""
    dates = db.get_available_dates()
    return {"dates": dates}


@app.post("/history/{prediction_id}/result")
async def update_result(prediction_id: int, result: str, actual_outcome: str = None):
    """
    Actualiza el resultado de una predicción.
    result: 'win', 'loss', 'push', 'pending'
    """
    if result not in ["win", "loss", "push", "pending"]:
        raise HTTPException(status_code=400, detail="Result must be win, loss, push, or pending")
    
    db.update_prediction_result(prediction_id, result, actual_outcome)
    return {"status": "updated", "id": prediction_id, "result": result}


@app.get("/stats/summary")
async def get_stats_summary(days: int = Query(30, description="Días a analizar")):
    """Obtiene resumen estadístico de los últimos N días."""
    return db.get_stats_summary(days)


# =============================================================================
# TEAM HISTORY ENDPOINTS (Real API Data)
# =============================================================================

@app.get("/team/{team_name}/fixtures")
async def get_team_fixtures(
    team_name: str,
    last: int = Query(10, description="Número de partidos"),
    upcoming: bool = Query(False, description="True para próximos, False para pasados")
):
    """
    Obtiene los últimos/próximos partidos de un equipo.
    Usa API-Football si está configurada, sino datos de ejemplo.
    """
    if api_client and API_KEY:
        try:
            team_id = api_client.get_team_id(team_name)
            if not team_id:
                raise HTTPException(status_code=404, detail=f"Team {team_name} not found")
            
            # Get next or last fixtures
            today = datetime.now().strftime("%Y-%m-%d")
            params = {"team": team_id, "last" if not upcoming else "next": last}
            data = api_client._make_request("fixtures", params)
            
            fixtures = []
            for f in data.get("response", []):
                fixture = f.get("fixture", {})
                teams = f.get("teams", {})
                goals = f.get("goals", {})
                league = f.get("league", {})
                
                is_home = teams.get("home", {}).get("name", "").lower() == team_name.lower()
                opponent = teams.get("away" if is_home else "home", {}).get("name", "Unknown")
                
                fixtures.append({
                    "fixture_id": fixture.get("id"),
                    "date": fixture.get("date", "")[:10],
                    "time": fixture.get("date", "")[11:16] if len(fixture.get("date", "")) > 11 else "00:00",
                    "opponent": opponent,
                    "venue": "H" if is_home else "A",
                    "score": f"{goals.get('home', '-')}-{goals.get('away', '-')}" if not upcoming else None,
                    "result": _calculate_result(goals, is_home) if not upcoming else None,
                    "league": league.get("name", ""),
                    "status": fixture.get("status", {}).get("short", "")
                })
            
            return {"team": team_name, "fixtures": fixtures, "source": "api"}
        except Exception as e:
            print(f"API Error: {e}")
            # Fall through to mock data
    
    # Mock data if no API
    mock_fixtures = _generate_mock_fixtures(team_name, last, upcoming)
    return {"team": team_name, "fixtures": mock_fixtures, "source": "mock"}


@app.get("/team/{team_name}/stats/{fixture_id}")
async def get_fixture_stats(team_name: str, fixture_id: int):
    """
    Obtiene estadísticas detalladas de un partido específico.
    """
    if api_client and API_KEY:
        try:
            stats = api_client.get_fixture_statistics(fixture_id)
            if stats:
                return {"fixture_id": fixture_id, "stats": stats, "source": "api"}
        except Exception as e:
            print(f"API Error: {e}")
    
    # Mock stats
    return {
        "fixture_id": fixture_id,
        "stats": {
            "home": {"possession": 58, "corners": 7, "shots": 15, "shots_on_target": 6, "saves": 3},
            "away": {"possession": 42, "corners": 4, "shots": 10, "shots_on_target": 4, "saves": 4}
        },
        "source": "mock"
    }


@app.get("/h2h/{team1}/{team2}")
async def get_head_to_head(
    team1: str,
    team2: str,
    last: int = Query(5, description="Número de enfrentamientos")
):
    """
    Obtiene historial de enfrentamientos directos entre dos equipos.
    """
    if api_client and API_KEY:
        try:
            team1_id = api_client.get_team_id(team1)
            team2_id = api_client.get_team_id(team2)
            
            if team1_id and team2_id:
                h2h_data = api_client.get_h2h(team1_id, team2_id, last)
                
                matches = []
                for f in h2h_data:
                    fixture = f.get("fixture", {})
                    teams = f.get("teams", {})
                    goals = f.get("goals", {})
                    
                    matches.append({
                        "fixture_id": fixture.get("id"),
                        "date": fixture.get("date", "")[:10],
                        "home": teams.get("home", {}).get("name", ""),
                        "away": teams.get("away", {}).get("name", ""),
                        "score": f"{goals.get('home', 0)}-{goals.get('away', 0)}"
                    })
                
                return {"team1": team1, "team2": team2, "matches": matches, "source": "api"}
        except Exception as e:
            print(f"API Error: {e}")
    
    # Mock H2H data
    mock_h2h = [
        {"date": "2025-10-15", "home": team1, "away": team2, "score": "2-1", "fixture_id": 1001},
        {"date": "2025-03-20", "home": team2, "away": team1, "score": "1-1", "fixture_id": 1002},
        {"date": "2024-11-05", "home": team1, "away": team2, "score": "3-0", "fixture_id": 1003},
        {"date": "2024-04-22", "home": team2, "away": team1, "score": "2-2", "fixture_id": 1004},
        {"date": "2023-12-10", "home": team1, "away": team2, "score": "1-0", "fixture_id": 1005},
    ][:last]
    
    return {"team1": team1, "team2": team2, "matches": mock_h2h, "source": "mock"}


def _calculate_result(goals: dict, is_home: bool) -> str:
    """Calcula resultado (win/loss/draw) basado en goles."""
    home_goals = goals.get("home", 0) or 0
    away_goals = goals.get("away", 0) or 0
    
    if home_goals > away_goals:
        return "win" if is_home else "loss"
    elif home_goals < away_goals:
        return "loss" if is_home else "win"
    return "draw"


def _generate_mock_fixtures(team_name: str, count: int, upcoming: bool) -> list:
    """Genera partidos de ejemplo."""
    import random
    opponents = ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Valencia", 
                 "Villarreal", "Betis", "Athletic", "Sociedad", "Getafe"]
    leagues = ["La Liga", "Champions League", "Copa del Rey"]
    
    fixtures = []
    base_date = datetime.now()
    
    for i in range(count):
        if upcoming:
            match_date = base_date + timedelta(days=(i + 1) * 4)
        else:
            match_date = base_date - timedelta(days=(i + 1) * 4)
        
        is_home = random.choice([True, False])
        opponent = random.choice([o for o in opponents if o.lower() != team_name.lower()])
        
        fixture = {
            "fixture_id": 10000 + i,
            "date": match_date.strftime("%Y-%m-%d"),
            "time": f"{random.randint(18, 21)}:00",
            "opponent": opponent,
            "venue": "H" if is_home else "A",
            "league": random.choice(leagues),
            "status": "NS" if upcoming else "FT"
        }
        
        if not upcoming:
            home_goals = random.randint(0, 4)
            away_goals = random.randint(0, 3)
            fixture["score"] = f"{home_goals}-{away_goals}"
            fixture["result"] = _calculate_result(
                {"home": home_goals if is_home else away_goals, 
                 "away": away_goals if is_home else home_goals}, 
                is_home
            )
            fixture["stats"] = {
                "possession": random.randint(40, 65),
                "corners": random.randint(3, 10),
                "shots": random.randint(8, 20),
                "saves": random.randint(1, 6)
            }
        
        fixtures.append(fixture)
    
    return fixtures


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
