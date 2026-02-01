"""
SmartBet Database - Gestión de predicciones e historial
"""

import sqlite3
import json
from datetime import datetime, date
from typing import List, Dict, Optional, Any
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def get_connection():
    """Obtiene conexión a la base de datos."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Inicializa la base de datos con las tablas necesarias."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Tabla de predicciones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            league TEXT NOT NULL,
            match_time TEXT,
            market TEXT NOT NULL,
            confidence REAL NOT NULL,
            edge REAL NOT NULL,
            model_prob REAL NOT NULL,
            implied_prob REAL NOT NULL,
            factors TEXT,
            reasoning TEXT,
            result TEXT DEFAULT 'pending',
            actual_outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Índices para búsquedas rápidas
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_result ON predictions(result)")
    
    conn.commit()
    conn.close()
    print("✅ Base de datos inicializada")


def save_prediction(prediction: Dict[str, Any]) -> int:
    """Guarda una predicción en la base de datos."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Extraer home y away del match o usar campos separados
    match = prediction.get("match", "")
    if " vs " in match:
        home, away = match.split(" vs ", 1)
    else:
        home = prediction.get("home", prediction.get("home_team", ""))
        away = prediction.get("away", prediction.get("away_team", ""))
    
    cursor.execute("""
        INSERT INTO predictions (
            date, home_team, away_team, league, match_time,
            market, confidence, edge, model_prob, implied_prob,
            factors, reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction.get("date", datetime.now().strftime("%Y-%m-%d")),
        home,
        away,
        prediction.get("league", ""),
        prediction.get("time", ""),
        prediction.get("market", ""),
        prediction.get("confidence", 0.0),
        prediction.get("edge", 0.0),
        prediction.get("model_prob", 0.0),
        prediction.get("implied_prob", 0.0),
        json.dumps(prediction.get("factors", {})),
        prediction.get("reasoning", "")
    ))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id


def save_predictions_batch(predictions: List[Dict[str, Any]]) -> List[int]:
    """Guarda múltiples predicciones a la vez."""
    ids = []
    for pred in predictions:
        pred_id = save_prediction(pred)
        ids.append(pred_id)
    return ids


def get_predictions_by_date(target_date: str) -> List[Dict[str, Any]]:
    """Obtiene todas las predicciones de una fecha específica."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM predictions 
        WHERE date = ? 
        ORDER BY created_at DESC
    """, (target_date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    predictions = []
    for row in rows:
        pred = dict(row)
        # Parse factors from JSON
        if pred.get("factors"):
            try:
                pred["factors"] = json.loads(pred["factors"])
            except:
                pred["factors"] = {}
        predictions.append(pred)
    
    return predictions


def update_prediction_result(prediction_id: int, result: str, actual_outcome: str = None):
    """Actualiza el resultado de una predicción (win/loss/push)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE predictions 
        SET result = ?, actual_outcome = ?
        WHERE id = ?
    """, (result, actual_outcome, prediction_id))
    
    conn.commit()
    conn.close()


def get_stats_summary(days: int = 30) -> Dict[str, Any]:
    """Obtiene estadísticas resumidas de los últimos N días."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            result,
            COUNT(*) as count
        FROM predictions
        WHERE date >= date('now', ?)
        GROUP BY result
    """, (f'-{days} days',))
    
    rows = cursor.fetchall()
    conn.close()
    
    stats = {"win": 0, "loss": 0, "pending": 0, "push": 0}
    for row in rows:
        stats[row["result"]] = row["count"]
    
    total_resolved = stats["win"] + stats["loss"]
    win_rate = (stats["win"] / total_resolved * 100) if total_resolved > 0 else 0
    
    return {
        "wins": stats["win"],
        "losses": stats["loss"],
        "pending": stats["pending"],
        "push": stats["push"],
        "total": sum(stats.values()),
        "win_rate": round(win_rate, 1)
    }


def get_available_dates() -> List[str]:
    """Obtiene lista de fechas con predicciones guardadas."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT date 
        FROM predictions 
        ORDER BY date DESC 
        LIMIT 30
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [row["date"] for row in rows]


def get_daily_stats(target_date: str) -> Dict[str, Any]:
    """Obtiene estadísticas de un día específico."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            result,
            COUNT(*) as count
        FROM predictions
        WHERE date = ?
        GROUP BY result
    """, (target_date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    stats = {"win": 0, "loss": 0, "pending": 0}
    for row in rows:
        stats[row["result"]] = row["count"]
    
    total_resolved = stats["win"] + stats["loss"]
    win_rate = (stats["win"] / total_resolved * 100) if total_resolved > 0 else 0
    
    return {
        "wins": stats["win"],
        "losses": stats["loss"],
        "pending": stats["pending"],
        "total": sum(stats.values()),
        "win_rate": round(win_rate, 1)
    }


# Initialize database on module load
init_db()
