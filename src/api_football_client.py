"""
SmartBet API-Football Integration
----------------------------------
Módulo para obtener datos adicionales de API-Football:
- Lesiones de equipos
- Alineaciones y jugadores clave
- Estadísticas por mitad (corners, remates, posesión)
- Head-to-Head histórico
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time


class APIFootballClient:
    """
    Cliente para API-Football vía RapidAPI.
    Documentación: https://www.api-football.com/documentation-v3
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        self.requests_remaining = None
        self.last_request_time = None
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Realiza una petición a la API con rate limiting."""
        # Rate limiting: máximo 10 requests por minuto en plan gratuito
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < 6:  # Esperar al menos 6 segundos entre requests
                time.sleep(6 - elapsed)
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.last_request_time = datetime.now()
            
            # Actualizar requests restantes
            self.requests_remaining = response.headers.get('x-ratelimit-requests-remaining')
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    # =========================================================================
    # FIXTURES & MATCHES
    # =========================================================================
    
    def get_fixtures_by_date(self, date: str, league_id: int = None) -> list:
        """
        Obtiene partidos de una fecha específica.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            league_id: ID de liga (opcional). Ej: 39=Premier League, 140=La Liga
        
        Returns:
            Lista de fixtures
        """
        params = {"date": date}
        if league_id:
            params["league"] = league_id
            
        data = self._make_request("fixtures", params)
        return data.get("response", []) if data else []
    
    def get_fixture_by_id(self, fixture_id: int) -> dict:
        """Obtiene detalles de un partido específico."""
        data = self._make_request("fixtures", {"id": fixture_id})
        response = data.get("response", []) if data else []
        return response[0] if response else None
    
    # =========================================================================
    # INJURIES
    # =========================================================================
    
    def get_injuries(self, fixture_id: int = None, team_id: int = None, 
                     league_id: int = None, season: int = None) -> list:
        """
        Obtiene lesiones.
        
        Args:
            fixture_id: ID del partido
            team_id: ID del equipo
            league_id: ID de la liga
            season: Temporada (ej: 2024)
        
        Returns:
            Lista de lesiones con jugador, tipo y estado
        """
        params = {}
        if fixture_id:
            params["fixture"] = fixture_id
        if team_id:
            params["team"] = team_id
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
            
        data = self._make_request("injuries", params)
        return data.get("response", []) if data else []
    
    def count_injuries_by_team(self, team_id: int, league_id: int, season: int) -> int:
        """Cuenta el número de jugadores lesionados en un equipo."""
        injuries = self.get_injuries(team_id=team_id, league_id=league_id, season=season)
        return len(injuries)
    
    # =========================================================================
    # LINEUPS
    # =========================================================================
    
    def get_lineups(self, fixture_id: int) -> dict:
        """
        Obtiene alineaciones de un partido.
        
        Returns:
            Dict con alineaciones de ambos equipos, formación y jugadores
        """
        data = self._make_request("fixtures/lineups", {"fixture": fixture_id})
        response = data.get("response", []) if data else []
        
        if len(response) >= 2:
            return {
                "home": response[0],
                "away": response[1]
            }
        return None
    
    # =========================================================================
    # FIXTURE STATISTICS (CORNERS, SHOTS BY HALF)
    # =========================================================================
    
    def get_fixture_statistics(self, fixture_id: int) -> dict:
        """
        Obtiene estadísticas detalladas de un partido.
        Incluye: Corners, Remates, Posesión, Tarjetas, etc.
        """
        data = self._make_request("fixtures/statistics", {"fixture": fixture_id})
        response = data.get("response", []) if data else []
        
        if len(response) >= 2:
            home_stats = self._parse_stats(response[0].get("statistics", []))
            away_stats = self._parse_stats(response[1].get("statistics", []))
            return {
                "home": home_stats,
                "away": away_stats
            }
        return None
    
    def _parse_stats(self, stats_list: list) -> dict:
        """Convierte lista de estadísticas a diccionario."""
        return {stat["type"]: stat["value"] for stat in stats_list}
    
    # =========================================================================
    # HEAD TO HEAD
    # =========================================================================
    
    def get_h2h(self, team1_id: int, team2_id: int, last: int = 5) -> list:
        """
        Obtiene historial de enfrentamientos directos.
        
        Returns:
            Lista de los últimos N partidos entre ambos equipos
        """
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last
        }
        data = self._make_request("fixtures/headtohead", params)
        return data.get("response", []) if data else []
    
    def get_h2h_corners_avg(self, team1_id: int, team2_id: int, last: int = 5) -> float:
        """Calcula promedio de corners en enfrentamientos directos."""
        h2h = self.get_h2h(team1_id, team2_id, last)
        
        if not h2h:
            return 10.0  # Default
        
        total_corners = 0
        count = 0
        
        for match in h2h:
            fixture_id = match.get("fixture", {}).get("id")
            if fixture_id:
                stats = self.get_fixture_statistics(fixture_id)
                if stats:
                    home_corners = stats["home"].get("Corner Kicks", 0) or 0
                    away_corners = stats["away"].get("Corner Kicks", 0) or 0
                    total_corners += home_corners + away_corners
                    count += 1
        
        return total_corners / count if count > 0 else 10.0
    
    # =========================================================================
    # TEAM SEARCH
    # =========================================================================
    
    def search_team(self, name: str, league_id: int = None) -> list:
        """Busca un equipo por nombre."""
        params = {"search": name}
        if league_id:
            params["league"] = league_id
            
        data = self._make_request("teams", params)
        return data.get("response", []) if data else []
    
    def get_team_id(self, name: str) -> int:
        """Obtiene el ID de un equipo por nombre."""
        results = self.search_team(name)
        if results:
            return results[0].get("team", {}).get("id")
        return None
    
    # =========================================================================
    # LEAGUES
    # =========================================================================
    
    LEAGUE_IDS = {
        "E0": 39,     # Premier League
        "SP1": 140,   # La Liga
        "D1": 78,     # Bundesliga
        "I1": 135,    # Serie A
        "F1": 61,     # Ligue 1
    }
    
    def get_league_id(self, code: str) -> int:
        """Convierte código de liga a ID de API-Football."""
        return self.LEAGUE_IDS.get(code)


class SmartBetAPIEnricher:
    """
    Enriquece datos de partidos con información de API-Football.
    """
    
    def __init__(self, api_key: str):
        self.client = APIFootballClient(api_key)
        self.cache = {}
        
    def enrich_match(self, home_team: str, away_team: str, 
                     match_date: str, league_code: str = "E0") -> dict:
        """
        Enriquece un partido con datos adicionales.
        
        Returns:
            Dict con:
            - home_injuries_count
            - away_injuries_count
            - h2h_corners_avg
            - home_key_players_missing
            - away_key_players_missing
        """
        enriched = {
            "home_injuries_count": 0,
            "away_injuries_count": 0,
            "h2h_corners_avg": 10.0,
            "home_form_api": 0.0,
            "away_form_api": 0.0
        }
        
        try:
            # Obtener IDs de equipos
            home_id = self.client.get_team_id(home_team)
            away_id = self.client.get_team_id(away_team)
            league_id = self.client.get_league_id(league_code)
            
            if not home_id or not away_id:
                print(f"Teams not found: {home_team}, {away_team}")
                return enriched
            
            # Obtener conteo de lesiones
            season = int(match_date[:4])  # Año de la fecha
            enriched["home_injuries_count"] = self.client.count_injuries_by_team(
                home_id, league_id, season
            )
            enriched["away_injuries_count"] = self.client.count_injuries_by_team(
                away_id, league_id, season
            )
            
            # H2H corners average
            enriched["h2h_corners_avg"] = self.client.get_h2h_corners_avg(home_id, away_id)
            
        except Exception as e:
            print(f"Error enriching match: {e}")
        
        return enriched
    
    def enrich_dataframe(self, df: pd.DataFrame, 
                         home_col: str = "home_team",
                         away_col: str = "away_team",
                         date_col: str = "date",
                         league_col: str = "league") -> pd.DataFrame:
        """
        Enriquece un DataFrame completo con datos de API.
        NOTA: Consume requests de API, usar con cuidado.
        """
        enriched_data = []
        
        for idx, row in df.iterrows():
            home = row[home_col]
            away = row[away_col]
            date = row[date_col].strftime("%Y-%m-%d") if hasattr(row[date_col], 'strftime') else str(row[date_col])
            league = row.get(league_col, "E0")
            
            enriched = self.enrich_match(home, away, date, league)
            enriched_data.append(enriched)
            
            # Progress
            if (idx + 1) % 10 == 0:
                print(f"Enriched {idx + 1}/{len(df)} matches...")
        
        # Merge con DataFrame original
        enriched_df = pd.DataFrame(enriched_data)
        return pd.concat([df.reset_index(drop=True), enriched_df], axis=1)


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Tu API key de RapidAPI
    API_KEY = "f718825e57msh4068446217d642bp14d814jsn0d29c83f4273"
    
    # Crear cliente
    client = APIFootballClient(API_KEY)
    
    # Ejemplo: Buscar equipo
    print("Buscando Barcelona...")
    teams = client.search_team("Barcelona")
    if teams:
        print(f"  ID: {teams[0]['team']['id']}")
        print(f"  Nombre: {teams[0]['team']['name']}")
    
    # Ejemplo: Partidos de hoy
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\nPartidos de hoy ({today}):")
    fixtures = client.get_fixtures_by_date(today, league_id=39)  # Premier League
    for f in fixtures[:5]:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        print(f"  {home} vs {away}")
