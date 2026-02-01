# 🎯 SmartBet - Predicciones Deportivas con ML

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Winan03/-SmartBet)

Sistema inteligente de predicciones de apuestas deportivas utilizando Machine Learning.

## 🚀 Demo

Accede a la aplicación desplegada en Vercel.

## 📊 Características

- **11 Mercados de Predicción**: Over/Under, BTTS, Corners, y más
- **71% Win Rate**: Modelo entrenado con datos de 5 ligas principales
- **Interfaz Premium**: Diseño profesional con animaciones 3D
- **API REST**: Backend FastAPI para integraciones
- **Historial Real**: Conexión con API-Football para datos en tiempo real

## 🛠️ Tecnologías

| Componente | Tecnología |
|------------|------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI, Python |
| ML | XGBoost, LightGBM, Scikit-learn |
| Data | API-Football (RapidAPI) |
| Deploy | Vercel |

## 📁 Estructura del Proyecto

```
SmartBet/
├── backend/
│   ├── main.py          # API FastAPI
│   └── database.py      # SQLite persistence
├── frontend/
│   ├── index.html       # App web
│   ├── styles.css       # Estilos premium
│   └── app.js           # Lógica cliente
├── src/
│   ├── specialized_models_v2.py
│   └── api_football_client.py
├── vercel.json          # Config de despliegue
└── requirements.txt
```

## 🔧 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/Winan03/-SmartBet.git
cd SmartBet

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
cd backend
uvicorn main:app --reload

# Abrir en navegador
# http://localhost:8000/app
```

## ☁️ Despliegue en Vercel

### Opción 1: Deploy automático
1. Click en el botón "Deploy with Vercel" arriba
2. Conecta tu cuenta de GitHub
3. Configura las variables de entorno

### Opción 2: Deploy manual
```bash
# Instalar Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

### Variables de Entorno (Vercel Dashboard)
```
RAPIDAPI_KEY=tu_api_key_de_rapidapi
```

## 🔑 API-Football Setup

1. Regístrate en [RapidAPI](https://rapidapi.com/api-sports/api/api-football)
2. Suscríbete al plan gratuito (100 requests/día)
3. Copia tu API Key
4. Configura en Vercel Dashboard → Settings → Environment Variables

## 📡 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/app` | Aplicación web |
| GET | `/opportunities` | Predicciones del día |
| GET | `/history/{date}` | Predicciones históricas |
| GET | `/team/{name}/fixtures` | Historial de equipo |
| GET | `/h2h/{team1}/{team2}` | Enfrentamientos directos |
| GET | `/stats/summary` | Estadísticas generales |

## 📈 Mercados Soportados

- Over/Under 2.5 Goles
- Over/Under 3.5 Goles
- BTTS (Ambos Equipos Anotan)
- Corners Over 7.5, 8.5, 9.5, 10.5
- Double Chance
- 1X2 (Match Winner)

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE)

## 👤 Autor

- GitHub: [@Winan03](https://github.com/Winan03)
