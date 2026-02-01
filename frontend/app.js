// SmartBet App JavaScript
// Connects to backend API for real data

const API_BASE = window.location.origin;
let currentDate = new Date();
let predictions = [];
let currentPrediction = null;
let selectedHistoryTeam = 'home';
let selectedUpcomingTeam = 'home';

const DAYS = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
const MONTHS = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateDateDisplay();
    loadPredictions();
});

function formatDate(d) {
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function updateDateDisplay() {
    const today = new Date();
    const isToday = formatDate(currentDate) === formatDate(today);
    const isYesterday = formatDate(new Date(today.getTime() - 86400000)) === formatDate(currentDate);

    document.getElementById('dateNumber').textContent = String(currentDate.getDate()).padStart(2, '0');
    document.getElementById('dateText').textContent = `${DAYS[currentDate.getDay()]}, ${currentDate.getDate()} de ${MONTHS[currentDate.getMonth()]}`;

    const badge = document.getElementById('dateBadge');
    if (isToday) {
        badge.textContent = 'HOY';
        badge.className = 'date-badge today';
    } else {
        const daysAgo = Math.floor((today - currentDate) / 86400000);
        badge.textContent = isYesterday ? 'AYER' : `HACE ${daysAgo} DÍAS`;
        badge.className = 'date-badge past';
    }
    document.getElementById('nextBtn').disabled = isToday;
}

function changeDate(delta) {
    currentDate = new Date(currentDate.getTime() + delta * 86400000);
    updateDateDisplay();
    loadPredictions();
}

async function loadPredictions() {
    const container = document.getElementById('predictionsContainer');
    const summaryCard = document.getElementById('summaryCard');
    container.innerHTML = '<div class="loading-container"><div class="spinner"></div><div class="loading-text">Cargando...</div></div>';

    const dateStr = formatDate(currentDate);
    const isToday = dateStr === formatDate(new Date());

    try {
        const health = await (await fetch(`${API_BASE}/`)).json();
        document.getElementById('statusIndicator').classList.add('online');
        document.getElementById('statusText').textContent = `${health.models_loaded} modelos`;
        document.getElementById('totalMarkets').textContent = health.available_markets.length;

        if (isToday) {
            predictions = await (await fetch(`${API_BASE}/opportunities?min_confidence=0.65&min_edge=0.10`)).json();
            summaryCard.classList.remove('visible');
        } else {
            const histData = await (await fetch(`${API_BASE}/history/${dateStr}`)).json();
            predictions = histData.predictions || [];
            if (predictions.length > 0 && histData.stats) {
                document.getElementById('winsCount').textContent = histData.stats.wins || 0;
                document.getElementById('lossesCount').textContent = histData.stats.losses || 0;
                document.getElementById('dayWinRate').textContent = `${histData.stats.win_rate || 0}%`;
                summaryCard.classList.add('visible');
            } else {
                summaryCard.classList.remove('visible');
            }
        }

        document.getElementById('totalPredictions').textContent = predictions.length;
        document.getElementById('predictionCount').textContent = predictions.length;
        renderPredictions(isToday);
    } catch (e) {
        console.error(e);
        document.getElementById('statusIndicator').classList.remove('online');
        document.getElementById('statusText').textContent = 'Desconectado';
        container.innerHTML = '<div class="empty-state"><div class="empty-icon"><span class="icon">cloud_off</span></div><div class="empty-title">Error de conexión</div></div>';
    }
}

function renderPredictions(isToday) {
    const container = document.getElementById('predictionsContainer');
    if (predictions.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon"><span class="icon">event_busy</span></div><div class="empty-title">Sin predicciones</div></div>';
        return;
    }

    container.innerHTML = predictions.map((pred, idx) => {
        const match = pred.match || `${pred.home} vs ${pred.away}`;
        const conf = pred.confidence || pred.model_prob;
        const confClass = conf >= 0.70 ? 'high' : 'medium';
        const resultChip = !isToday && pred.result ? `<div class="result-chip ${pred.result}"><span class="icon icon-sm">${pred.result === 'win' ? 'check_circle' : 'cancel'}</span>${pred.result === 'win' ? 'Acierto' : 'Fallo'}</div>` : `<span class="match-time">${pred.time}</span>`;

        return `
            <article class="prediction-card" onclick="openModal(${idx})">
                <div class="prediction-header">
                    <div class="league-info">
                        <div class="league-icon"><span class="icon" style="font-size:14px">sports_soccer</span></div>
                        <span class="league-name">${pred.league}</span>
                    </div>
                    ${resultChip}
                </div>
                <div class="match-teams">${match}</div>
                <div class="prediction-footer">
                    <span class="market-chip">${pred.market}</span>
                    <div class="confidence-display">
                        <div class="confidence-bar-container"><div class="confidence-bar ${confClass}" style="width:${conf * 100}%"></div></div>
                        <span class="confidence-value">${Math.round(conf * 100)}%</span>
                    </div>
                </div>
            </article>
        `;
    }).join('');
}

function openModal(index) {
    currentPrediction = predictions[index];
    const p = currentPrediction;
    const match = p.match || `${p.home} vs ${p.away}`;
    const conf = p.confidence || p.model_prob || 0.70;
    const edge = p.edge || 0.20;
    const modelProb = p.model_prob || conf;
    const houseProb = p.implied_prob || 0.50;

    document.getElementById('modalLeague').textContent = p.league;
    document.getElementById('modalTeams').textContent = match;
    document.getElementById('modalTime').textContent = p.time;
    document.getElementById('modalMarket').textContent = p.market;
    document.getElementById('modalConfidence').textContent = Math.round(conf * 100) + '%';
    document.getElementById('modalModelProb').textContent = Math.round(modelProb * 100) + '%';
    document.getElementById('modalHouseProb').textContent = Math.round(houseProb * 100) + '%';
    document.getElementById('modalEdge').textContent = '+' + Math.round(edge * 100) + '%';
    document.getElementById('modalReasoning').textContent = p.reasoning || 'Análisis basado en estadísticas históricas.';

    // Animate ring
    const offset = 283 - (283 * conf);
    setTimeout(() => document.getElementById('ringFill').style.strokeDashoffset = offset, 100);

    // Edge bar
    document.getElementById('edgeFill').style.width = Math.min(edge * 200, 100) + '%';

    // Factors
    const factors = p.factors || { corners_home: 6.2, corners_away: 4.1, goals_avg: 2.8, xG: 3.1 };
    document.getElementById('modalFactors').innerHTML = Object.entries(factors).slice(0, 4).map(([k, v]) => `
        <div class="factor-item">
            <div class="factor-label">${formatFactorName(k)}</div>
            <div class="factor-value ${parseFloat(v) > 5 ? 'positive' : ''}">${v}</div>
        </div>
    `).join('');

    // Team names for tabs
    const home = p.home || match.split(' vs ')[0];
    const away = p.away || match.split(' vs ')[1];
    document.getElementById('homeTeamName').textContent = home;
    document.getElementById('awayTeamName').textContent = away;
    document.getElementById('upHomeTeamName').textContent = home;
    document.getElementById('upAwayTeamName').textContent = away;

    switchTab('analysis');
    document.getElementById('modalOverlay').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('modalOverlay').classList.remove('active');
    document.body.style.overflow = '';
    document.getElementById('ringFill').style.strokeDashoffset = 283;
}

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tab));

    if (tab === 'history') {
        selectTeam(selectedHistoryTeam);
    } else if (tab === 'upcoming') {
        selectUpcoming(selectedUpcomingTeam);
    }
}

async function selectTeam(team) {
    selectedHistoryTeam = team;
    document.getElementById('btnHome').classList.toggle('active', team === 'home');
    document.getElementById('btnH2H').classList.toggle('active', team === 'h2h');
    document.getElementById('btnAway').classList.toggle('active', team === 'away');

    const list = document.getElementById('historyList');
    list.innerHTML = '<div class="loading-container"><div class="spinner"></div></div>';

    const match = currentPrediction?.match || `${currentPrediction?.home} vs ${currentPrediction?.away}`;
    const home = currentPrediction?.home || match.split(' vs ')[0];
    const away = currentPrediction?.away || match.split(' vs ')[1];

    try {
        if (team === 'h2h') {
            // Fetch real H2H data from API
            const res = await fetch(`${API_BASE}/h2h/${encodeURIComponent(home)}/${encodeURIComponent(away)}?last=5`);
            const data = await res.json();

            list.innerHTML = data.matches.map(m => `
                <div class="history-item" onclick="openStatsModal('${m.date}', '${m.home} vs ${m.away}', '${m.score}', ${m.fixture_id})">
                    <div class="history-item-header">
                        <span class="history-date">${m.date}</span>
                    </div>
                    <div class="history-teams">${m.home} vs ${m.away}</div>
                    <div class="history-score">${m.score}</div>
                </div>
            `).join('');
        } else {
            // Fetch real team fixtures from API
            const teamName = team === 'home' ? home : away;
            const res = await fetch(`${API_BASE}/team/${encodeURIComponent(teamName)}/fixtures?last=10&upcoming=false`);
            const data = await res.json();

            list.innerHTML = data.fixtures.map(m => `
                <div class="history-item" onclick="openStatsModal('${m.date}', '${teamName} vs ${m.opponent}', '${m.score}', ${m.fixture_id})">
                    <div class="history-item-header">
                        <span class="history-date">${m.date}</span>
                        <span class="history-result ${m.result}">${m.result?.toUpperCase() || ''}</span>
                    </div>
                    <div class="history-teams">${m.venue === 'H' ? teamName : m.opponent} vs ${m.venue === 'A' ? teamName : m.opponent}</div>
                    <div class="history-score">${m.score}</div>
                </div>
            `).join('');
        }
    } catch (e) {
        console.error(e);
        list.innerHTML = '<div class="empty-state"><div class="empty-title">Error cargando datos</div></div>';
    }
}

async function selectUpcoming(team) {
    selectedUpcomingTeam = team;
    document.getElementById('btnUpHome').classList.toggle('active', team === 'home');
    document.getElementById('btnUpAway').classList.toggle('active', team === 'away');

    const list = document.getElementById('upcomingList');
    list.innerHTML = '<div class="loading-container"><div class="spinner"></div></div>';

    const match = currentPrediction?.match || `${currentPrediction?.home} vs ${currentPrediction?.away}`;
    const home = currentPrediction?.home || match.split(' vs ')[0];
    const away = currentPrediction?.away || match.split(' vs ')[1];
    const teamName = team === 'home' ? home : away;

    try {
        const res = await fetch(`${API_BASE}/team/${encodeURIComponent(teamName)}/fixtures?last=5&upcoming=true`);
        const data = await res.json();

        list.innerHTML = data.fixtures.map(m => `
            <div class="upcoming-item">
                <div class="upcoming-date">${m.date} - ${m.time}</div>
                <div class="upcoming-teams">${m.venue === 'H' ? teamName : m.opponent} vs ${m.venue === 'A' ? teamName : m.opponent}</div>
                <div class="upcoming-league">${m.league}</div>
            </div>
        `).join('');
    } catch (e) {
        console.error(e);
        list.innerHTML = '<div class="empty-state"><div class="empty-title">Error cargando datos</div></div>';
    }
}

async function openStatsModal(date, match, score, fixtureId) {
    document.getElementById('statsMatchTitle').textContent = match;
    document.getElementById('statsScore').textContent = score;
    document.getElementById('statsGrid').innerHTML = '<div class="loading-container"><div class="spinner"></div></div>';
    document.getElementById('statsModalOverlay').classList.add('active');

    try {
        const res = await fetch(`${API_BASE}/team/stats/${fixtureId}`);
        const data = await res.json();

        const stats = data.stats;
        const home = stats.home || {};
        const awayStats = stats.away || {};

        const statsList = [
            { label: 'Posesión', home: home.possession || 50, away: awayStats.possession || 50, pct: true },
            { label: 'Corners', home: home.corners || 5, away: awayStats.corners || 4 },
            { label: 'Remates', home: home.shots || 12, away: awayStats.shots || 10 },
            { label: 'A Puerta', home: home.shots_on_target || 5, away: awayStats.shots_on_target || 4 },
            { label: 'Atajadas', home: home.saves || 3, away: awayStats.saves || 4 }
        ];

        document.getElementById('statsGrid').innerHTML = statsList.map(s => {
            const total = s.home + s.away || 1;
            const pctHome = (s.home / total) * 100;
            const pctAway = (s.away / total) * 100;
            return `
                <div class="stat-row">
                    <span class="stat-val">${s.pct ? s.home + '%' : s.home}</span>
                    <div class="stat-bar-left"><div class="stat-bar-fill" style="width:${pctHome}%"></div></div>
                    <span class="stat-label-center">${s.label}</span>
                    <div class="stat-bar-right"><div class="stat-bar-fill" style="width:${pctAway}%"></div></div>
                    <span class="stat-val">${s.pct ? s.away + '%' : s.away}</span>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.error(e);
        document.getElementById('statsGrid').innerHTML = '<div class="empty-state"><div class="empty-title">Error cargando estadísticas</div></div>';
    }
}

function closeStatsModal() {
    document.getElementById('statsModalOverlay').classList.remove('active');
}

function formatFactorName(key) {
    const names = {
        corners_home: 'Corners Local',
        corners_away: 'Corners Visit.',
        goals_home: 'Goles Local',
        goals_away: 'Goles Visit.',
        xG_total: 'xG Total',
        goals_avg: 'Media Goles',
        xG: 'xG',
        total_avg: 'Promedio'
    };
    return names[key] || key.replace(/_/g, ' ');
}
