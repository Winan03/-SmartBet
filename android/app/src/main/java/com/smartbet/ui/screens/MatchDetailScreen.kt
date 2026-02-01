package com.smartbet.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.smartbet.api.Opportunity
import com.smartbet.api.TeamStats

// =============================================================================
// MATCH DETAIL SCREEN
// =============================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MatchDetailScreen(
    opportunity: Opportunity,
    teamStats: TeamStats?,
    isLoading: Boolean,
    onBack: () -> Unit,
    onLoadTeamStats: (String) -> Unit
) {
    val scrollState = rememberScrollState()
    
    // Load team stats when screen opens
    LaunchedEffect(opportunity) {
        val homeTeam = opportunity.match.split(" vs ").firstOrNull()
        if (homeTeam != null && teamStats == null) {
            onLoadTeamStats(homeTeam)
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Detalle del Partido", color = Color.White) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = "Volver",
                            tint = Color.White
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF1A1A2E)
                )
            )
        },
        containerColor = Color(0xFF0F0F23)
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(scrollState)
                .padding(16.dp)
        ) {
            // Match Header
            MatchHeader(opportunity)
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Probability Chart
            ProbabilityChart(
                modelProb = opportunity.model_prob,
                impliedProb = opportunity.implied_prob
            )
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Team Stats Section
            if (isLoading) {
                Box(
                    modifier = Modifier.fillMaxWidth(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = AccentGreen)
                }
            } else if (teamStats != null) {
                TeamStatsSection(teamStats)
            }
        }
    }
}

@Composable
fun MatchHeader(opportunity: Opportunity) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                opportunity.league,
                color = Color.Gray,
                fontSize = 14.sp
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                opportunity.match,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                "${opportunity.date} - ${opportunity.time}",
                color = Color.Gray,
                fontSize = 14.sp
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Recommendation Badge
            Box(
                modifier = Modifier
                    .background(
                        AccentGreen.copy(alpha = 0.2f),
                        RoundedCornerShape(8.dp)
                    )
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Text(
                    "✅ ${opportunity.market}",
                    color = AccentGreen,
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                )
            }
        }
    }
}

@Composable
fun ProbabilityChart(
    modelProb: Double,
    impliedProb: Double
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                "📊 Modelo vs Casa de Apuestas",
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
            
            Spacer(modifier = Modifier.height(20.dp))
            
            // Pie Chart
            Box(
                modifier = Modifier.size(200.dp),
                contentAlignment = Alignment.Center
            ) {
                PieChart(
                    data = listOf(
                        PieChartData("Modelo", modelProb.toFloat(), AccentGreen),
                        PieChartData("Casa", impliedProb.toFloat(), Color(0xFFFF6B6B))
                    )
                )
                
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        "Edge",
                        color = Color.Gray,
                        fontSize = 12.sp
                    )
                    Text(
                        "+${((modelProb - impliedProb) * 100).toInt()}%",
                        color = AccentGreen,
                        fontWeight = FontWeight.Bold,
                        fontSize = 24.sp
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(20.dp))
            
            // Legend
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                LegendItem("Modelo: ${(modelProb * 100).toInt()}%", AccentGreen)
                LegendItem("Casa: ${(impliedProb * 100).toInt()}%", Color(0xFFFF6B6B))
            }
        }
    }
}

data class PieChartData(
    val label: String,
    val value: Float,
    val color: Color
)

@Composable
fun PieChart(data: List<PieChartData>) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val total = data.sumOf { it.value.toDouble() }.toFloat()
        var startAngle = -90f
        
        val strokeWidth = 30.dp.toPx()
        val radius = (size.minDimension - strokeWidth) / 2
        val center = Offset(size.width / 2, size.height / 2)
        
        data.forEach { pieData ->
            val sweepAngle = (pieData.value / total) * 360f
            
            drawArc(
                color = pieData.color,
                startAngle = startAngle,
                sweepAngle = sweepAngle,
                useCenter = false,
                topLeft = Offset(center.x - radius, center.y - radius),
                size = Size(radius * 2, radius * 2),
                style = Stroke(width = strokeWidth)
            )
            
            startAngle += sweepAngle
        }
    }
}

@Composable
fun LegendItem(label: String, color: Color) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(12.dp)
                .background(color, RoundedCornerShape(2.dp))
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            label,
            color = Color.White,
            fontSize = 14.sp
        )
    }
}

@Composable
fun TeamStatsSection(stats: TeamStats) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
            Text(
                "📈 ¿Por qué esta predicción?",
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                stats.prediction_reasoning,
                color = Color.LightGray,
                fontSize = 14.sp,
                lineHeight = 22.sp
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Key Factors
            Text(
                "Factores Clave:",
                color = AccentGreen,
                fontWeight = FontWeight.Medium,
                fontSize = 14.sp
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            stats.key_factors.forEach { factor ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(factor.factor, color = Color.Gray, fontSize = 13.sp)
                    Text(
                        factor.value.toString(),
                        color = if (factor.impact == "high") AccentGreen else Color.White,
                        fontWeight = FontWeight.Medium,
                        fontSize = 13.sp
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Strengths
            if (stats.strengths.isNotEmpty()) {
                Text("✅ Fortalezas:", color = AccentGreen, fontSize = 13.sp)
                stats.strengths.forEach { strength ->
                    Text("  • $strength", color = Color.LightGray, fontSize = 12.sp)
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Weaknesses
            if (stats.weaknesses.isNotEmpty()) {
                Text("⚠️ Debilidades:", color = Color(0xFFFF6B6B), fontSize = 13.sp)
                stats.weaknesses.forEach { weakness ->
                    Text("  • $weakness", color = Color.LightGray, fontSize = 12.sp)
                }
            }
        }
    }
}
