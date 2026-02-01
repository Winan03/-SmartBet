package com.smartbet.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.smartbet.api.Opportunity
import com.smartbet.viewmodel.UiState

// =============================================================================
// COLORS
// =============================================================================

val GradientPrimary = Brush.linearGradient(
    colors = listOf(Color(0xFF1A1A2E), Color(0xFF16213E))
)
val AccentGreen = Color(0xFF00D26A)
val AccentGold = Color(0xFFFFD700)
val CardBackground = Color(0xFF1F2937)

// =============================================================================
// HOME SCREEN
// =============================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    uiState: UiState,
    onOpportunityClick: (Opportunity) -> Unit,
    onRefresh: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "SmartBet AI",
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        Text(
                            "${uiState.modelsLoaded} modelos activos",
                            fontSize = 12.sp,
                            color = Color.Gray
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF1A1A2E)
                ),
                actions = {
                    IconButton(onClick = onRefresh) {
                        Icon(
                            Icons.Default.Refresh,
                            contentDescription = "Actualizar",
                            tint = AccentGreen
                        )
                    }
                }
            )
        },
        containerColor = Color(0xFF0F0F23)
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Header Stats
            HeaderStats(uiState)
            
            // Opportunities Section
            Text(
                "🎯 Oportunidades del Día",
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                modifier = Modifier.padding(16.dp)
            )
            
            if (uiState.isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = AccentGreen)
                }
            } else if (uiState.error != null) {
                ErrorMessage(uiState.error)
            } else {
                OpportunityList(
                    opportunities = uiState.opportunities,
                    onOpportunityClick = onOpportunityClick
                )
            }
        }
    }
}

@Composable
fun HeaderStats(uiState: UiState) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        StatCard(
            title = "Oportunidades",
            value = "${uiState.opportunities.size}",
            icon = "🎯"
        )
        StatCard(
            title = "Win Rate",
            value = "71.6%",
            icon = "📈"
        )
        StatCard(
            title = "Mercados",
            value = "${uiState.availableMarkets.size}",
            icon = "⚽"
        )
    }
}

@Composable
fun StatCard(title: String, value: String, icon: String) {
    Card(
        modifier = Modifier.size(100.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(icon, fontSize = 24.sp)
            Text(
                value,
                color = AccentGreen,
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
            Text(
                title,
                color = Color.Gray,
                fontSize = 10.sp
            )
        }
    }
}

@Composable
fun OpportunityList(
    opportunities: List<Opportunity>,
    onOpportunityClick: (Opportunity) -> Unit
) {
    LazyColumn(
        contentPadding = PaddingValues(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        items(opportunities) { opportunity ->
            OpportunityCard(
                opportunity = opportunity,
                onClick = { onOpportunityClick(opportunity) }
            )
        }
    }
}

@Composable
fun OpportunityCard(
    opportunity: Opportunity,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    opportunity.league,
                    color = Color.Gray,
                    fontSize = 12.sp
                )
                Text(
                    "${opportunity.date} ${opportunity.time}",
                    color = Color.Gray,
                    fontSize = 12.sp
                )
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Match Name
            Text(
                opportunity.match,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Market & Confidence
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Market Badge
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(AccentGreen.copy(alpha = 0.2f))
                        .padding(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Text(
                        opportunity.market,
                        color = AccentGreen,
                        fontWeight = FontWeight.Medium,
                        fontSize = 14.sp
                    )
                }
                
                // Confidence
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Default.Star,
                        contentDescription = null,
                        tint = AccentGold,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        "${(opportunity.confidence * 100).toInt()}%",
                        color = AccentGold,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Edge indicator
            LinearProgressIndicator(
                progress = opportunity.edge.toFloat().coerceIn(0f, 1f),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .clip(RoundedCornerShape(2.dp)),
                color = AccentGreen,
                trackColor = Color.Gray.copy(alpha = 0.3f)
            )
            
            Text(
                "Edge: ${(opportunity.edge * 100).toInt()}%",
                color = Color.Gray,
                fontSize = 10.sp,
                modifier = Modifier.padding(top = 4.dp)
            )
        }
    }
}

@Composable
fun ErrorMessage(message: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Card(
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFF4A1515)
            ),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text(
                message,
                color = Color.White,
                modifier = Modifier.padding(16.dp)
            )
        }
    }
}
