package com.smartbet

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import com.smartbet.api.Opportunity
import com.smartbet.ui.screens.HomeScreen
import com.smartbet.ui.screens.MatchDetailScreen
import com.smartbet.ui.theme.SmartBetTheme
import com.smartbet.viewmodel.MainViewModel

class MainActivity : ComponentActivity() {
    
    private val viewModel: MainViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            SmartBetTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    SmartBetApp(viewModel = viewModel)
                }
            }
        }
    }
}

@Composable
fun SmartBetApp(viewModel: MainViewModel) {
    var selectedOpportunity by remember { mutableStateOf<Opportunity?>(null) }
    val uiState by viewModel.uiState.collectAsState()
    
    if (selectedOpportunity != null) {
        MatchDetailScreen(
            opportunity = selectedOpportunity!!,
            teamStats = uiState.selectedTeamStats,
            isLoading = uiState.isLoading,
            onBack = {
                selectedOpportunity = null
                viewModel.clearSelectedTeam()
            },
            onLoadTeamStats = { teamName ->
                viewModel.loadTeamStats(teamName)
            }
        )
    } else {
        HomeScreen(
            uiState = uiState,
            onOpportunityClick = { opportunity ->
                selectedOpportunity = opportunity
            },
            onRefresh = {
                viewModel.loadOpportunities()
            }
        )
    }
}
