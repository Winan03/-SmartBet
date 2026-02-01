package com.smartbet.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.smartbet.api.Opportunity
import com.smartbet.api.SmartBetApi
import com.smartbet.api.TeamStats
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class UiState(
    val isLoading: Boolean = false,
    val opportunities: List<Opportunity> = emptyList(),
    val selectedTeamStats: TeamStats? = null,
    val error: String? = null,
    val modelsLoaded: Int = 0,
    val availableMarkets: List<String> = emptyList()
)

class MainViewModel : ViewModel() {
    
    private val api = SmartBetApi.create()
    
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState
    
    init {
        loadInitialData()
    }
    
    private fun loadInitialData() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true)
            
            try {
                // Health check
                val health = api.healthCheck()
                _uiState.value = _uiState.value.copy(
                    modelsLoaded = health.models_loaded,
                    availableMarkets = health.available_markets
                )
                
                // Load opportunities
                loadOpportunities()
                
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Error conectando al servidor: ${e.message}"
                )
            }
        }
    }
    
    fun loadOpportunities(minConfidence: Double = 0.68) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            
            try {
                val opportunities = api.getOpportunities(minConfidence = minConfidence)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    opportunities = opportunities
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Error cargando oportunidades: ${e.message}"
                )
            }
        }
    }
    
    fun loadTeamStats(teamName: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            
            try {
                val stats = api.getTeamStats(teamName)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    selectedTeamStats = stats
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Error cargando estadísticas: ${e.message}"
                )
            }
        }
    }
    
    fun clearError() {
        _uiState.value = _uiState.value.copy(error = null)
    }
    
    fun clearSelectedTeam() {
        _uiState.value = _uiState.value.copy(selectedTeamStats = null)
    }
}
