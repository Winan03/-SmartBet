package com.smartbet.api

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

// =============================================================================
// API DATA CLASSES
// =============================================================================

data class HealthResponse(
    val status: String,
    val version: String,
    val models_loaded: Int,
    val available_markets: List<String>
)

data class MarketMetrics(
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val threshold: Double
)

data class Opportunity(
    val id: Int,
    val match: String,
    val league: String,
    val date: String,
    val time: String,
    val market: String,
    val confidence: Double,
    val edge: Double,
    val implied_prob: Double,
    val model_prob: Double,
    val type: String
)

data class MatchPrediction(
    val match_id: String,
    val home_team: String,
    val away_team: String,
    val date: String,
    val league: String,
    val predictions: Map<String, PredictionDetail>,
    val best_bet: String?,
    val best_confidence: Double?,
    val is_value_bet: Boolean
)

data class PredictionDetail(
    val probability: Double?,
    val threshold: Double?,
    val recommended: Boolean?,
    val confidence: String?,
    val error: String?
)

data class TeamStats(
    val team: String,
    val key_factors: List<KeyFactor>,
    val strengths: List<String>,
    val weaknesses: List<String>,
    val prediction_reasoning: String
)

data class KeyFactor(
    val factor: String,
    val value: Double,
    val impact: String
)

data class PredictRequest(
    val home_team: String,
    val away_team: String,
    val home_stats: Map<String, Double>,
    val away_stats: Map<String, Double>
)

// =============================================================================
// API INTERFACE
// =============================================================================

interface SmartBetApi {
    
    @GET("/")
    suspend fun healthCheck(): HealthResponse
    
    @GET("/markets")
    suspend fun getMarkets(): Map<String, MarketMetrics>
    
    @GET("/opportunities")
    suspend fun getOpportunities(
        @Query("min_confidence") minConfidence: Double = 0.68,
        @Query("min_edge") minEdge: Double = 0.12
    ): List<Opportunity>
    
    @GET("/predictions/upcoming")
    suspend fun getUpcomingPredictions(
        @Query("league") league: String? = null,
        @Query("min_confidence") minConfidence: Double = 0.65,
        @Query("date_str") date: String? = null
    ): List<MatchPrediction>
    
    @GET("/stats/{team_name}")
    suspend fun getTeamStats(
        @Path("team_name") teamName: String
    ): TeamStats
    
    @POST("/predict")
    suspend fun predict(
        @Body request: PredictRequest
    ): MatchPrediction
    
    companion object {
        // Cambiar esta URL según el entorno
        private const val BASE_URL = "http://10.0.2.2:8000/"  // Para emulador Android
        // private const val BASE_URL = "http://192.168.1.X:8000/"  // Para dispositivo físico
        
        fun create(): SmartBetApi {
            return Retrofit.Builder()
                .baseUrl(BASE_URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(SmartBetApi::class.java)
        }
    }
}
