package com.smartbet.firebase

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import com.smartbet.MainActivity
import com.smartbet.R

/**
 * Servicio de Firebase Cloud Messaging para notificaciones push.
 * Recibe notificaciones del backend cuando hay oportunidades de alta confianza.
 */
class SmartBetMessagingService : FirebaseMessagingService() {
    
    companion object {
        private const val CHANNEL_ID = "smartbet_opportunities"
        private const val CHANNEL_NAME = "Oportunidades"
        private const val CHANNEL_DESC = "Notificaciones de oportunidades de alta confianza"
    }
    
    override fun onNewToken(token: String) {
        super.onNewToken(token)
        // Enviar token al backend para suscribir el dispositivo
        sendTokenToServer(token)
    }
    
    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)
        
        // Procesar notificación
        message.notification?.let { notification ->
            showNotification(
                title = notification.title ?: "Nueva Oportunidad",
                body = notification.body ?: "Detectamos una oportunidad de alta confianza",
                data = message.data
            )
        }
        
        // Procesar datos adicionales
        message.data.isNotEmpty().let {
            handleDataPayload(message.data)
        }
    }
    
    private fun showNotification(title: String, body: String, data: Map<String, String>) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        
        // Crear canal (Android 8+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = CHANNEL_DESC
                enableVibration(true)
            }
            notificationManager.createNotificationChannel(channel)
        }
        
        // Intent para abrir la app
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            // Pasar datos de la oportunidad
            data["opportunity_id"]?.let { putExtra("opportunity_id", it) }
            data["market"]?.let { putExtra("market", it) }
        }
        
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        // Construir notificación
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification) // Agregar este icono
            .setContentTitle(title)
            .setContentText(body)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .build()
        
        // Mostrar
        notificationManager.notify(System.currentTimeMillis().toInt(), notification)
    }
    
    private fun handleDataPayload(data: Map<String, String>) {
        // Procesar datos de la oportunidad
        val matchId = data["match_id"]
        val market = data["market"]
        val confidence = data["confidence"]?.toDoubleOrNull()
        val edge = data["edge"]?.toDoubleOrNull()
        
        // Log o almacenar para análisis
        android.util.Log.d("SmartBet", "Oportunidad: $matchId - $market ($confidence% conf, $edge% edge)")
    }
    
    private fun sendTokenToServer(token: String) {
        // TODO: Implementar envío de token al backend
        // Esto permitirá enviar notificaciones personalizadas
        android.util.Log.d("SmartBet", "FCM Token: $token")
    }
}
