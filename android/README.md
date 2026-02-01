# SmartBet Android App

## Estructura del Proyecto

Este directorio contiene los archivos necesarios para crear la app Android.

## Requisitos

- Android Studio Arctic Fox o superior
- Kotlin 1.8+
- Jetpack Compose
- Retrofit para API calls

## Setup

1. Abre Android Studio
2. Crea nuevo proyecto: `Empty Compose Activity`
3. Nombre: `SmartBet`
4. Package: `com.smartbet.app`
5. Copia los archivos de este directorio

## Archivos

```
app/
├── src/main/
│   ├── java/com/smartbet/
│   │   ├── MainActivity.kt
│   │   ├── api/
│   │   │   └── SmartBetApi.kt
│   │   ├── models/
│   │   │   └── Models.kt
│   │   ├── ui/
│   │   │   ├── screens/
│   │   │   │   ├── HomeScreen.kt
│   │   │   │   └── MatchDetailScreen.kt
│   │   │   └── components/
│   │   │       └── PieChart.kt
│   │   └── viewmodel/
│   │       └── MainViewModel.kt
│   └── AndroidManifest.xml
├── build.gradle.kts
└── google-services.json (para Firebase)
```

## Dependencias (build.gradle.kts)

```kotlin
dependencies {
    // Compose
    implementation("androidx.compose.ui:ui:1.5.0")
    implementation("androidx.compose.material3:material3:1.1.0")
    
    // Retrofit
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    
    // ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.0")
    
    // Firebase (opcional)
    implementation("com.google.firebase:firebase-messaging:23.0.0")
}
```
