"""Constantes del proyecto de sueño.

Un solo lugar para los nombres, la semilla, el target y las columnas que se usan
como pistas. Así los scripts de entrenamiento, registro y predicción no repiten
texto que podría quedar distinto en cada archivo.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- nombres MLflow
EXPERIMENTO = "clasificacion-sueno"          # experimento que agrupa los runs
MODELO_REGISTRADO = "clasificador-sueno"     # nombre estable en el Model Registry
MODELO_ALIAS = "champion"                     # alias que consume la API por defecto
SEMILLA = 42                                 # misma partición en cada ejecución

# ---------------------------------------------------------------------- datos
RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "sleep_debt_screen_time.csv"

# Ruta absoluta al almacén de MLflow: funciona aunque se ejecute desde otra carpeta.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
)


def activar_seguimiento():
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)

# Variable objetivo: el tutorial usa un clasificador, así que se predice la
# categoría de deuda de sueño (No Debt / Mild / Moderate / Severe).
OBJETIVO = "sleep_debt_category"

# Variables de comportamiento conocidas ANTES de dormir. Se excluyen a propósito
# las métricas fisiológicas de la misma noche (total_sleep_hours, rem_sleep_pct,
# deep_sleep_pct, sleep_efficiency_pct, awakenings_count,
# sleep_onset_latency_min) porque usarlas sería fuga de información: cuando la
# persona se acuesta todavía no conoce esos valores.
FEATURES = [
    "age",
    "gender",
    "chronotype",
    "occupation",
    "bedtime_hour",
    "screen_time_before_bed_min",
    "primary_app_category",
    "screen_brightness_pct",
    "blue_light_filter_on",
    "caffeine_intake_mg",
    "caffeine_last_hour",
    "physical_activity_min",
    "room_temperature_f",
    "ambient_noise_db",
]

# Tipos de columna para el preprocesamiento.
COLUMNAS_NUMERICAS = [
    "age",
    "bedtime_hour",
    "screen_time_before_bed_min",
    "screen_brightness_pct",
    "blue_light_filter_on",       # ya viene como 0/1 (bool)
    "caffeine_intake_mg",
    "caffeine_last_hour",
    "physical_activity_min",
    "room_temperature_f",
    "ambient_noise_db",
]

COLUMNAS_CATEGORICAS = [
    "gender",
    "chronotype",
    "occupation",
    "primary_app_category",
]
