"""Carga el dataset de sueño y prepara la partición de entrenamiento y prueba.

Si cambian los datos o las columnas, se cambia solo este archivo. El resto de
scripts recibe siempre las mismas cuatro variables: X_train, X_test, y_train,
y_test.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import FEATURES, OBJETIVO, RUTA_DATOS, SEMILLA


def cargar_dataframe():
    """Devuelve el CSV completo, con los nombres de columna limpios."""
    df = pd.read_csv(RUTA_DATOS)
    df.columns = [columna.strip() for columna in df.columns]
    return df


def cargar_datos(test_size=0.2):
    """Devuelve X_train, X_test, y_train, y_test con la misma partición siempre.

    Se usa `stratify=y` para que las cuatro categorías de deuda de sueño queden
    representadas en entrenamiento y prueba, incluso la clase minoritaria
    "Severe" (menos del 1 % de las filas).
    """
    df = cargar_dataframe()
    X = df[FEATURES].copy()
    y = df[OBJETIVO].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEMILLA,
        stratify=y,
    )
