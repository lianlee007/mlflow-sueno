"""Carga un modelo del registro por alias y predice sobre un archivo CSV.

El CSV debe tener las mismas columnas de entrada que se usaron al entrenar
(las 14 variables de comportamiento). Si además trae la columna objetivo, se
conserva en la salida para comparar, pero no se envía al modelo.
"""

import argparse

import mlflow
import pandas as pd

from config import FEATURES, MODELO_REGISTRADO, OBJETIVO


def main():
    parser = argparse.ArgumentParser(description="Predicción con un modelo del registro")
    parser.add_argument(
        "csv",
        help="archivo con las mismas columnas que se usaron al entrenar",
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="alias del modelo (champion, challenger...)",
    )
    parser.add_argument(
        "--output",
        default="predicciones.csv",
        help="archivo CSV de salida",
    )
    args = parser.parse_args()

    modelo = mlflow.pyfunc.load_model(f"models:/{MODELO_REGISTRADO}@{args.alias}")
    datos = pd.read_csv(args.csv)
    datos.columns = [columna.strip() for columna in datos.columns]

    faltantes = [columna for columna in FEATURES if columna not in datos.columns]
    if faltantes:
        raise SystemExit(
            "Faltan columnas en el CSV: " + ", ".join(faltantes)
            + "\nEl modelo espera: " + ", ".join(FEATURES)
        )

    datos["prediccion"] = modelo.predict(datos[FEATURES])

    print(datos["prediccion"].value_counts().sort_index().to_string())
    datos.to_csv(args.output, index=False)
    print(f"Guardado en {args.output}")

    if OBJETIVO in datos.columns:
        aciertos = (datos[OBJETIVO] == datos["prediccion"]).mean()
        print(f"Coincidencia con la columna real '{OBJETIVO}': {aciertos:.1%}")


if __name__ == "__main__":
    main()
