"""Evalúa un modelo ya registrado con `mlflow.models.evaluate`.

Calcula el conjunto estándar de métricas de clasificación y las guarda en un run
nuevo. Sirve como criterio de aceptación antes de promover un modelo a
producción.
"""

import argparse
import os

import mlflow

from config import EXPERIMENTO, MODELO_REGISTRADO, OBJETIVO
from data import cargar_datos


def main():
    parser = argparse.ArgumentParser(description="Evaluar un modelo del registro")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO)

    _, X_test, _, y_test = cargar_datos(test_size=args.test_size)
    datos_prueba = X_test.copy()
    datos_prueba[OBJETIVO] = y_test

    uri = f"models:/{MODELO_REGISTRADO}@{args.alias}"
    nombre = args.run_name or f"evaluacion_{args.alias}"

    with mlflow.start_run(run_name=nombre):
        mlflow.set_tags(
            {
                "etapa": "evaluacion",
                "modelo": MODELO_REGISTRADO,
                "alias": args.alias,
            }
        )
        resultado = mlflow.models.evaluate(
            model=uri,
            data=datos_prueba,
            targets=OBJETIVO,
            model_type="classifier",
        )
        print("Métricas guardadas en el run:")
        for clave, valor in sorted(resultado.metrics.items()):
            if isinstance(valor, (int, float)):
                print(f"  {clave}: {valor:.4f}")


if __name__ == "__main__":
    main()
