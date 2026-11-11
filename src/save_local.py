"""Guarda una copia autocontenida del modelo en una carpeta local.

La carpeta resultante incluye el pipeline, la firma, el ejemplo de entrada y las
dependencias. Se puede mover a otra máquina o servir sin el tracking URI:

    python src/save_local.py
    mlflow models serve -m ./modelo_local -p 5002 --env-manager local
"""

import argparse

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from config import MODELO_REGISTRADO
from data import cargar_datos


def main():
    parser = argparse.ArgumentParser(description="Guardar el modelo en una carpeta")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--output", default="modelo_local")
    args = parser.parse_args()

    uri = f"models:/{MODELO_REGISTRADO}@{args.alias}"
    modelo = mlflow.sklearn.load_model(uri)

    X_train, _, _, _ = cargar_datos()
    mlflow.sklearn.save_model(
        modelo,
        path=args.output,
        signature=infer_signature(X_train, modelo.predict(X_train)),
        input_example=X_train.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
    )
    print(f"Modelo guardado en ./{args.output}")
    print(
        "Para servirlo sin tracking URI: "
        f"mlflow models serve -m ./{args.output} -p 5002 --env-manager local"
    )


if __name__ == "__main__":
    main()
