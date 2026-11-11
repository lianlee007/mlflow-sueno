"""Mueve el alias `champion` a una versión del modelo registrado.

Ejemplos:
    python src/promote.py                  # promueve la versión que apunta 'challenger'
    python src/promote.py --version 1      # rollback a una versión concreta

Mover un alias no copia ni borra el modelo: solo cambia a qué versión apunta el
nombre. El código de producción sigue pidiendo `models:/...@champion`.
"""

import argparse

from mlflow import MlflowClient

from config import MODELO_REGISTRADO


def main():
    parser = argparse.ArgumentParser(description="Promover o revertir un modelo")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--desde-alias",
        default="challenger",
        help="alias que se quiere promover (por defecto: challenger)",
    )
    grupo.add_argument(
        "--version",
        type=str,
        default=None,
        help="versión concreta a la que mover champion (para rollback)",
    )
    args = parser.parse_args()

    cliente = MlflowClient()

    if args.version:
        version = args.version
        origen = f"versión {version}"
        cliente.update_model_version(
            MODELO_REGISTRADO,
            version,
            description=f"Promovida manualmente a champion desde {origen}.",
        )
    else:
        mv = cliente.get_model_version_by_alias(MODELO_REGISTRADO, args.desde_alias)
        version = mv.version
        origen = f"alias '{args.desde_alias}'"

    cliente.set_registered_model_alias(MODELO_REGISTRADO, "champion", version)
    print(f"champion ahora apunta a la versión {version} ({origen}).")


if __name__ == "__main__":
    main()
