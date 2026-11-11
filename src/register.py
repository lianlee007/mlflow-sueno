"""Registra el mejor run del experimento como una versión nueva del modelo.

Por defecto la versión queda con el alias `challenger`. Para pasarla a
producción se mueve el alias `champion` con `src/promote.py`.

Se ordena por `f1_macro` y no por accuracy porque las clases están muy
desbalanceadas: "Severe" es menos del 1 % del dataset y accuracy alta puede
esconder que el modelo nunca acierta esa clase.
"""

import mlflow
from mlflow import MlflowClient

from config import EXPERIMENTO, MODELO_REGISTRADO

mejores = mlflow.search_runs(
    experiment_names=[EXPERIMENTO],
    order_by=["metrics.f1_macro DESC"],
    max_results=20,
)
if "tags.model_uri" not in mejores.columns:
    raise SystemExit(
        f"Ningún run del experimento '{EXPERIMENTO}' tiene la etiqueta model_uri. "
        "Ejecuta antes src/train.py"
    )
mejores = mejores[
    mejores["tags.model_uri"].notna() & (mejores["tags.model_uri"] != "")
]
if mejores.empty:
    raise SystemExit(
        f"No hay runs en el experimento '{EXPERIMENTO}'. Ejecuta antes src/train.py"
    )

mejor = mejores.iloc[0]
uri = mejor["tags.model_uri"]
version = mlflow.register_model(uri, MODELO_REGISTRADO)

cliente = MlflowClient()
cliente.update_registered_model(
    MODELO_REGISTRADO,
    description="Clasifica la deuda de sueño en No Debt / Mild / Moderate / Severe "
    "a partir de hábitos de la noche anterior.",
)
cliente.update_model_version(
    MODELO_REGISTRADO,
    version.version,
    description=(
        f"Run {mejor['run_id']} | accuracy={mejor['metrics.accuracy']:.3f} | "
        f"f1_macro={mejor['metrics.f1_macro']:.3f}"
    ),
)
cliente.set_model_version_tag(
    MODELO_REGISTRADO, version.version, "dataset", "sleep_debt_screen_time"
)
cliente.set_registered_model_alias(
    MODELO_REGISTRADO, "challenger", version.version
)

print(
    f"Registrada la versión {version.version} "
    f"(accuracy={mejor['metrics.accuracy']:.3f}, "
    f"f1_macro={mejor['metrics.f1_macro']:.3f}) como 'challenger'"
)
print("Para promoverla a producción: python src/promote.py")
