"""Entrena un clasificador de deuda de sueño y registra todo en MLflow.

Sigue el flujo del tutorial de referencia (parámetros, métricas, artefactos,
dataset y modelo), adaptado al dataset `sleep_debt_screen_time.csv`.

La variable objetivo es `sleep_debt_category`:
    No Debt / Mild / Moderate / Severe

Las pistas son solo variables de comportamiento conocidas antes de dormir.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # genera gráficos sin abrir ventanas (servidores y scripts)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    EXPERIMENTO,
    FEATURES,
    OBJETIVO,
    RUTA_DATOS,
    SEMILLA,
)
from data import cargar_datos


def construir_modelo(n_estimators, max_depth, min_samples_leaf, class_weight):
    """Pipeline: imputación + escalado + one-hot + Random Forest.

    El pipeline se guarda junto con el modelo, así que al servir la API se pueden
    enviar las columnas originales (texto incluido) y el preprocesamiento se
    aplica automáticamente.
    """
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                COLUMNAS_NUMERICAS,
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                COLUMNAS_CATEGORICAS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clasificador = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=SEMILLA,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", clasificador),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento con seguimiento en MLflow")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--class-weight",
        default="balanced_subsample",
        help="balanced_subsample, balanced o none",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--run-name", default=None, help="nombre legible del run")
    args = parser.parse_args()

    # `mlflow run` ya asigna el experimento; con `python src/train.py` lo asignamos aquí.
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO)

    class_weight = None if args.class_weight.lower() == "none" else args.class_weight

    X_train, X_test, y_train, y_test = cargar_datos(test_size=args.test_size)

    nombre_run = args.run_name or (
        f"rf_n{args.n_estimators}_d{args.max_depth}_leaf{args.min_samples_leaf}"
    )

    with mlflow.start_run(run_name=nombre_run) as run:
        # 1) Qué se probó
        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "class_weight": args.class_weight,
                "test_size": args.test_size,
                "seed": SEMILLA,
                "target": OBJETIVO,
                "n_features": len(FEATURES),
            }
        )
        mlflow.set_tags(
            {
                "modelo": "RandomForestClassifier",
                "dataset": "sleep_debt_screen_time",
                "etapa": "desarrollo",
                "target": OBJETIVO,
            }
        )

        # 2) Entrenar
        modelo = construir_modelo(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=class_weight,
        ).fit(X_train, y_train)

        # 3) Evaluar con datos que el modelo no vio y guardar las métricas
        pred = modelo.predict(X_test)
        pred_train = modelo.predict(X_train)

        metricas = {
            "accuracy": accuracy_score(y_test, pred),
            "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
            "precision_macro": precision_score(
                y_test, pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "accuracy_train": accuracy_score(y_train, pred_train),
            "f1_macro_train": f1_score(
                y_train, pred_train, average="macro", zero_division=0
            ),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        mlflow.log_metrics(metricas)

        # 4) Artefactos: gráficos, reporte y tabla de muestra
        etiquetas = list(modelo.named_steps["modelo"].classes_)

        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            pred,
            labels=etiquetas,
            display_labels=etiquetas,
            cmap="Blues",
            ax=ax,
            xticks_rotation=30,
            colorbar=False,
        )
        ax.set_title("Matriz de confusión - deuda de sueño")
        mlflow.log_figure(fig, "matriz_confusion.png")
        plt.close(fig)

        preprocesador = modelo.named_steps["preprocesador"]
        nombres = [
            nombre.split("__", 1)[-1]
            for nombre in preprocesador.get_feature_names_out()
        ]
        importancias = (
            pd.Series(modelo.named_steps["modelo"].feature_importances_, index=nombres)
            .sort_values(ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        importancias.sort_values().plot.barh(ax=ax, color="#55A868")
        ax.set_title("Variables más importantes (top 15)")
        ax.set_xlabel("Importancia")
        mlflow.log_figure(fig, "importancias.png")
        plt.close(fig)

        mlflow.log_dict(
            classification_report(
                y_test, pred, labels=etiquetas, zero_division=0, output_dict=True
            ),
            "reporte_clasificacion.json",
        )
        mlflow.log_dict(
            {
                "target": OBJETIVO,
                "clases": etiquetas,
                "distribucion_train": y_train.value_counts().to_dict(),
                "distribucion_test": y_test.value_counts().to_dict(),
                "features": FEATURES,
            },
            "resumen_datos.json",
        )
        mlflow.log_table(
            data=X_test.head(25).assign(
                real=y_test.head(25).values,
                prediccion=pred[:25],
            ),
            artifact_file="muestra_predicciones.json",
        )
        mlflow.log_text(
            "Entrenamiento del clasificador de deuda de sueño.\n"
            "Se usan solo variables de comportamiento para evitar fuga de información.\n"
            f"Métricas de prueba: accuracy={metricas['accuracy']:.3f}, "
            f"f1_macro={metricas['f1_macro']:.3f}.\n",
            "notas.txt",
        )

        # 5) El dataset que se usó (huella y esquema)
        df_train = X_train.copy()
        df_train[OBJETIVO] = y_train
        dataset_train = mlflow.data.from_pandas(
            df_train,
            source=str(RUTA_DATOS),
            name="sueno-train",
            targets=OBJETIVO,
        )
        mlflow.log_input(dataset_train, context="training")

        df_test = X_test.copy()
        df_test[OBJETIVO] = y_test
        dataset_test = mlflow.data.from_pandas(
            df_test,
            source=str(RUTA_DATOS),
            name="sueno-test",
            targets=OBJETIVO,
        )
        mlflow.log_input(dataset_test, context="testing")

        # 6) El modelo, con su firma y un ejemplo de entrada
        info = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(X_train, modelo.predict(X_train)),
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
        )
        mlflow.set_tag("model_uri", info.model_uri)  # para encontrar el modelo después
        mlflow.set_tag(
            "mlflow.note.content",
            "Random Forest sobre hábitos de la noche anterior. "
            "Target: sleep_debt_category. Sin variables fisiológicas de la misma noche.",
        )

        print(
            f"accuracy={metricas['accuracy']:.3f}  "
            f"f1_macro={metricas['f1_macro']:.3f}  "
            f"model_uri={info.model_uri}"
        )
        print(f"run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
