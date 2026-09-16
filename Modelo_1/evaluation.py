"""Execucao e avaliacao dos baselines UCI Heart Disease do CardIA.

Este experimento e destinado a pesquisa e desenvolvimento. Seus resultados nao
constituem diagnostico nem validacao clinica.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataset import load_heart_disease
from models import build_logistic_regression_pipeline, build_random_forest_pipeline


TEST_SIZE = 0.2
RANDOM_STATE = 42
ALL_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
FEATURE_SETS = {
    "FULL": ALL_FEATURES,
    "STRICT": ["age", "sex"],
    "EXPANDED": ["age", "sex", "cp", "exang"],
}


@dataclass(frozen=True)
class EvaluationResult:
    """Metricas e componentes da matriz de confusao de um modelo."""

    feature_set: str
    name: str
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def remove_incomplete_cases(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Remove casos incompletos sem imputar e preserva o alinhamento de X e y."""

    complete_cases = X.notna().all(axis=1)
    excluded_count = int((~complete_cases).sum())
    return X.loc[complete_cases].copy(), y.loc[complete_cases].copy(), excluded_count


def evaluate_model(
    feature_set: str,
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Calcula as metricas de avaliacao final para uma pipeline ja treinada."""

    probabilities = pipeline.predict_proba(X_test)
    positive_class_index = list(pipeline.classes_).index(1)
    positive_probabilities = probabilities[:, positive_class_index]
    # Threshold fixo do baseline: probabilidade >= 0.5 corresponde a classe 1.
    predictions = (positive_probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return EvaluationResult(
        feature_set=feature_set,
        name=name,
        accuracy=accuracy_score(y_test, predictions),
        sensitivity=recall_score(y_test, predictions),
        specificity=specificity,
        precision=precision_score(y_test, predictions),
        f1=f1_score(y_test, predictions),
        roc_auc=roc_auc_score(y_test, positive_probabilities),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def print_result(result: EvaluationResult) -> None:
    """Exibe as metricas e identifica cada celula da matriz de confusao."""

    print(f"\n{'=' * 50}\n{result.name.upper()}\n{'=' * 50}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Sensitivity/Recall: {result.sensitivity:.4f}")
    print(f"Specificity: {result.specificity:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"F1-score: {result.f1:.4f}")
    print(f"ROC-AUC: {result.roc_auc:.4f}")
    print("\nConfusion Matrix (linhas: real [0, 1]; colunas: previsto [0, 1]):")
    print(f"[[{result.tn} {result.fp}]")
    print(f" [{result.fn} {result.tp}]]")
    print(f"TN={result.tn}, FP={result.fp}, FN={result.fn}, TP={result.tp}")


def run_experiment() -> list[EvaluationResult]:
    """Executa os modelos com os mesmos complete cases e o mesmo split."""

    X, y = load_heart_disease()
    X_complete, y_complete, excluded_count = remove_incomplete_cases(X[ALL_FEATURES], y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_complete,
        y_complete,
        test_size=TEST_SIZE,
        stratify=y_complete,
        random_state=RANDOM_STATE,
    )

    print(f"{'=' * 50}\nDATASET\n{'=' * 50}")
    print(f"Total de observacoes: {len(X)}")
    print(f"Observacoes completas utilizadas: {len(X_complete)}")
    print(f"Observacoes excluidas por missing: {excluded_count}")
    print(f"Treino (complete cases): {len(X_train)}")
    print(f"Teste (complete cases): {len(X_test)}")

    results = []
    reference_train_indices = X_train.index
    reference_test_indices = X_test.index
    for feature_set_name, selected_features in FEATURE_SETS.items():
        assert X_train.index.equals(reference_train_indices)
        assert X_test.index.equals(reference_test_indices)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        pipelines = [
            (
                "Logistic Regression",
                build_logistic_regression_pipeline(selected_features),
            ),
            ("Random Forest", build_random_forest_pipeline(selected_features)),
        ]

        for model_name, pipeline in pipelines:
            pipeline.fit(X_train_selected, y_train)
            result = evaluate_model(
                feature_set_name,
                model_name,
                pipeline,
                X_test_selected,
                y_test,
            )
            results.append(result)
            print_result(result)

    comparison = pd.DataFrame(
        [
            {
                "Feature Set": result.feature_set,
                "Model": result.name,
                "Accuracy": result.accuracy,
                "Sensitivity/Recall": result.sensitivity,
                "Specificity": result.specificity,
                "Precision": result.precision,
                "F1": result.f1,
                "ROC-AUC": result.roc_auc,
            }
            for result in results
        ]
    )
    print(f"\n{'=' * 50}\nCOMPARACAO\n{'=' * 50}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return results


if __name__ == "__main__":
    run_experiment()
