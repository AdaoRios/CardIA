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


@dataclass(frozen=True)
class EvaluationResult:
    """Metricas e componentes da matriz de confusao de um modelo."""

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


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    """Calcula as metricas de avaliacao final para uma pipeline ja treinada."""

    probabilities = pipeline.predict_proba(X_test)
    positive_class_index = list(pipeline.classes_).index(1)
    positive_probabilities = probabilities[:, positive_class_index]
    # Threshold fixo do baseline: probabilidade >= 0.5 corresponde a classe 1.
    predictions = (positive_probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return EvaluationResult(
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


def run_experiment() -> tuple[EvaluationResult, EvaluationResult]:
    """Executa split estratificado, complete-case analysis, treino e avaliacao.

    O split ocorre no dataset original antes de qualquer transformacao aprendida.
    Em seguida, casos incompletos sao excluidos de cada particicao, sem imputacao.
    Assim, scaler, encoder e modelos sao ajustados exclusivamente no treino.
    """

    X, y = load_heart_disease()
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, y_train, excluded_train = remove_incomplete_cases(X_train_raw, y_train_raw)
    X_test, y_test, excluded_test = remove_incomplete_cases(X_test_raw, y_test_raw)

    print(f"{'=' * 50}\nDATASET\n{'=' * 50}")
    print(f"Total de observacoes: {len(X)}")
    print(f"Observacoes completas utilizadas: {len(X_train) + len(X_test)}")
    print(f"Observacoes excluidas por missing: {excluded_train + excluded_test}")
    print(f"Treino (complete cases): {len(X_train)}")
    print(f"Teste (complete cases): {len(X_test)}")

    logistic_pipeline = build_logistic_regression_pipeline()
    random_forest_pipeline = build_random_forest_pipeline()
    logistic_pipeline.fit(X_train, y_train)
    random_forest_pipeline.fit(X_train, y_train)

    logistic_result = evaluate_model("Logistic Regression", logistic_pipeline, X_test, y_test)
    forest_result = evaluate_model("Random Forest", random_forest_pipeline, X_test, y_test)
    print_result(logistic_result)
    print_result(forest_result)

    comparison = pd.DataFrame(
        [
            {
                "Modelo": result.name,
                "Accuracy": result.accuracy,
                "Sensitivity/Recall": result.sensitivity,
                "Specificity": result.specificity,
                "Precision": result.precision,
                "F1-score": result.f1,
                "ROC-AUC": result.roc_auc,
            }
            for result in (logistic_result, forest_result)
        ]
    )
    print(f"\n{'=' * 50}\nCOMPARACAO\n{'=' * 50}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return logistic_result, forest_result


if __name__ == "__main__":
    run_experiment()
