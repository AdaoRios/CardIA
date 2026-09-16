"""Construcao dos modelos baseline do CardIA."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


RANDOM_STATE = 42


def build_logistic_regression_pipeline(selected_features: Sequence[str]) -> Pipeline:
    """Cria a pipeline de Regressao Logistica.

    ``max_iter=1000`` e usado apenas para permitir a convergencia com as
    features codificadas; nao representa ajuste de hiperparametros.
    """

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(selected_features, scale_numeric_features=True),
            ),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def build_random_forest_pipeline(selected_features: Sequence[str]) -> Pipeline:
    """Cria a pipeline de Random Forest com estado aleatorio reproduzivel."""

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(selected_features, scale_numeric_features=False),
            ),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )
