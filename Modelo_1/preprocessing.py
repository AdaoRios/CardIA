"""Preprocessamento para os baselines do dataset UCI Heart Disease.

As observacoes com valores faltantes sao excluidas na etapa de avaliacao.
Por isso, estes preprocessadores nao realizam imputacao.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]


def build_preprocessor(
    selected_features: Sequence[str], *, scale_numeric_features: bool
) -> ColumnTransformer:
    """Cria o preprocessor apropriado para um dos modelos baseline.

    Args:
        selected_features: Features que serao utilizadas pelo modelo.
        scale_numeric_features: Quando ``True``, aplica ``StandardScaler`` nas
            variaveis numericas continuas, como requerido pela regressao
            logistica. A Random Forest recebe essas variaveis sem escala.

    Returns:
        Um ``ColumnTransformer`` ainda nao ajustado aos dados.
    """

    selected_features = list(selected_features)
    numeric_features = [
        feature for feature in selected_features if feature in NUMERIC_FEATURES
    ]
    binary_features = [
        feature for feature in selected_features if feature in BINARY_FEATURES
    ]
    categorical_features = [
        feature for feature in selected_features if feature in CATEGORICAL_FEATURES
    ]
    known_features = set(NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES)
    unknown_features = set(selected_features) - known_features
    if unknown_features:
        raise ValueError(f"Features desconhecidas: {sorted(unknown_features)}")

    numeric_transformer = StandardScaler() if scale_numeric_features else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("binary", "passthrough", binary_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )
