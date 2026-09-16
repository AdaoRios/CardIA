"""Preprocessamento para os baselines do dataset UCI Heart Disease.

As observacoes com valores faltantes sao excluidas na etapa de avaliacao.
Por isso, estes preprocessadores nao realizam imputacao.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]


def build_preprocessor(*, scale_numeric_features: bool) -> ColumnTransformer:
    """Cria o preprocessor apropriado para um dos modelos baseline.

    Args:
        scale_numeric_features: Quando ``True``, aplica ``StandardScaler`` nas
            variaveis numericas continuas, como requerido pela regressao
            logistica. A Random Forest recebe essas variaveis sem escala.

    Returns:
        Um ``ColumnTransformer`` ainda nao ajustado aos dados.
    """

    numeric_transformer = StandardScaler() if scale_numeric_features else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
