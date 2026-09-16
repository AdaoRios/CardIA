"""Carregamento e preparação inicial do dataset Heart Disease."""

from ucimlrepo import fetch_ucirepo


def load_heart_disease():
    """Carrega o dataset Heart Disease da UCI."""

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    # Transformamos o problema original (0-4)
    # em classificação binária:
    # 0 = ausência de doença
    # 1 = presença de doença
    y = (y["num"] > 0).astype(int)

    return X, y


if __name__ == "__main__":
    X, y = load_heart_disease()

    print("Formato das features:", X.shape)
    print("Formato do target:", y.shape)

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(y.head())

    print("\nDistribuição do target:")
    print(y.value_counts())

    print("\nTipos das variáveis:")
    print(X.dtypes)

    print("\nValores faltantes:")
    print(X.isnull().sum())

    print("\nValores únicos:")
    for coluna in X.columns:
        print(f"{coluna}: {X[coluna].unique()}")

    print("\nEstatísticas:")
    print(X.describe())
    print("\nDistribuição das variáveis categóricas:")

    for coluna in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        print(f"\n{coluna}:")
        print(X[coluna].value_counts(dropna=False))