from ucimlrepo import fetch_ucirepo

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFormato de X:")
print(X.shape)

print("\nFormato de y:")
print(y.shape)

print("\nNomes das variáveis:")
print(X.columns)

print("\nValores do target:")
print(y.value_counts())