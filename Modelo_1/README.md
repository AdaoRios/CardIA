# CardIA - Modelo 1

## Experimento 1 — Redução de Features e Compatibilidade com a Jornada CardIA

### Objetivo

Avaliar experimentalmente quanto do poder preditivo do dataset UCI Heart Disease permanece quando reduzimos o conjunto de variáveis para aquelas que possuem correspondência com informações potencialmente disponíveis na jornada da CardIA.

### FULL

Utiliza as 13 features originais do UCI:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

### STRICT

Utiliza somente:

`age`, `sex`

Representa um conjunto extremamente reduzido de informações demográficas/básicas.

### EXPANDED

Utiliza:

`age`, `sex`, `cp`, `exang`

A ideia é representar, de forma experimental, informações que podem ter correspondência com a jornada/questionário cardiovascular da CardIA.

`cp` representa o tipo de dor torácica do dataset UCI, enquanto `exang` representa angina induzida por exercício. Existe uma correspondência conceitual dessas variáveis com perguntas cardiovasculares do projeto, mas elas ainda não são obtidas diretamente pelo questionário da CardIA. Essa correspondência precisará ser formalizada e validada antes de transformar respostas reais do questionário nessas variáveis do UCI.

A variável `trestbps` **não foi incluída no EXPANDED**. No dataset UCI, ela representa a pressão arterial sistólica em repouso. O rPPG do projeto ainda não deve ser considerado equivalente à pressão arterial. A utilização de um biomarcador rPPG como proxy de pressão arterial exigirá validação técnica específica.

## Metodologia

- **Dataset:** UCI Heart Disease, ID 45.
- **Observações originais:** 303.
- **Features:** 13.
- **Target binário:**
  - `0` = ausência de doença;
  - `1` = presença de doença.
- **Valores faltantes:** 6 observações excluídas.
- **Observações completas utilizadas:** 297.
- **Estratégia:** complete-case analysis, sem imputação.
- **Observações de treino:** 237.
- **Observações de teste:** 60.
- **TEST_SIZE:** `0.2`.
- **RANDOM_STATE:** `42`.

As observações completas foram identificadas considerando todas as 13 features antes da divisão dos dados. O mesmo conjunto de 297 observações foi utilizado pelos três feature sets, e o mesmo train/test split foi reutilizado entre eles.

Essa decisão permite comparar o efeito da seleção de features sem misturá-lo com diferenças na composição da amostra ou no conjunto de teste.

Não foi realizada imputação para os valores faltantes. Em particular, os valores ausentes de `ca` e `thal` não foram preenchidos por média, mediana, moda ou outro valor.

## Modelos

Foram avaliados:

- Logistic Regression;
- Random Forest.

A Logistic Regression utiliza `StandardScaler` nas variáveis numéricas. A Random Forest recebe as variáveis numéricas sem escala. As variáveis categóricas são codificadas por `OneHotEncoder`, com tratamento de categorias desconhecidas.

O preprocessing permanece dentro do `Pipeline`. Dessa forma, o scaler e o encoder são ajustados somente nos dados de treino, evitando data leakage em relação ao conjunto de teste.

As métricas baseadas em classes utilizam threshold fixo de `0.5`. O ROC-AUC é calculado a partir das probabilidades produzidas por `predict_proba`.

## Resultados

| Feature Set | Model | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| FULL | Logistic Regression | 0.8167 | 0.7143 | 0.9062 | 0.8696 | 0.7843 | 0.9375 |
| FULL | Random Forest | 0.8500 | 0.8214 | 0.8750 | 0.8519 | 0.8364 | 0.9208 |
| STRICT | Logistic Regression | 0.7167 | 0.6786 | 0.7500 | 0.7037 | 0.6909 | 0.8047 |
| STRICT | Random Forest | 0.6333 | 0.7500 | 0.5312 | 0.5833 | 0.6562 | 0.7165 |
| EXPANDED | Logistic Regression | 0.8333 | 0.7143 | 0.9375 | 0.9091 | 0.8000 | 0.8778 |
| EXPANDED | Random Forest | 0.7500 | 0.6429 | 0.8438 | 0.7826 | 0.7059 | 0.8817 |

## Interpretação

1. A utilização apenas de `age` e `sex` apresentou desempenho inferior ao conjunto FULL nas métricas avaliadas.
2. A inclusão de `cp` e `exang` no conjunto EXPANDED aumentou o desempenho em relação ao STRICT em várias métricas, indicando que essas variáveis carregam informação preditiva adicional no dataset utilizado.
3. Na Logistic Regression, o ROC-AUC passou de `0.8047` no STRICT para `0.8778` no EXPANDED.
4. Na Random Forest, o ROC-AUC passou de `0.7165` no STRICT para `0.8817` no EXPANDED.
5. O conjunto FULL continua contendo informações adicionais não presentes no EXPANDED. Portanto, o experimento não demonstra que as quatro features são suficientes para reproduzir todo o comportamento do modelo completo.
6. Os resultados não devem ser interpretados como desempenho clínico da CardIA ou como capacidade diagnóstica validada.
7. O experimento foi realizado em um dataset pequeno, histórico e específico, utilizando uma única divisão treino/teste. Portanto, os resultados podem sofrer variabilidade amostral e precisam de validação adicional.
8. O objetivo principal foi metodológico: entender a relação entre redução de features e perda ou ganho de informação preditiva.

## Principais aprendizados para a CardIA

- Não precisamos necessariamente alimentar o futuro modelo com todas as variáveis clínicas disponíveis em uma base externa.
- É importante identificar quais informações estarão realmente disponíveis na jornada da CardIA.
- Variáveis demográficas isoladas podem conter sinal preditivo, mas não representam necessariamente toda a informação clínica relevante.
- Perguntas cardiovasculares adicionais podem acrescentar informação preditiva.
- O desenho final deve separar claramente:
  1. dados provenientes do questionário;
  2. biomarcadores derivados do rPPG;
  3. outras informações clínicas eventualmente disponíveis;
  4. target utilizado para treinamento.
- Antes de incorporar qualquer biomarcador, precisamos definir exatamente como ele será calculado, sua unidade, qualidade do sinal, tratamento de valores ausentes e validação.
- A variável rPPG não deve ser tratada automaticamente como substituta de uma variável clínica apenas porque existe uma hipótese de correlação.

## Limitações

- Dataset com apenas 303 observações.
- Apenas 297 observações completas utilizadas.
- Apenas 60 observações no teste.
- Avaliação baseada em um único split.
- Não foi realizada cross-validation neste experimento.
- Não houve validação externa.
- Não houve validação clínica.
- As variáveis `cp` e `exang` ainda não foram derivadas de respostas reais do questionário CardIA.
- O dataset UCI não representa necessariamente a população-alvo da CardIA.
- Os resultados não devem ser generalizados diretamente para usuários reais.

## Próximos passos

1. Consolidar o aprendizado deste experimento.
2. Estruturar os dados que realmente queremos utilizar na CardIA.
3. Definir claramente os grupos de variáveis:
   - questionário/anamnese;
   - biomarcadores rPPG;
   - dados demográficos;
   - target.
4. Definir o formato do dataset final de treinamento.
5. Definir o preprocessing de cada tipo de variável.
6. Continuar o desenvolvimento e validação do pipeline rPPG.
7. Definir quais biomarcadores rPPG serão candidatos ao modelo.
8. Construir uma base de dados adequada para treinamento.
9. Posteriormente avaliar modelos com validação mais robusta, incluindo cross-validation.
10. Somente depois estudar otimização de hiperparâmetros e modelos mais complexos.

Este README apresenta o experimento como um estudo exploratório e metodológico, não como uma validação clínica.
