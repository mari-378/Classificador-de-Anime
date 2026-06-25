import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 1. Carregar o seu dataset de animes
df_anime = pd.read_csv("dados/anime_dataset.csv")

# 2. Tratamento dos dados
df_anime["episodes"] = pd.to_numeric(df_anime["episodes"], errors="coerce")
df_anime["score"] = pd.to_numeric(df_anime["score"], errors="coerce")
df_anime["members"] = pd.to_numeric(df_anime["members"], errors="coerce")

colunas_alvo = ["score", "episodes", "members"]
df_anime = df_anime.dropna(subset=colunas_alvo)

# 3. Criar a variável alvo binária conforme o objetivo:
df_anime["bem_avaliado"] = np.where(df_anime["score"] >= 7, 1, 0)

# Separando as features (X) e o alvo (y)
X = df_anime[["episodes", "members"]].values
y = df_anime["bem_avaliado"].values

# 4. Split treino/teste PADRONIZADO (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Criando e treinando o modelo Naive Bayes Gaussiano
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# 6. Fazendo previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# 7. Calculando e exibindo a acurácia do modelo
print("=== RESULTADOS: NAIVE BAYES ===")
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do Modelo: {acuracia:.2%}")

# Gerando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão (Texto):")
print(cm)

# 8. Visualizando a matriz de confusão graficamente
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Mal Avaliado (< 7)", "Bem Avaliado (>= 7)"],
    yticklabels=["Mal Avaliado (< 7)", "Bem Avaliado (>= 7)"],
)
plt.xlabel("Classificação Prevista")
plt.ylabel("Classificação Real")
plt.title("Matriz de Confusão - Predição de Avaliação de Animes")
plt.show()