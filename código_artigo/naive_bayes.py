import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 1. Carrega o dataset de animes
df_anime = pd.read_csv("dados/anime_dataset.csv")

# 2. Tratamento dos dados
df_anime["episodes"] = pd.to_numeric(df_anime["episodes"], errors="coerce")
df_anime["score"] = pd.to_numeric(df_anime["score"], errors="coerce")
df_anime["members"] = pd.to_numeric(df_anime["members"], errors="coerce")

colunas_alvo = ["score", "episodes", "members"]
df_anime = df_anime.dropna(subset=colunas_alvo)

# 3. Criação da resposta do modelo: 1 para bom (>=7) e 0 para ruim (<7)
df_anime["bem_avaliado"] = np.where(df_anime["score"] >= 7, 1, 0)

# Separação das features (X) e do alvo (y)
X = df_anime[["episodes", "members"]].values
y = df_anime["bem_avaliado"].values

# 4. Split treino/teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Criação e treino do modelo Naive Bayes Gaussiano
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# 6. Previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# 7. Cálculo e exibição da acurácia do modelo
print("=== RESULTADOS: NAIVE BAYES ===")
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do Modelo: {acuracia:.2%}")

# Geração da matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão (Texto):")
print(cm)

# 8. Visualização da matriz de confusão
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