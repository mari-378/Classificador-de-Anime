import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Carrega o dataset de animes
anime = pd.read_csv("dados/anime_dataset.csv")

# 2. Tratamento de dados
anime["score"] = pd.to_numeric(anime["score"], errors="coerce")
anime["episodes"] = pd.to_numeric(anime["episodes"], errors="coerce")
anime["members"] = pd.to_numeric(anime["members"], errors="coerce")

# Remove as linhas que possuem valores nulos nas colunas
anime = anime.dropna(subset=["score", "episodes", "members"])

# Garante que episódios seja inteiro
anime["episodes"] = anime["episodes"].astype(int)

# 3. Problema Binário
anime["target"] = (anime["score"] >= 7).astype(int)

# Features escolhidas
X = anime[["episodes", "members"]].values
y = anime["target"].values

# 4. Split treino/teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Treinamento com Regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("=== RESULTADOS: REGRESSÃO LINEAR ===")
print(f"Registros processados: {len(anime)}")
print(f"Pesos:      w1={modelo.coef_[0]:.6f} (episódios), w2={modelo.coef_[1]:.10f} (membros)")
print(f"Intercepto: b ={modelo.intercept_:.3f}")

# 6. Predição e Threshold
y_pred_continuo = modelo.predict(X_test)
y_pred = (y_pred_continuo >= 0.5).astype(int)

# 7. Avaliação
print(f"\nAcurácia: {accuracy_score(y_test, y_pred):.2%}")
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# 8. Visualização fronteira de decisão
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min(), np.percentile(X[:, 0], 95), 300),
    np.linspace(X[:, 1].min(), np.percentile(X[:, 1], 95), 300),
)

Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z >= 0.5, alpha=0.2, cmap="coolwarm")
plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Score < 7", alpha=0.4, s=10)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Score >= 7", alpha=0.4, s=10)

plt.xlim(0, np.percentile(X[:, 0], 95))
plt.ylim(0, np.percentile(X[:, 1], 95))
plt.xlabel("Número de Episódios")
plt.ylabel("Número de Membros")
plt.title("Classificação de Anime (Regressão Linear)")
plt.legend()
plt.show()