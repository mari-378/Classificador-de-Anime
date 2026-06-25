# Análise Comparativa entre Regressão Linear e Naive Bayes na Classificação de Animes

Este repositório contém o código-fonte e o conjunto de dados utilizados para os experimentos do artigo científico intitulado **"Análise Comparativa entre Regressão Linear e Naive Bayes na Classificação de Animes"**.

O objetivo principal do estudo é avaliar o desempenho de dois algoritmos de Machine Learning na tarefa de classificação binária de animes, predizendo se uma obra será **"Bem Avaliada" (Score >= 7)** ou **"Mal Avaliada" (Score < 7)** com base em características estruturais e de engajamento do público.

---

## 📁 Estrutura do Projeto

O projeto está organizado da seguinte forma dentro da pasta principal:

```text
código_artigo/
│
├── dados/
│   └── anime_dataset.csv       # Dataset contendo os dados
│
├── naive_bayes.py              # Script de execução do modelo Naive Bayes Gaussiano
└── regressao_linear.py         # Script de execução do modelo de Regressão Linear