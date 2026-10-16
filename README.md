# ARTIFICIAL INTELLIGENCE & MACHINE LEARNING

Este repositório contém notebooks desenvolvidos para a disciplina de Inteligência Artificial & Machine Learning no segundo semestre, abrangendo detecção de anomalias, classificação supervisionada com dados desbalanceados, comparação de algoritmos e redes neurais com PyTorch.

Abaixo estão os detalhes de cada projeto:

## CHECKPOINT 4 (CP4)

**Projeto: Detecção de fraudes em cartão de crédito**

Comparação entre uma abordagem supervisionada e uma não supervisionada em um problema de classe extremamente rara.

- **Descrição:**
  - **Análise exploratória:** medição do desbalanceamento (0,17% de fraudes), verificação de tipos, escala e duplicados, e discussão de por que a acurácia não basta para avaliar o modelo.
  - **Regressão Logística (supervisionado):** busca sobre penalização L1, L2 e ElasticNet, quatro valores de `C` e `class_weight` padrão ou balanceado, com e sem PCA, usando divisão treino/validação/teste (60/20/20).
  - **Isolation Forest (não supervisionado):** variação de `n_estimators`, `contamination` (incluindo a taxa real de fraude) e `max_samples`, com e sem PCA.
  - **Comparação crítica:** a Regressão Logística sem PCA obteve o melhor F1 no teste (0,747) e o Isolation Forest detectou 27 das 99 fraudes sem usar rótulos. Discussão de por que o PCA piorou os dois modelos e de como combinar as abordagens.
- **Dataset:** [Credit Card Fraud Detection — ULB / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284.807 transações de cartões europeus (set/2013), 30 atributos (`Time`, `Amount` e `V1` a `V28`, já transformados por PCA) e alvo binário. O arquivo tem 150 MB e não é versionado: o notebook baixa automaticamente uma cópia pública idêntica na primeira execução.
- **Tecnologias:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (LogisticRegression, IsolationForest, PCA, StandardScaler, Pipeline).
- **Arquivo Principal:** `CP4-AI.ipynb`

## CHALLENGE

**Projeto: Previsão de evasão da rede oficial Ford (VIN Share) — Ford FIAP 2026, Sprint 3**

Modelo preditivo para identificar veículos com risco de deixar a rede de concessionárias no pós-venda e transformá-los em leads de serviço.

- **Descrição:**
  - **Base sintética ancorada na frota real:** 20.000 veículos gerados com a composição da frota Ford no Brasil (SENATRAN/RENAVAM, jul/2026), com perfil de uso, relacionamento com a rede e problemas de qualidade inseridos de propósito.
  - **Preparação:** tratamento de duplicados, grafias inconsistentes, valores impossíveis e outliers (IQR), criação de variáveis de negócio (`em_garantia`, `nunca_revisou`, `respondeu_nps`, entre outras) e pré-processamento com `ColumnTransformer`.
  - **Modelagem:** comparação de Regressão Logística, Random Forest, XGBoost e Rede Neural (MLP) com busca em grade e validação cruzada, usando ROC-AUC como métrica principal.
  - **Resultado:** Regressão Logística selecionada (AUC de 0,785 no teste, com o menor sobreajuste). Com ponto de corte de 38%, identifica 79,7% das evasões; a análise de razão de chances explica cada lead ao consultor.
  - Proposta de deploy via API REST, com geração diária de leads por concessionária e consulta individual.
- **Dataset:** gerado dentro do próprio notebook, a partir de dados públicos do [RENAVAM](https://dados.transportes.gov.br/dataset/registro-nacional-de-veiculos-automotores-renavam).
- **Tecnologias:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (LogisticRegression, RandomForestClassifier, MLPClassifier, GridSearchCV, ColumnTransformer, permutation_importance), XGBoost.
- **Arquivo Principal:** `CHALLENGE/CHALLENGE-AI.ipynb`

## AULAS

Notebooks desenvolvidos em sala, em ordem cronológica, introduzindo redes neurais com PyTorch.

| Notebook | Conteúdo |
|---|---|
| `AULAS/01-PRIMEIRA-REDE-NEURAL.ipynb` | Rede com uma camada oculta (3 neurônios, sigmoid) classificando o dataset `make_circles` e visualização da fronteira de decisão |
| `AULAS/02-CAMADA-OCULTA-E-OVERFITTING.ipynb` | Mesmo problema com mais ruído e 100 neurônios ocultos: erro de treino × teste e visualização da saída de cada neurônio |
| `AULAS/03-MLP-MNIST.ipynb` | MLP de três camadas (784 → 256 → 128 → 10) classificando dígitos manuscritos do MNIST |
| `AULAS/04-CNN-MNIST.ipynb` | Rede convolucional no MNIST: acurácia, contagem de parâmetros, exemplos de acertos e erros e mapas de ativação |

<br>

# TECNOLOGIAS

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23189FDD.svg?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
