# Tech Challenge - Fase 02 | Machine Learning Engineering

## Sobre o projeto

Este projeto foi desenvolvido para o **Tech Challenge da Fase 02** da Pós-Graduação em **Machine Learning Engineering** da FIAP.

A ideia principal foi construir um projeto de recomendação de ponta a ponta, não ficando apenas no treinamento do modelo. Além da parte de Machine Learning, também foram aplicadas práticas de organização de código, versionamento de dados, rastreamento de experimentos, testes e execução com Docker.

O foco foi criar uma solução simples, mas bem estruturada, que pudesse ser reproduzida por outra pessoa sem depender apenas do ambiente local.

---

## Problema

O problema proposto simula o cenário de uma empresa de e-commerce que quer recomendar produtos para seus usuários com base no comportamento de interação.

Como base para o projeto, foi utilizado o dataset MovieLens. Apesar de ser um dataset de filmes, a estrutura é parecida com um problema de recomendação de produtos: existem usuários, itens e interações entre eles.

A partir dessas interações, o objetivo foi treinar um modelo capaz de prever se uma interação entre usuário e item tende a ser positiva ou negativa.

---

## Objetivos

Os principais objetivos deste projeto foram:

- preparar um dataset para um problema de recomendação;
- criar um target binário a partir das avaliações dos usuários;
- implementar um baseline com Scikit-Learn;
- desenvolver um modelo neural com PyTorch;
- organizar o pipeline com DVC;
- registrar experimentos e métricas com MLflow;
- criar testes automatizados;
- rodar o pipeline em Docker;
- documentar os resultados, limitações e próximos passos.

---

## Dataset

### Dataset escolhido

Foi utilizado o **MovieLens Latest Small**, baixado via KaggleHub.

### Por que esse dataset?

O MovieLens é bastante usado em estudos e projetos de recomendação. Ele tem uma estrutura simples de entender, mas suficiente para trabalhar conceitos importantes, como interação usuário-item, matriz de recomendação, target supervisionado e modelos baseados em embeddings.

Mesmo não sendo um dataset real de e-commerce, ele funciona bem para simular esse tipo de problema.

### Arquivos utilizados

Os dados brutos ficam em:

```text
data/raw/movielens_100k/
```

Principais arquivos:

| Arquivo | Descrição |
|---|---|
| `ratings.csv` | Avaliações dos usuários para os filmes |
| `movies.csv` | Informações dos filmes |
| `tags.csv` | Tags atribuídas por usuários |
| `links.csv` | Identificadores externos dos filmes |

O arquivo mais importante para o treinamento foi o `ratings.csv`.

### Colunas principais

| Coluna | Descrição |
|---|---|
| `userId` | Identificador do usuário |
| `movieId` | Identificador do item |
| `rating` | Nota dada pelo usuário |
| `timestamp` | Momento da interação |

---

## Definição do target

Como o dataset possui avaliações numéricas, foi necessário transformar o problema em uma classificação binária.

A regra usada foi:

| Regra | Target |
|---|---|
| `rating >= 4` | 1, interação positiva |
| `rating < 4` | 0, interação negativa |

Assim, o modelo tenta prever se uma interação entre usuário e item seria positiva ou não.

---

## Solução implementada

O projeto foi organizado como um pipeline de Machine Learning com quatro etapas principais:

```text
download_data -> preprocess -> feature_eng -> train
```

Cada etapa tem uma responsabilidade:

1. **download_data**: baixa o dataset;
2. **preprocess**: trata as interações e cria a base processada;
3. **feature_eng**: cria os mapeamentos de usuários e itens e separa treino/teste;
4. **train**: treina o modelo em PyTorch e salva métricas e artefatos.

O pipeline foi configurado com DVC para facilitar a reprodução dos resultados.

---

## Estrutura do projeto

```text
.
├── Dockerfile
├── README.md
├── docker-compose.yml
├── dvc.lock
├── dvc.yaml
├── main.py
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── data
│   ├── external
│   ├── features
│   │   ├── item_mapping.csv
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── user_mapping.csv
│   ├── processed
│   │   └── interactions.csv
│   └── raw
│       └── movielens_100k
│           ├── README.txt
│           ├── links.csv
│           ├── movies.csv
│           ├── ratings.csv
│           └── tags.csv
├── models
│   └── recommender_net.pt
├── notebooks
├── reports
│   ├── baseline_metrics.json
│   ├── dataset_notes.md
│   ├── figures
│   └── torch_metrics.json
├── scripts
│   ├── download_data.py
│   └── validate_env.py
├── src
│   ├── __init__.py
│   ├── configs
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── preprocess.py
│   ├── evaluation
│   │   └── metrics.py
│   └── models
│       ├── baseline.py
│       ├── factory.py
│       ├── recommender.py
│       ├── train_torch.py
│       └── trainer.py
└── tests
    ├── test_feature_engineering.py
    ├── test_metrics.py
    └── test_preprocess.py
```

---

## Organização das pastas

| Pasta | Finalidade |
|---|---|
| `data/raw/` | Dados brutos baixados do MovieLens |
| `data/processed/` | Base tratada após o preprocessamento |
| `data/features/` | Bases de treino, teste e mapeamentos |
| `src/configs/` | Configurações do projeto |
| `src/data/` | Scripts de preprocessamento e feature engineering |
| `src/evaluation/` | Funções de métricas |
| `src/models/` | Código dos modelos e treinamento |
| `scripts/` | Scripts auxiliares |
| `reports/` | Métricas, anotações e documentações |
| `models/` | Modelo treinado |
| `tests/` | Testes automatizados |

Os arquivos gerados em `data/` e `models/` não são versionados diretamente pelo Git. Eles são controlados pelo DVC, para evitar subir arquivos grandes ou gerados automaticamente no repositório.

---

## Tecnologias utilizadas

- Python 3.12
- Poetry
- Pandas
- NumPy
- Scikit-Learn
- PyTorch
- MLflow
- DVC
- Docker
- Docker Compose
- Ruff
- Pre-commit
- Pytest
- Pydantic Settings
- KaggleHub

---

## Como configurar o projeto

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd tech-challenge-fase-02
```

### 2. Instalar as dependências

```bash
poetry install
```

### 3. Criar o arquivo `.env`

Use o `.env.example` como base:

```bash
cp .env.example .env
```

Exemplo:

```env
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DVC_REMOTE=local
```

### 4. Validar o ambiente

```bash
poetry run python scripts/validate_env.py
```

Saída esperada:

```text
Environment validation: OK
```

---

## Como rodar o pipeline localmente

### Download dos dados

```bash
poetry run python scripts/download_data.py
```

Saída esperada:

```text
Dataset downloaded to: data/raw/movielens_100k
```

### Preprocessamento

```bash
poetry run python -m src.data.preprocess
```

Saída esperada:

```text
Processed data saved to: data/processed/interactions.csv
Rows: 100836
```

### Feature engineering

```bash
poetry run python -m src.data.feature_engineering
```

Saída esperada:

```text
Train data saved to: data/features/train.csv
Test data saved to: data/features/test.csv
Train rows: 80668
Test rows: 20168
```

### Treinamento do modelo

```bash
poetry run python -m src.models.train_torch --run-name local_recommender_net --device cpu
```

---

## Como rodar com DVC

O DVC foi usado para organizar e reproduzir as etapas do pipeline.

Para rodar tudo de uma vez:

```bash
poetry run dvc repro
```

Para verificar se existe alguma etapa desatualizada:

```bash
poetry run dvc status
```

Para visualizar o fluxo do pipeline:

```bash
poetry run dvc dag
```

Fluxo esperado:

```text
+---------------+
| download_data |
+---------------+
        *
        *
        *
  +------------+
  | preprocess |
  +------------+
        *
        *
        *
 +-------------+
 | feature_eng |
 +-------------+
        *
        *
        *
    +-------+
    | train |
    +-------+
```

---

## Como rodar com Docker

O projeto também pode ser executado dentro de um container Docker.

### Construir a imagem

```bash
docker compose build
```

### Rodar o pipeline no container

```bash
docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db pipeline
```

Esse comando roda o pipeline DVC dentro do container e usa um backend SQLite para o MLflow.

Ao final da execução, a saída esperada inclui:

```text
Early stopping triggered.
PyTorch metrics saved to: reports/torch_metrics.json
Model saved to: models/recommender_net.pt
Use `dvc push` to send your updates to remote storage.
```

---

## MLflow

O MLflow foi usado para registrar os experimentos do modelo.

Durante o treinamento, são registrados:

- parâmetros do treino;
- métricas finais;
- artefatos gerados;
- modelo treinado.

Na execução com Docker, foi usado SQLite como backend do MLflow, para evitar depender de um servidor externo durante a execução do pipeline.

As métricas finais ficam em:

```text
reports/torch_metrics.json
```

O modelo treinado fica em:

```text
models/recommender_net.pt
```

O uso do MLflow Model Registry ficou como uma melhoria futura. Nesta versão, o foco foi garantir o tracking dos experimentos e o salvamento dos artefatos.

---

## Modelos implementados

### Baseline com Scikit-Learn

Foi criado um modelo baseline com Scikit-Learn para ter uma referência inicial de desempenho.

As métricas do baseline são salvas em:

```text
reports/baseline_metrics.json
```

### Modelo neural com PyTorch

O modelo principal foi uma rede neural baseada em embeddings.

A ideia foi transformar usuários e itens em vetores aprendidos durante o treinamento. Esses vetores são combinados e passam por camadas densas para prever se a interação tende a ser positiva.

Essa abordagem foi escolhida porque embeddings são comuns em problemas de recomendação e permitem que o modelo aprenda relações entre usuários e itens sem depender apenas de regras manuais.

---

## Resultados

O modelo final foi treinado com os seguintes parâmetros principais:

| Parâmetro | Valor |
|---|---:|
| Batch size | 512 |
| Épocas máximas | 20 |
| Learning rate | 0.001 |
| Embedding dim | 32 |
| Hidden dim | 64 |
| Dropout | 0.2 |
| Patience | 3 |
| Device | CPU |

O treinamento utilizou early stopping, então o modelo poderia treinar por até 20 épocas, mas parou antes quando a validação deixou de melhorar.

Métricas finais:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.6989 |
| Precision | 0.6809 |
| Recall | 0.7059 |
| F1-score | 0.6932 |
| ROC AUC | 0.7650 |
| Loss | 0.5822 |

O resultado ficou adequado para uma primeira versão do modelo neural.

A acurácia ficou próxima de 70%, e o F1-score mostra que o modelo manteve um equilíbrio razoável entre precision e recall. O ROC AUC de 0.7650 indica que o modelo conseguiu diferenciar interações positivas e negativas melhor do que uma escolha aleatória.

Ainda assim, esse modelo deve ser visto como uma primeira versão. Para um sistema real de recomendação, seria importante incluir métricas específicas de ranking, como Precision@K, Recall@K e NDCG@K.

---

## Testes e qualidade de código

O projeto usa Ruff, pre-commit e Pytest para manter o código mais organizado.

### Rodar Ruff

```bash
poetry run ruff check .
```

### Formatar com Ruff

```bash
poetry run ruff format .
```

### Rodar pre-commit

```bash
poetry run pre-commit run --all-files
```

### Rodar testes

```bash
poetry run pytest
```

Resultado atual:

```text
6 passed
```

Os testes cobrem partes principais do preprocessamento, feature engineering e métricas.

---

## Versionamento com Git e DVC

O Git versiona o código, configurações e documentação.

O DVC controla os dados e artefatos gerados pelo pipeline, como:

- dados brutos;
- dados processados;
- bases de treino e teste;
- modelo treinado;
- métricas do pipeline.

Alguns arquivos e pastas são mantidos fora do Git:

```text
.venv/
__pycache__/
mlruns/
mlflow.db
data/raw/
data/processed/
data/features/
models/*.pt
```

Isso ajuda a manter o repositório mais limpo e evita versionar arquivos pesados ou gerados automaticamente.

---

## Etapas do projeto

- [x] Estrutura inicial do projeto
- [x] Configuração do ambiente
- [x] Criação do `.env.example`
- [x] Criação do `settings.py`
- [x] Criação do `validate_env.py`
- [x] Script de download do dataset
- [x] Documentação inicial do dataset
- [x] Preprocessamento dos dados
- [x] Engenharia de atributos
- [x] Configuração de pre-commit hooks
- [x] Testes automatizados
- [x] Baseline com Scikit-Learn
- [x] Modelo neural com PyTorch
- [x] Avaliação dos modelos
- [x] Tracking com MLflow
- [x] Versionamento com DVC
- [x] Pipeline reproduzível com `dvc repro`
- [x] Execução com Docker
- [x] Documentação principal do projeto
- [ ] Model Registry no MLflow
- [ ] Model Card
- [ ] Vídeo STAR

---

## Padrão de commits

O projeto segue uma convenção simples baseada em Conventional Commits.

Exemplos:

| Tipo | Quando usar | Exemplo |
|---|---|---|
| `feat` | Nova funcionalidade | `feat: add data preprocessing step` |
| `fix` | Correção | `fix: copy dataset files recursively` |
| `docs` | Documentação | `docs: update project readme` |
| `chore` | Configuração ou manutenção | `chore: configure pre-commit hooks` |
| `test` | Testes | `test: add preprocessing tests` |
| `refactor` | Refatoração | `refactor: simplify feature engineering functions` |
| `data` | Versionamento ou organização de dados | `data: track raw dataset with dvc` |

Formato usado:

```bash
tipo: descrição curta
```

Exemplo:

```bash
feat: add baseline model
```

---
## Deploy em nuvem

Como bônus, foi criado um container web com FastAPI e Docker para disponibilizar uma URL pública do projeto.

A API expõe endpoints simples para validação do serviço e consulta das informações do modelo:

- `/`: status geral da aplicação;
- `/health`: health check do container;
- `/model-info`: informações do modelo e métricas finais;
- `/docs`: documentação automática da API.

URL pública:

```text
https://tech-challenge-fase-02-api.onrender.com

----

## Referências

- Documentação oficial do Python.
- Documentação oficial do PyTorch.
- Documentação oficial do Scikit-Learn.
- Documentação oficial do MLflow.
- Documentação oficial do DVC.
- Documentação oficial do Docker.
- MovieLens Dataset.
- Material das disciplinas da FIAP.
