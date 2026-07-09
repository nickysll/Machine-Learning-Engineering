# Tech Challenge - Fase 02 | Machine Learning Engineering

## Sobre o desafio

Este projeto foi desenvolvido como parte da **Fase 02** da Pós-Graduação em **Machine Learning Engineering** da FIAP.

O objetivo desta etapa é consolidar conhecimentos de Engenharia de Machine Learning por meio da construção de uma solução completa, contemplando modelagem, versionamento de dados, rastreamento de experimentos, testes automatizados, containerização e documentação técnica.

---

## Problema proposto

Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento dos usuários.

O sistema deve utilizar uma rede neural, como uma **MLP** ou um modelo baseado em **embeddings**, desenvolvida em **PyTorch**.

Além do modelo neural, o projeto contempla o ciclo completo de desenvolvimento de Machine Learning, incluindo:

- pipeline de treinamento;
- comparação com modelo baseline;
- versionamento de dados e artefatos com DVC;
- rastreamento de experimentos com MLflow;
- containerização com Docker;
- testes automatizados;
- organização modular seguindo princípios de Clean Code.

---

## Objetivos do projeto

- Desenvolver um sistema de recomendação utilizando PyTorch.
- Criar um modelo baseline com Scikit-Learn.
- Comparar o modelo neural com uma abordagem mais simples.
- Aplicar boas práticas de Engenharia de Software.
- Garantir reprodutibilidade do ambiente e dos experimentos.
- Versionar dados, métricas e artefatos com DVC.
- Rastrear experimentos com MLflow.
- Containerizar a execução do pipeline com Docker.
- Documentar decisões técnicas, resultados e limitações.

---

## Dataset

### Dataset escolhido

**MovieLens Latest Small**

### Fonte

Dataset obtido via KaggleHub a partir de uma versão do MovieLens.

### Justificativa

O MovieLens foi escolhido por ser um dataset clássico para problemas de recomendação, contendo interações entre usuários e itens.

Embora o domínio original seja filmes, a estrutura do problema é equivalente a um cenário de e-commerce, em que usuários interagem com produtos e o modelo aprende padrões para recomendar novos itens.

### Arquivos principais

Os arquivos brutos são armazenados localmente em:

```text
data/raw/movielens_100k/
````

Principais arquivos utilizados:

* `ratings.csv`: interações entre usuários e itens;
* `movies.csv`: metadados dos itens;
* `tags.csv`: tags atribuídas por usuários;
* `links.csv`: identificadores externos dos filmes.

### Colunas principais do `ratings.csv`

| Coluna      | Descrição                   |
| ----------- | --------------------------- |
| `userId`    | Identificador do usuário    |
| `movieId`   | Identificador do item       |
| `rating`    | Avaliação dada pelo usuário |
| `timestamp` | Momento da interação        |

### Definição de interação

Uma interação ocorre quando um usuário avalia um item.

### Target

Para transformar o problema em uma tarefa supervisionada inicial, foi criado um target binário:

| Regra         | Target                |
| ------------- | --------------------- |
| `rating >= 4` | 1, interação positiva |
| `rating < 4`  | 0, interação negativa |

---

## Solução implementada

A solução foi estruturada como um pipeline reprodutível de Machine Learning.

O fluxo principal é:

```text
download_data -> preprocess -> feature_eng -> train
```

As etapas executam:

1. download automático do dataset;
2. preprocessamento das interações;
3. criação das bases de treino e teste;
4. treinamento do modelo neural em PyTorch;
5. geração de métricas;
6. salvamento do modelo treinado;
7. rastreamento do experimento com MLflow.

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

| Pasta             | Finalidade                                           |
| ----------------- | ---------------------------------------------------- |
| `data/raw/`       | Dados brutos baixados localmente                     |
| `data/processed/` | Dados tratados pelo preprocessamento                 |
| `data/features/`  | Bases finais para treino e teste                     |
| `src/configs/`    | Configurações do projeto                             |
| `src/data/`       | Código de preprocessamento e engenharia de atributos |
| `src/evaluation/` | Métricas e funções auxiliares de avaliação           |
| `src/models/`     | Código dos modelos, treino e arquitetura neural      |
| `scripts/`        | Scripts auxiliares de ambiente e download            |
| `reports/`        | Documentação auxiliar, métricas e resultados         |
| `models/`         | Modelo treinado e artefatos de modelagem             |
| `tests/`          | Testes automatizados                                 |

Os arquivos gerados dentro de `data/raw/`, `data/processed/`, `data/features/` e `models/` não são versionados diretamente pelo Git. Esses artefatos são controlados pelo DVC, permitindo rastreabilidade e reprodutibilidade sem armazenar arquivos grandes diretamente no repositório Git.

---

## Tecnologias utilizadas

* Python 3.12
* Poetry
* Pandas
* NumPy
* Scikit-Learn
* PyTorch
* MLflow
* DVC
* Docker
* Docker Compose
* Ruff
* Pre-commit
* Pytest
* Pydantic Settings
* KaggleHub

---

## Configuração do ambiente local

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd tech-challenge-fase-02
```

### 2. Instalar dependências

```bash
poetry install
```

### 3. Criar arquivo `.env`

Use o arquivo `.env.example` como referência:

```bash
cp .env.example .env
```

Exemplo de variáveis esperadas:

```env
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DVC_REMOTE=local
```

### 4. Validar o ambiente

```bash
poetry run python scripts/validate_env.py
```

Resultado esperado:

```text
Environment validation: OK
```

---

## Execução local do pipeline

### 1. Download do dataset

```bash
poetry run python scripts/download_data.py
```

Saída esperada:

```text
Dataset downloaded to: data/raw/movielens_100k
```

### 2. Preprocessamento

```bash
poetry run python -m src.data.preprocess
```

Saída esperada:

```text
Processed data saved to: data/processed/interactions.csv
Rows: 100836
```

### 3. Engenharia de atributos

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

### 4. Treinamento do modelo PyTorch

```bash
poetry run python -m src.models.train_torch --run-name local_recommender_net --device cpu
```

---

## Execução com DVC

O pipeline completo foi configurado com DVC.

Para executar todas as etapas de forma reproduzível:

```bash
poetry run dvc repro
```

Para verificar o status do pipeline:

```bash
poetry run dvc status
```

Para visualizar o grafo do pipeline:

```bash
poetry run dvc dag
```

Fluxo do pipeline:

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

## Execução com Docker

O projeto foi containerizado com Docker para permitir execução em ambiente isolado e reprodutível.

### Construir a imagem

```bash
docker compose build
```

### Executar o pipeline dentro do container

```bash
docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db pipeline
```

Esse comando executa o pipeline DVC dentro do container e registra o experimento do MLflow em um backend SQLite local.

Ao final da execução, a saída esperada inclui:

```text
Early stopping triggered.
PyTorch metrics saved to: reports/torch_metrics.json
Model saved to: models/recommender_net.pt
Use `dvc push` to send your updates to remote storage.
```

---

## MLflow

O projeto utiliza MLflow para rastreamento de experimentos.

Durante o treinamento, são registrados:

* nome da execução;
* parâmetros do modelo;
* métricas de avaliação;
* artefatos gerados.

Principais parâmetros registrados:

| Parâmetro       | Valor |
| --------------- | ----: |
| `batch_size`    |   512 |
| `epochs`        |    20 |
| `learning_rate` | 0.001 |
| `patience`      |     3 |
| `embedding_dim` |    32 |
| `hidden_dim`    |    64 |
| `dropout_rate`  |   0.2 |
| `device`        |   CPU |

As métricas finais são salvas em:

```text
reports/torch_metrics.json
```

O modelo treinado é salvo em:

```text
models/recommender_net.pt
```

O registro formal no MLflow Model Registry é considerado uma evolução futura do projeto.

---

## Modelos implementados

### Baseline com Scikit-Learn

Foi implementado um modelo baseline utilizando Scikit-Learn para servir como referência comparativa inicial.

As métricas do baseline são armazenadas em:

```text
reports/baseline_metrics.json
```

### Recommender Net com PyTorch

O modelo principal é uma rede neural de recomendação baseada em embeddings.

A arquitetura utiliza representações vetoriais para usuários e itens, combinando essas informações em camadas densas para prever a probabilidade de uma interação positiva.

O modelo foi treinado com early stopping para evitar overfitting.

---

## Resultados do modelo

O modelo final implementado foi uma rede neural de recomendação baseada em embeddings, utilizando PyTorch.

O treinamento foi executado com os seguintes principais parâmetros:

* `batch_size`: 512
* `epochs`: 20
* `learning_rate`: 0.001
* `embedding_dim`: 32
* `hidden_dim`: 64
* `dropout_rate`: 0.2
* `patience`: 3
* `device`: CPU

Durante o treinamento, foi utilizado early stopping para evitar overfitting. O processo foi interrompido automaticamente após estabilização da perda de validação.

As métricas finais obtidas foram:

| Métrica   |  Valor |
| --------- | -----: |
| Accuracy  | 0.6989 |
| Precision | 0.6809 |
| Recall    | 0.7059 |
| F1-score  | 0.6932 |
| ROC AUC   | 0.7650 |
| Loss      | 0.5822 |

Os resultados indicam que o modelo conseguiu aprender padrões relevantes nas interações entre usuários e itens. O ROC AUC de 0.7650 demonstra capacidade satisfatória de diferenciação entre interações positivas e negativas, superando um comportamento aleatório. Além disso, o equilíbrio entre precision, recall e F1-score mostra que o modelo apresenta desempenho consistente tanto na identificação de recomendações relevantes quanto na recuperação de casos positivos.

---

## Qualidade de código

Este projeto utiliza Ruff, pre-commit e Pytest para manter qualidade, padronização e segurança nas alterações.

### Rodar Ruff

```bash
poetry run ruff check .
```

### Rodar formatação com Ruff

```bash
poetry run ruff format .
```

### Rodar pre-commit em todos os arquivos

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

A suíte de testes automatizados cobre validações essenciais de preprocessamento, engenharia de atributos e métricas.

---

## Versionamento com Git e DVC

O Git é utilizado para versionar código, configurações e documentação.

O DVC é utilizado para controlar dados e artefatos gerados pelo pipeline, como:

* dados brutos;
* dados processados;
* bases de treino e teste;
* modelo treinado;
* métricas do pipeline.

Arquivos grandes e artefatos locais não são armazenados diretamente no Git.

Exemplos de arquivos e pastas ignorados:

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

---

## Etapas de desenvolvimento

* [x] Estrutura inicial do projeto
* [x] Configuração do ambiente
* [x] Criação do `.env.example`
* [x] Criação do `settings.py`
* [x] Criação do `validate_env.py`
* [x] Script de download do dataset
* [x] Documentação inicial do dataset
* [x] Desenvolvimento do preprocessamento
* [x] Engenharia de atributos
* [x] Configuração de pre-commit hooks
* [x] Testes automatizados
* [x] Implementação do baseline com Scikit-Learn
* [x] Implementação do modelo neural com PyTorch
* [x] Avaliação dos modelos
* [x] Tracking de experimentos com MLflow
* [x] Versionamento dos dados com DVC
* [x] Pipeline reproduzível com `dvc repro`
* [x] Containerização com Docker
* [x] Documentação principal do projeto
* [ ] Model Registry no MLflow
* [ ] Model Card
* [ ] Vídeo STAR

---

## Padrão de commits

Este projeto utiliza uma convenção de commits semânticos baseada no padrão Conventional Commits.

### Tipos utilizados

| Tipo       | Quando usar                                       | Exemplo                                            |
| ---------- | ------------------------------------------------- | -------------------------------------------------- |
| `feat`     | Nova funcionalidade                               | `feat: add data preprocessing step`                |
| `fix`      | Correção de bug ou comportamento incorreto        | `fix: copy dataset files recursively`              |
| `docs`     | Alterações em documentação                        | `docs: update project documentation`               |
| `chore`    | Tarefas de manutenção ou configuração             | `chore: configure pre-commit hooks`                |
| `test`     | Criação ou alteração de testes                    | `test: add preprocessing tests`                    |
| `style`    | Formatação de código sem alterar comportamento    | `style: format data pipeline files`                |
| `refactor` | Refatoração sem mudar resultado final             | `refactor: simplify feature engineering functions` |
| `data`     | Alterações relacionadas ao versionamento de dados | `data: track raw dataset with dvc`                 |

A mensagem deve seguir o formato:

```bash
tipo: descrição curta da alteração
```

Exemplo:

```bash
feat: add baseline model
```

---

## Limitações

Apesar dos resultados positivos, o projeto possui algumas limitações:

* o dataset é baseado em avaliações explícitas, não em navegação real de e-commerce;
* o target binário foi criado a partir da nota do usuário;
* o modelo não considera contexto de sessão, preço, estoque, categoria real de produto ou sazonalidade;
* as métricas atuais avaliam a classificação da interação, mas ainda não incluem métricas específicas de ranking;
* o modelo foi treinado em CPU para manter simplicidade e reprodutibilidade local.

---

## Próximos passos

Como evolução futura, o projeto pode ser aprimorado com:

* criação de um Model Card detalhado;
* registro formal do modelo no MLflow Model Registry;
* uso de métricas específicas de recomendação, como Precision@K, Recall@K e NDCG@K;
* comparação com outros algoritmos de recomendação;
* ajuste de hiperparâmetros;
* inclusão de novas features;
* criação de uma API para servir recomendações;
* criação de pipeline de CI/CD com GitHub Actions.

---

## Referências

* Documentação oficial do Python.
* Documentação oficial do PyTorch.
* Documentação oficial do Scikit-Learn.
* Documentação oficial do MLflow.
* Documentação oficial do DVC.
* Documentação oficial do Docker.
* MovieLens Dataset.
* Material das disciplinas da FIAP.
