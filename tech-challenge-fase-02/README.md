# Tech Challenge - Fase 02 | Machine Learning Engineering

## Sobre o desafio

Este projeto foi desenvolvido como parte da **Fase 02** da Pós-Graduação em **Machine Learning Engineering** da FIAP.

O objetivo desta etapa é consolidar os conhecimentos adquiridos ao longo das disciplinas da fase, desenvolvendo uma solução completa de Engenharia de Machine Learning com foco em:

---

## Problema proposto

Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento de navegação dos usuários.

O sistema deve utilizar uma rede neural, como uma **MLP** ou um modelo baseado em **embeddings**, desenvolvida em **PyTorch**.

Além do modelo neural, o projeto deve contemplar o ciclo completo de desenvolvimento de Machine Learning, incluindo:

- pipeline de treinamento;
- comparação com modelos baseline;
- versionamento de dados com DVC;
- rastreamento de experimentos com MLflow;
- containerização com Docker;
- organização seguindo princípios de Clean Code.

---

## Objetivos do projeto

- Desenvolver um sistema de recomendação utilizando PyTorch.
- Comparar o modelo neural com baselines em Scikit-Learn.
- Aplicar boas práticas de Engenharia de Software.
- Garantir reprodutibilidade do ambiente e dos experimentos.
- Versionar dados e pipeline com DVC.
- Rastrear experimentos com MLflow.
- Containerizar a aplicação com Docker.
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
```

Principais arquivos utilizados:

- `ratings.csv`: interações entre usuários e itens;
- `movies.csv`: metadados dos itens;
- `tags.csv`: tags atribuídas por usuários;
- `links.csv`: identificadores externos dos filmes.

### Colunas principais do `ratings.csv`

| Coluna | Descrição |
|---|---|
| `userId` | Identificador do usuário |
| `movieId` | Identificador do item |
| `rating` | Avaliação dada pelo usuário |
| `timestamp` | Momento da interação |

### Definição de interação

Uma interação ocorre quando um usuário avalia um item.

### Target inicial

Para transformar o problema em uma tarefa supervisionada inicial, foi criado um target binário:

| Regra | Target |
|---|---|
| `rating >= 4` | 1, interação positiva |
| `rating < 4` | 0, interação negativa |

---

## Estrutura do projeto

```text
.
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── data
│   ├── external
│   ├── features
│   ├── processed
│   └── raw
├── main.py
├── models
├── notebooks
├── poetry.lock
├── pyproject.toml
├── reports
│   ├── dataset_notes.md
│   └── figures
├── scripts
│   ├── download_data.py
│   └── validate_env.py
├── src
│   ├── configs
│   │   └── settings.py
│   └── data
│       ├── feature_engineering.py
│       └── preprocess.py
└── tests
```

---

## Organização das pastas

| Pasta | Finalidade |
|---|---|
| `data/raw/` | Dados brutos baixados localmente |
| `data/processed/` | Dados tratados pelo preprocessamento |
| `data/features/` | Bases finais para treino e teste |
| `src/configs/` | Configurações do projeto |
| `src/data/` | Código de preprocessamento e engenharia de atributos |
| `scripts/` | Scripts auxiliares de ambiente e download |
| `reports/` | Documentação auxiliar e resultados |
| `models/` | Modelos treinados e artefatos futuros |
| `tests/` | Testes automatizados |

Os arquivos gerados dentro de `data/raw/`, `data/processed/` e `data/features/` não são versionados diretamente pelo Git. Eles serão controlados posteriormente com DVC.

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
- Ruff
- Pre-commit
- Pytest
- Pydantic Settings
- KaggleHub

---

## Configuração do ambiente

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
MLFLOW_TRACKING_URI=http://localhost:5000
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

## Pipeline atual de dados

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
poetry run python src/data/preprocess.py
```

Saída esperada:

```text
Processed data saved to: data/processed/interactions.csv
Rows: 100836
```

### 3. Engenharia de atributos

```bash
poetry run python src/data/feature_engineering.py
```

Saída esperada:

```text
Train data saved to: data/features/train.csv
Test data saved to: data/features/test.csv
Train rows: 80668
Test rows: 20168
```

---

## Etapas de desenvolvimento

- [x] Estrutura inicial do projeto
- [x] Configuração do ambiente
- [x] Criação do `.env.example`
- [x] Criação do `settings.py`
- [x] Criação do `validate_env.py`
- [x] Script de download do dataset
- [x] Documentação inicial do dataset
- [x] Desenvolvimento do preprocessamento
- [x] Engenharia de atributos
- [x] Configuração de pre-commit hooks
- [x] Testes automatizados
- [x] Implementação do baseline com Scikit-Learn
- [x] Implementação do modelo neural com PyTorch
- [x] Avaliação dos modelos
- [x] Tracking de experimentos com MLflow
- [x] Versionamento dos dados com DVC
- [x] Pipeline reproduzível com `dvc repro`
- [ ] Containerização com Docker
- [ ] Model Registry no MLflow
- [ ] Model Card
- [ ] Documentação final
- [ ] Vídeo STAR

---

## Qualidade de código

Este projeto utiliza Ruff e pre-commit para manter o código padronizado.

### Rodar Ruff

```bash
poetry run ruff check .
```

### Rodar pre-commit em todos os arquivos

```bash
poetry run pre-commit run --all-files
```

### Rodar testes

```bash
poetry run pytest
```

No momento, os testes automatizados ainda serão adicionados.

---

## Padrão de commits

Este projeto utiliza uma convenção de commits semânticos baseada no padrão Conventional Commits. A ideia é deixar o histórico do Git mais organizado e facilitar o entendimento da evolução do projeto.

### Tipos utilizados

| Tipo | Quando usar | Exemplo |
|---|---|---|
| `feat` | Nova funcionalidade | `feat: add data preprocessing step` |
| `fix` | Correção de bug ou comportamento incorreto | `fix: copy dataset files recursively` |
| `docs` | Alterações em documentação | `docs: add dataset selection notes` |
| `chore` | Tarefas de manutenção, configuração ou ajustes que não mudam regra de negócio | `chore: configure pre-commit hooks` |
| `test` | Criação ou alteração de testes | `test: add preprocessing tests` |
| `style` | Formatação de código sem alterar comportamento | `style: format data pipeline files` |
| `refactor` | Refatoração sem mudar o resultado final | `refactor: simplify feature engineering functions` |
| `data` | Alterações relacionadas ao versionamento ou organização de dados | `data: track raw dataset with dvc` |

### Exemplos no projeto

- `feat: add data preprocessing step`
- `feat: add feature engineering step`
- `docs: add dataset selection notes`
- `fix: copy dataset files recursively`
- `chore: configure pre-commit hooks`

### Regra geral

A mensagem deve seguir o formato:

```bash
tipo: descrição curta da alteração
```

Exemplo:

```bash
feat: add baseline model
```

---
## Resultados do modelo

O modelo final implementado foi uma rede neural de recomendação baseada em embeddings, utilizando PyTorch. A arquitetura considera representações vetoriais para usuários e itens, combinando essas informações em camadas densas para prever a probabilidade de uma interação positiva.

O treinamento foi executado com os seguintes principais parâmetros:

* `batch_size`: 512
* `epochs`: 20
* `learning_rate`: 0.001
* `embedding_dim`: 32
* `hidden_dim`: 64
* `dropout_rate`: 0.2
* `patience`: 3
* `device`: CPU

Durante o treinamento, foi utilizado early stopping para evitar overfitting. O processo foi interrompido automaticamente na época 14, após estabilização da perda de validação.

As métricas finais obtidas foram:

| Métrica   |  Valor |
| --------- | -----: |
| Accuracy  | 0.6989 |
| Precision | 0.6809 |
| Recall    | 0.7059 |
| F1-score  | 0.6932 |
| ROC AUC   | 0.7650 |
| Loss      | 0.5822 |

Os resultados indicam que o modelo conseguiu aprender padrões relevantes nas interações entre usuários e itens. O ROC AUC de 0.7650 demonstra uma capacidade satisfatória de diferenciação entre interações positivas e negativas, superando um comportamento aleatório. Além disso, o equilíbrio entre precision, recall e F1-score mostra que o modelo apresenta desempenho consistente tanto na identificação de recomendações relevantes quanto na recuperação de casos positivos.

Como próximos passos, o modelo pode ser aprimorado com novas features, ajuste de hiperparâmetros, validação com métricas específicas de recomendação, como Precision@K, Recall@K e NDCG@K, além da comparação com modelos baseline mais simples.

-----

## Referências

- Documentação oficial das bibliotecas utilizadas.
- Material das disciplinas da FIAP.
- MovieLens Dataset.
