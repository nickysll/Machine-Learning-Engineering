# Tech Challenge — Fase 03

## Machine Learning Engineering — Classificação de Textos Médicos em Produção

Projeto desenvolvido para a **Fase 03 da Pós-Tech em Machine Learning Engineering**, com foco no ciclo de vida de um modelo de Machine Learning em ambiente de produção.

A solução implementa um pipeline de **classificação de textos médicos**, contemplando treinamento, API de inferência, testes automatizados, containerização, CI, orquestração, monitoramento e análise de performance.

---

## Visão geral

O projeto utiliza textos de abstracts médicos como entrada e realiza uma classificação multiclasse utilizando um pipeline composto por:

```text
Texto médico
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Classe prevista
```

Além do modelo, foi construída uma arquitetura de MLOps contendo:

```text
                         ┌─────────────────────┐
                         │   Medical Abstracts │
                         │      Dataset        │
                         └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │      Airflow        │
                         │ Training Pipeline   │
                         └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ TF-IDF + Logistic   │
                         │     Regression      │
                         └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │    model.joblib     │
                         └─────────┬───────────┘
                                   │
                                   ▼
┌─────────────────┐      ┌─────────────────────┐
│ GitHub Actions  │─────▶│      FastAPI        │
│ Ruff / Pytest / │      │ /health /predict   │
│ Docker Build    │      │      /metrics       │
└─────────────────┘      └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │     Prometheus      │
                         └─────────┬───────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │       Grafana       │
                         │ Requests / P95 /    │
                         │       Errors        │
                         └─────────────────────┘
```

Também foi avaliada uma versão do classificador em **ONNX Runtime**, permitindo comparar latência e throughput com o pipeline original em scikit-learn.

---

# Dataset

Foi utilizado o dataset:

**Medical Abstracts — `TimSchopf/medical_abstracts`**

O download é realizado automaticamente através da biblioteca Hugging Face `datasets`, portanto os arquivos brutos não precisam ser armazenados no repositório.

Estrutura utilizada:

```text
condition_label
medical_abstract
```

Quantidade de registros:

| Split | Registros |
|---|---:|
| Treino | 11.550 |
| Teste | 2.888 |
| Total | 14.438 |

O problema possui **5 classes**.

Distribuição observada no conjunto de treinamento:

| Classe | Registros |
|---:|---:|
| 1 | 2.530 |
| 2 | 1.195 |
| 3 | 1.540 |
| 4 | 2.441 |
| 5 | 3.844 |

Não foram encontrados valores ausentes nas colunas utilizadas.

O download pode ser reproduzido com:

```bash
poetry run python scripts/download_data.py
```

Os arquivos são gerados localmente em:

```text
data/raw/train.parquet
data/raw/test.parquet
```

---

# Modelo de Machine Learning

O pipeline utiliza:

- `TfidfVectorizer`
- `LogisticRegression`
- `class_weight="balanced"`

Configuração principal do TF-IDF:

```text
lowercase=True
stop_words="english"
ngram_range=(1, 2)
min_df=2
max_df=0.95
sublinear_tf=True
```

O uso de `class_weight="balanced"` foi escolhido após comparação com uma versão sem balanceamento.

A versão balanceada apresentou melhor desempenho global, principalmente nas classes menos representadas.

---

## Resultados do modelo

Resultados no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Accuracy | **58,73%** |
| Macro F1 | **0,5884** |
| Weighted F1 | **0,5734** |

Resultados por classe:

| Classe | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 1 | 0,675 | 0,750 | 0,711 |
| 2 | 0,464 | 0,662 | 0,545 |
| 3 | 0,520 | 0,660 | 0,582 |
| 4 | 0,645 | 0,741 | 0,690 |
| 5 | 0,558 | 0,330 | 0,415 |

O modelo apresenta melhor desempenho nas classes 1 e 4 e maior dificuldade na classe 5.

As métricas completas são armazenadas em:

```text
models/metrics.json
```

---

# Treinamento

Para treinar o modelo:

```bash
poetry run python src/training/train.py
```

O pipeline completo é salvo como:

```text
models/model.joblib
```

O arquivo contém tanto o TF-IDF quanto o classificador, garantindo que o mesmo pré-processamento utilizado no treinamento seja aplicado durante a inferência.

---

# API de inferência

A aplicação utiliza **FastAPI**.

Endpoints disponíveis:

| Método | Endpoint | Função |
|---|---|---|
| GET | `/health` | Verificação de saúde da API |
| POST | `/predict` | Realiza uma classificação |
| GET | `/metrics` | Expõe métricas para o Prometheus |

Executar localmente:

```bash
poetry run uvicorn app.main:app --reload
```

Documentação Swagger:

```text
http://localhost:8000/docs
```

---

## Exemplo de requisição

```json
{
  "text": "Patient presents with persistent chest pain and cardiovascular symptoms requiring medical evaluation."
}
```

Exemplo de resposta:

```json
{
  "prediction": 4,
  "confidence": 0.73
}
```

A propriedade `confidence` representa a maior probabilidade estimada pelo classificador entre as cinco classes e não deve ser interpretada como confiança clínica.

---

# Testes automatizados

A suíte utiliza **Pytest** e contempla:

- disponibilidade da API;
- endpoint `/health`;
- inferência em `/predict`;
- validação de payload inválido;
- existência e carregamento do modelo;
- formato das probabilidades;
- existência dos datasets;
- schema dos dados;
- ausência de valores nulos;
- validação das classes.

Executar:

```bash
poetry run pytest -v
```

Resultado atual:

```text
11 passed
```

A qualidade do código também é validada com Ruff:

```bash
poetry run ruff check .
```

---

# Docker

A API possui uma imagem Docker própria.

Build:

```bash
docker build -t medical-text-api:latest .
```

Execução:

```bash
docker run --rm -p 8000:8000 medical-text-api:latest
```

Teste:

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

---

# Docker Compose

O ambiente completo pode ser iniciado com:

```bash
docker compose up --build
```

Serviços disponíveis:

| Serviço | Porta |
|---|---:|
| FastAPI | `8000` |
| Grafana | `3000` |
| Airflow | `8080` |
| Prometheus | `9090` |

Para finalizar:

```bash
docker compose down
```

---

# Integração Contínua — GitHub Actions

O projeto possui pipeline de CI executado automaticamente em pushes e pull requests relacionados à Fase 03.

Workflow:

```text
Checkout
   ↓
Python 3.12
   ↓
Poetry
   ↓
Install dependencies
   ↓
Download dataset
   ↓
Train model
   ↓
Ruff
   ↓
Pytest
   ↓
Docker build
```

Dessa forma, o ambiente do GitHub consegue reconstruir todo o fluxo sem depender dos dados ou do modelo existentes na máquina local.

Arquivo:

```text
.github/workflows/phase3-ci.yml
```

---

# Orquestração com Apache Airflow

O pipeline de treinamento também foi implementado como uma DAG do Airflow.

DAG:

```text
medical_text_training_pipeline
```

Fluxo:

```text
download_data
      ↓
validate_data
      ↓
train_model
      ↓
evaluate_model
      ↓
save_model
```

Função das tarefas:

| Task | Responsabilidade |
|---|---|
| `download_data` | Recupera o dataset |
| `validate_data` | Valida schema, classes e valores ausentes |
| `train_model` | Treina TF-IDF + Logistic Regression |
| `evaluate_model` | Calcula e salva métricas |
| `save_model` | Promove uma cópia do modelo como artefato de produção |

Interface:

```text
http://localhost:8080
```

## Execução do pipeline

![Pipeline Airflow](docs/images/airflow_fluxo.png)

---

# Monitoramento

A API possui instrumentação com `prometheus-client`.

São monitoradas duas métricas customizadas principais:

```text
api_requests_total
```

Quantidade total de requisições, segmentada por:

- método;
- endpoint;
- status HTTP.

E:

```text
api_request_latency_seconds
```

Histograma de latência das requisições.

---

## Prometheus

O Prometheus coleta automaticamente:

```text
http://api:8000/metrics
```

Intervalo de coleta:

```text
5 segundos
```

Interface local:

```text
http://localhost:9090
```

---

# Grafana

O Grafana utiliza o Prometheus como datasource e possui dashboard provisionado automaticamente pelo Docker Compose.

O dashboard inclui:

### Requisições por endpoint

Permite visualizar o volume de utilização da API.

### Latência P95

Exibe o percentil 95 de latência por endpoint.

### Taxa de erros HTTP

Monitora requisições com status `5xx`.

Exemplo observado durante os testes locais:

```text
/health   P95 ≈ 4,91 ms
/predict  P95 ≈ 17,50 ms
Taxa de erros = 0%
```

Os valores representam uma execução local específica e podem variar conforme hardware, carga e ambiente.

## Dashboard

![Dashboard Grafana](docs/images/grafana-monitoring-dashboard.png)

Interface:

```text
http://localhost:3000
```

---

# Otimização com ONNX

Também foi analisada a conversão do modelo para **ONNX Runtime**.

Inicialmente foi avaliada a conversão do pipeline completo contendo:

```text
TF-IDF + Logistic Regression
```

Entretanto, foram observadas divergências relacionadas ao processamento textual do `TfidfVectorizer` convertido.

Para garantir equivalência entre as previsões, a estratégia final manteve o **mesmo TF-IDF do scikit-learn** e converteu somente a `LogisticRegression` para ONNX.

A validação realizada em 100 amostras apresentou:

```text
Matching predictions: 100
Agreement: 100%
```

Assim, a comparação de performance foi feita entre classificadores equivalentes.

---

# Benchmark de latência

Foram realizadas 20 chamadas de warm-up e 300 inferências utilizadas nas métricas finais.

## Benchmark end-to-end

Nesse cenário foi considerado todo o processo necessário para realizar uma previsão.

| Métrica | Scikit-learn | ONNX Runtime |
|---|---:|---:|
| Média | **2,100 ms** | 2,533 ms |
| P50 | **1,967 ms** | 2,261 ms |
| P95 | **3,041 ms** | 3,646 ms |
| P99 | **3,685 ms** | 6,229 ms |
| Throughput | **476,10 req/s** | 394,83 req/s |

Resultado:

```text
ONNX apresentou aproximadamente 20,6% mais latência média
e aproximadamente 17,1% menos throughput.
```

---

## Benchmark do classificador isolado

Também foi realizada uma comparação após pré-calcular as features TF-IDF, isolando a etapa do classificador.

Resultados da execução versionada:

| Métrica | Scikit-learn | ONNX Runtime |
|---|---:|---:|
| Média | **0,235 ms** | 0,369 ms |
| P50 | **0,212 ms** | 0,335 ms |
| P95 | **0,345 ms** | 0,571 ms |
| P99 | **0,455 ms** | 0,855 ms |
| Throughput | **4.254 req/s** | 2.709 req/s |

Nesta execução, o scikit-learn também apresentou melhor desempenho.

Como as latências do classificador isolado estão abaixo de 1 ms, pequenas variações de ambiente e agendamento da CPU podem gerar diferenças relevantes entre execuções. Por isso, os resultados versionados são utilizados como referência do experimento.

---

# Conclusão da otimização

A adoção de ONNX **não apresentou ganho de latência para este pipeline específico**.

Esse resultado é tecnicamente relevante: uma ferramenta de otimização não deve ser considerada superior apenas por sua adoção.

Neste projeto, fatores como:

- modelo extremamente leve;
- implementação já eficiente da Logistic Regression no scikit-learn;
- custo de transformação dos dados;
- conversão de matriz esparsa para representação densa;
- overhead de chamada ao ONNX Runtime;

fazem com que o ganho esperado não apareça no cenário avaliado.

Dessa forma, a versão scikit-learn foi mantida como alternativa mais eficiente para a inferência completa neste ambiente.

Os resultados completos estão disponíveis em:

```text
models/benchmark.json
models/benchmark_classifier.json
```

---

# Arquitetura proposta para produção em nuvem

A aplicação foi construída localmente com containers, mas uma arquitetura de produção poderia utilizar serviços gerenciados em nuvem.

Uma possibilidade seria:

```text
Cliente
   ↓
HTTPS
   ↓
Cloud Run
FastAPI container
   ↓
Modelo de ML
   │
   ├──── métricas / logs
   ↓
Observabilidade

Artifact Registry
   ↑
Docker Image
   ↑
GitHub Actions
```

Para o treinamento:

```text
Dataset
   ↓
Pipeline Airflow
   ↓
Validação
   ↓
Treinamento
   ↓
Avaliação
   ↓
Novo modelo
```

O modelo de atendimento **online/real-time** foi escolhido porque a aplicação expõe uma API REST para classificação sob demanda.

O uso de uma plataforma serverless para a API permite:

- escalabilidade automática;
- redução de infraestrutura operacional;
- execução baseada em containers;
- facilidade de versionamento;
- possibilidade de rollback;
- cobrança associada ao uso.

Em um ambiente corporativo, Airflow poderia ser executado em uma solução gerenciada de orquestração e os artefatos do modelo armazenados em um serviço de objetos/model registry.

> A arquitetura cloud apresentada é uma proposta de produção. A implementação deste repositório utiliza Docker Compose localmente.

---

# Estrutura do projeto

```text
tech-challenge-fase-03/
│
├── airflow/
│   └── dags/
│       └── retrain_model.py
│
├── app/
│   ├── main.py
│   ├── metrics.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── images/
│
├── models/
│   ├── metrics.json
│   ├── benchmark.json
│   └── benchmark_classifier.json
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│
├── scripts/
│   ├── download_data.py
│   └── validate_data.py
│
├── src/
│   ├── optimization/
│   │   ├── benchmark.py
│   │   ├── benchmark_classifier.py
│   │   ├── convert_onnx.py
│   │   └── validate_onnx.py
│   │
│   └── training/
│       ├── train.py
│       └── evaluate.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
│
├── Dockerfile
├── Dockerfile.airflow
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
├── requirements-airflow.txt
└── README.md
```

---

# Como executar o projeto

## 1. Clonar o repositório

```bash
git clone git@github.com:nickysll/Machine-Learning-Engineering.git
cd Machine-Learning-Engineering/tech-challenge-fase-03
```

## 2. Instalar as dependências

Requisitos:

```text
Python 3.12
Poetry
Docker
Docker Compose
```

Instalação:

```bash
poetry install
```

## 3. Baixar os dados

```bash
poetry run python scripts/download_data.py
```

## 4. Validar os dados

```bash
poetry run python scripts/validate_data.py
```

## 5. Treinar

```bash
poetry run python src/training/train.py
```

## 6. Avaliar

```bash
poetry run python src/training/evaluate.py
```

## 7. Executar os testes

```bash
poetry run pytest -v
```

## 8. Executar análise estática

```bash
poetry run ruff check .
```

## 9. Subir a solução

Em Linux/WSL:

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up --build
```

Depois:

```text
FastAPI:    http://localhost:8000
Swagger:    http://localhost:8000/docs
Airflow:    http://localhost:8080
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
```

Para encerrar:

```bash
docker compose down
```

---

# Tecnologias

- Python 3.12
- Poetry
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- FastAPI
- Pydantic
- Pytest
- Ruff
- Docker
- Docker Compose
- GitHub Actions
- Apache Airflow
- Prometheus
- Grafana
- ONNX
- ONNX Runtime
- skl2onnx
- Hugging Face Datasets

---

# Principais resultados

O projeto demonstrou um fluxo completo de Machine Learning Engineering:

```text
dados
  ↓
validação
  ↓
treinamento
  ↓
avaliação
  ↓
API
  ↓
testes
  ↓
container
  ↓
CI
  ↓
orquestração
  ↓
monitoramento
  ↓
otimização
  ↓
benchmark
```

O principal aprendizado da análise de performance foi que **otimização deve ser medida e não presumida**. Apesar da adoção de ONNX Runtime, o pipeline original em scikit-learn apresentou menor latência no ambiente avaliado.

---

## Status

✅ Implementação técnica concluída  
✅ Modelo e avaliação  
✅ API de inferência  
✅ Testes automatizados  
✅ Containerização  
✅ CI com GitHub Actions  
✅ Orquestração com Airflow  
✅ Monitoramento com Prometheus e Grafana  
✅ Conversão e avaliação ONNX  
✅ Benchmark de latência  

---

## Autora

**Nicoly da Silva Moreira**

Tech Challenge — Fase 03  
Pós-Tech — Machine Learning Engineering