# Tech Challenge - Fase 01

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como parte da **Fase 01** da Pós-Graduação em **Machine Learning Engineering** da FIAP.

O objetivo deste Tech Challenge é aplicar, de forma integrada, os conhecimentos adquiridos ao longo da fase para desenvolver uma solução completa de Machine Learning, contemplando desde a análise exploratória dos dados até a disponibilização do modelo por meio de uma API.

O projeto segue as boas práticas de Engenharia de Machine Learning, priorizando organização, reprodutibilidade, documentação e qualidade de código.

---

# 🎯 Problema Proposto

Uma operadora de telecomunicações enfrenta um alto índice de cancelamento de clientes (**Customer Churn**), impactando diretamente sua receita e crescimento.

O desafio consiste em desenvolver um modelo preditivo capaz de identificar clientes com maior probabilidade de cancelar seus serviços, permitindo que a empresa realize ações preventivas de retenção.

A solução deverá ser construída utilizando uma **Rede Neural Multilayer Perceptron (MLP)** desenvolvida em **PyTorch**, comparando seu desempenho com modelos tradicionais de Machine Learning.

Além da modelagem, todo o pipeline deverá seguir práticas profissionais de Engenharia de Machine Learning, incluindo rastreamento de experimentos, testes automatizados, documentação e disponibilização do modelo via API.

---

# 🎯 Objetivos

Ao final do projeto, espera-se entregar:

* Modelo de classificação de churn utilizando PyTorch;
* Comparação com modelos baseline utilizando Scikit-Learn;
* Rastreamento de experimentos com MLflow;
* API REST desenvolvida com FastAPI;
* Estrutura organizada seguindo princípios de Clean Code;
* Testes automatizados;
* Model Card documentando limitações e desempenho do modelo;
* Projeto totalmente reproduzível.

---

# 📋 Requisitos do Tech Challenge

O projeto deve contemplar obrigatoriamente:

* Estrutura organizada do repositório;
* README completo;
* Gerenciamento de dependências com `pyproject.toml`;
* Histórico de commits semântico;
* Rede Neural desenvolvida em PyTorch;
* Baselines utilizando Scikit-Learn;
* Tracking de experimentos com MLflow;
* API de inferência utilizando FastAPI;
* Testes automatizados;
* Logging estruturado;
* Model Card;
* Linting utilizando Ruff;
* Reprodutibilidade do ambiente.

---

# 🛠️ Tecnologias Utilizadas

* Python 3.13
* PyTorch
* Scikit-Learn
* Pandas
* NumPy
* MLflow
* FastAPI
* Pydantic
* Pytest
* Ruff
* UV
* Git & GitHub

---

# 📂 Estrutura do Projeto

```text
.
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── README.md
└── pyproject.toml
```

---

# 🚀 Etapas de Desenvolvimento

## ✅ Etapa 1 — Entendimento do Problema

* Definição do problema de negócio;
* Exploração dos dados (EDA);
* Construção do ML Canvas;
* Desenvolvimento dos modelos baseline;
* Registro dos experimentos no MLflow.

---

## 🔄 Etapa 2 — Modelagem

* Construção da Rede Neural MLP em PyTorch;
* Processo de treinamento;
* Early Stopping;
* Avaliação utilizando múltiplas métricas;
* Comparação com modelos tradicionais.

---

## 🔄 Etapa 3 — Engenharia de Software

* Refatoração do código;
* Organização em módulos;
* Desenvolvimento da API com FastAPI;
* Testes automatizados;
* Logging estruturado;
* Configuração do ambiente de desenvolvimento.

---

## 🔄 Etapa 4 — Documentação e Entrega

* Elaboração do Model Card;
* Documentação da arquitetura;
* Plano de monitoramento;
* Finalização do README;
* Gravação do vídeo utilizando o método STAR.

---

# 📊 Dataset

O projeto utilizará um conjunto de dados de telecomunicações para classificação binária de churn.

Dataset sugerido pela FIAP:

* IBM Telco Customer Churn

Também poderão ser utilizados outros datasets públicos que atendam aos requisitos mínimos propostos pelo desafio.

---

# 📌 Status do Projeto

🚧 Em desenvolvimento.

---

# 📚 Aprendizados

Durante este projeto serão aplicados conceitos de:

* Ciclo de Vida de Machine Learning;
* Engenharia de Software;
* Desenvolvimento de APIs;
* Modelagem com Redes Neurais;
* Rastreamento de Experimentos;
* Boas práticas de documentação;
* Reprodutibilidade de projetos de ML.

---

# 👨‍💻 Autor

Projeto desenvolvido como parte da Pós-Graduação em **Machine Learning Engineering** da **FIAP**.
