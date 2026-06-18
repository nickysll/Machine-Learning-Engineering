# Tech Challenge - Fase 02 | Big Data Architecture

## Sobre o desafio

Este projeto foi desenvolvido como parte da **Fase 02** da Pós-Graduação em **Machine Learning Engineering** da FIAP.

O objetivo desta etapa é consolidar os conhecimentos adquiridos ao longo das disciplinas da fase, desenvolvendo uma solução completa de Engenharia de Machine Learning utilizando boas práticas de desenvolvimento, versionamento, reprodutibilidade e deploy.

---

## Problema proposto

Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento de navegação dos usuários.

O sistema deve utilizar uma rede neural (MLP ou modelo baseado em embeddings) desenvolvida em **PyTorch**, contemplando todo o ciclo de desenvolvimento de Machine Learning, incluindo:

* Pipeline de treinamento containerizado com Docker;
* Versionamento de dados utilizando DVC;
* Rastreamento de experimentos com MLflow;
* Estrutura organizada seguindo princípios de Clean Code.

---

## Objetivos do projeto

* Desenvolver um sistema de recomendação utilizando PyTorch.
* Aplicar boas práticas de Engenharia de Software.
* Garantir reprodutibilidade dos experimentos.
* Containerizar toda a aplicação.
* Versionar dados e modelos.
* Documentar o projeto adequadamente.

---

## Requisitos obrigatórios

* Estrutura baseada em Clean Code.
* Versionamento utilizando Git.
* Gerenciamento de dependências com uv.
* Docker multi-stage.
* Pipeline utilizando DVC.
* Tracking de experimentos com MLflow.
* Comparação entre modelo neural e modelos baseline.
* README completo.
* Vídeo utilizando metodologia STAR.

---

## Estrutura do projeto

```text
.
├── data
│   ├── raw
│   ├── processed
│   └── external
├── models
├── notebooks
├── reports
├── src
└── tests
```

---

## Etapas de desenvolvimento

* [ ] Estrutura inicial do projeto
* [ ] Configuração do ambiente
* [ ] Desenvolvimento do pré-processamento
* [ ] Engenharia de atributos
* [ ] Implementação dos modelos
* [ ] Containerização
* [ ] Versionamento dos dados
* [ ] Tracking de experimentos
* [ ] Avaliação dos modelos
* [ ] Documentação
* [ ] Vídeo STAR

---

## Dataset

**Dataset escolhido:** *(preencher)*

**Fonte:** *(preencher)*

**Descrição:**

*(Explicar o conjunto de dados utilizado.)*

---

## Tecnologias

* Python
* PyTorch
* Scikit-Learn
* MLflow
* DVC
* Docker
* UV
* Git

---

## Resultados

Esta seção será atualizada conforme o desenvolvimento do projeto.

---

## Aprendizados

Ao final do projeto, serão documentados:

* principais desafios;
* decisões arquiteturais;
* dificuldades encontradas;
* melhorias futuras.

---

## Referências

* Documentação oficial das bibliotecas utilizadas.
* Material das disciplinas da FIAP.
