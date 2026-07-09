# Model Card - Recommender Net

## Visão geral

Este Model Card documenta o modelo principal desenvolvido para o Tech Challenge da Fase 02.

A proposta do projeto foi construir um sistema de recomendação usando uma rede neural em PyTorch. Como base, foi utilizado o dataset MovieLens, que contém avaliações de usuários para filmes. Embora o domínio original seja de filmes, a estrutura do problema é parecida com um cenário de e-commerce: temos usuários, itens e interações entre eles.

O modelo tenta aprender padrões nessas interações para prever se um usuário teria uma interação positiva com determinado item.

---

## Modelo desenvolvido

O modelo final foi uma rede neural baseada em embeddings, chamada no projeto de `RecommenderNet`.

A ideia principal foi representar usuários e itens como vetores numéricos aprendidos durante o treinamento. Depois, essas representações são combinadas e passam por camadas densas para gerar uma previsão binária:

- interação positiva;
- interação negativa.

Essa abordagem foi escolhida porque embeddings são bastante usados em problemas de recomendação, já que conseguem capturar relações latentes entre usuários e itens sem depender apenas de regras manuais.

---

## Objetivo do modelo

O objetivo do modelo é prever a probabilidade de uma interação positiva entre um usuário e um item.

No contexto do dataset utilizado, a interação positiva foi definida a partir da nota dada pelo usuário:

| Regra | Classe |
|---|---|
| `rating >= 4` | Interação positiva |
| `rating < 4` | Interação negativa |

Essa transformação permitiu tratar o problema como uma tarefa supervisionada de classificação binária.

---

## Dataset utilizado

O projeto utiliza o dataset **MovieLens Latest Small**, baixado via KaggleHub.

Arquivos principais utilizados:

- `ratings.csv`
- `movies.csv`
- `tags.csv`
- `links.csv`

O arquivo mais importante para o treinamento foi o `ratings.csv`, pois contém as interações entre usuários e itens.

Após o processamento, o projeto gerou:

| Base | Quantidade de registros |
|---|---:|
| Base processada | 100.836 |
| Treino | 80.668 |
| Teste | 20.168 |

---

## Principais etapas antes do treinamento

Antes do treinamento do modelo, o pipeline executa algumas etapas:

1. download do dataset;
2. tratamento das interações;
3. criação do target binário;
4. mapeamento de usuários e itens para identificadores internos;
5. separação entre treino e teste;
6. treinamento do modelo neural;
7. geração das métricas finais.

Essas etapas foram organizadas com DVC para permitir reprodutibilidade do pipeline.

---

## Parâmetros do treinamento

Os principais parâmetros usados no treinamento foram:

| Parâmetro | Valor |
|---|---:|
| Batch size | 512 |
| Épocas máximas | 20 |
| Learning rate | 0.001 |
| Embedding dim | 32 |
| Hidden dim | 64 |
| Dropout | 0.2 |
| Patience | 3 |
| Seed | 42 |
| Device | CPU |

O treinamento utilizou early stopping. Isso significa que o modelo poderia treinar por até 20 épocas, mas pararia antes caso a perda de validação deixasse de melhorar.

Na execução final, o treinamento foi interrompido automaticamente após a estabilização da validação.

---

## Resultados obtidos

As métricas finais do modelo foram:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.6989 |
| Precision | 0.6809 |
| Recall | 0.7059 |
| F1-score | 0.6932 |
| ROC AUC | 0.7650 |
| Loss | 0.5822 |

De forma geral, o modelo apresentou um resultado consistente para uma primeira versão neural do recomendador.

A acurácia ficou próxima de 70%, e o F1-score também ficou equilibrado, indicando que o modelo não ficou extremamente enviesado apenas para uma das classes.

O principal resultado observado foi o ROC AUC de 0.7650. Essa métrica sugere que o modelo conseguiu separar interações positivas e negativas melhor do que uma escolha aleatória. Ainda há espaço para melhoria, mas o resultado mostra que a rede conseguiu aprender padrões úteis nas interações entre usuários e itens.

---

## Como interpretar o resultado

Este modelo não deve ser entendido como uma solução final de recomendação pronta para produção.

Ele funciona como uma primeira versão estruturada, com foco em:

- criar um pipeline reprodutível;
- treinar uma rede neural em PyTorch;
- registrar métricas;
- salvar artefatos;
- organizar o projeto com boas práticas de MLOps.

O resultado mostra que a abordagem é viável, mas ainda precisaria passar por mais testes e comparações antes de ser usada em um ambiente real.

---


## Próximos passos

Algumas melhorias possíveis para próximas versões:

- implementar métricas como Precision@K, Recall@K e NDCG@K;
- comparar o modelo neural com outros algoritmos de recomendação;
- testar diferentes tamanhos de embedding;
- ajustar hiperparâmetros;
- incluir mais atributos dos itens;
- adicionar informações temporais;
- registrar o modelo no MLflow Model Registry;
- criar uma API para servir recomendações;
- automatizar validações com CI/CD.

---

## Artefatos gerados

O treinamento gera os seguintes arquivos principais:

```text
models/recommender_net.pt
reports/torch_metrics.json
