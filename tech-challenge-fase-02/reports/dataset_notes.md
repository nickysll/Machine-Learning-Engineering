# Dataset Notes

## Dataset escolhido

MovieLens Latest Small

## Objetivo

Usar interações entre usuários e itens para treinar um sistema de recomendação.

## Justificativa

O dataset foi escolhido por conter interações usuário-item em volume suficiente para o desenvolvimento de um sistema de recomendação. Embora o domínio seja filmes, a estrutura do problema é equivalente a um e-commerce, em que usuários interagem com itens e o modelo aprende padrões para recomendar novos produtos.

## Arquivos principais

- `ratings.csv`: interações entre usuários e itens
- `movies.csv`: informações dos itens
- `tags.csv`: tags atribuídas por usuários
- `links.csv`: identificadores externos dos filmes

## Colunas principais do `ratings.csv`

- `userId`: identificador do usuário
- `movieId`: identificador do item
- `rating`: avaliação do usuário
- `timestamp`: momento da interação

## Definição de interação

Uma interação ocorre quando um usuário avaliou um item.

## Target inicial

Para transformar o problema em classificação binária:

- `rating >= 4`: interação positiva
- `rating < 4`: interação negativa

## Uso no projeto

O dataset será usado para:

- preprocessamento
- engenharia de features
- baseline com Scikit-Learn
- modelo neural com PyTorch
- tracking com MLflow
- pipeline com DVC