# Tech Challenge — Fase 4 | Credit Scoring Reliability

Protótipo acadêmico para confiabilidade de modelo de risco de crédito.

## Contexto

Uma instituição financeira quer identificar entradas inválidas e monitorar a degradação silenciosa de um classificador de inadimplência. **Não é um sistema adequado para uso em decisões reais de crédito.**

## Fonte de dados

UCI Machine Learning Repository — *Default of Credit Card Clients* (30.000 registros, Taiwan, 2005), DOI: https://doi.org/10.24432/C55S3H. Licença CC BY 4.0, autor I-Cheng Yeh. O dataset não representa o mercado brasileiro contemporâneo.

## Etapa 1 — Rodar no VS Code (Python 3.11 ou 3.12)

Na pasta raiz, crie e ative um ambiente virtual e instale `requirements.txt`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.prepare_data
python -m src.validate_batch --batch valid
python -m src.validate_batch --batch invalid
python -m src.train
python -m pytest -q
```

**Nota:** `--batch invalid` termina com código de saída 1 **por design**. Execute `train` na sequência, como comando separado, após verificar o relatório em `reports/validation_invalid.json`.

## Separação de conjuntos

- `train.csv` (60%): ajusta o modelo baseline.
- `test.csv` (20%): mede desempenho de generalização sem drift.
- `production_holdout.csv` (20%): reserva intacta para simulação da Etapa 2.

Divisão aleatória estratificada, não temporal. Para simular passagem de tempo na Etapa 2 serão criadas janelas artificiais, explicitamente identificadas como simulação; nunca serão descritas como dados históricos de produção.

## Contratos

`src/contracts.py` verifica não nulos, unicidade de ID, idade entre 18 e 100, limite de crédito positivo, pagamento não negativo e faixa de dois indicadores de atraso. O batch inválido contém idade 15, limite negativo, pagamento ausente e ID duplicado.

## Baseline

Regressão logística com escalonamento, imputação e pesos balanceados. Métricas: accuracy, precision, recall, F1, ROC-AUC, Average Precision e matriz de confusão. A etapa 1 não possui resultados previamente simulados; rode os scripts e registre métricas produzidas localmente.

## Etapas 2–4 (planejadas, ainda não implementadas)

- Etapa 2: simulação com mudanças de limite de crédito e atraso, Evidently, PSI/KS, análise de desempenho com rótulos simulados ou existentes.
- Etapa 3: logs estruturados, monitoramento de latência/volume/alertas e painel ou MLflow.
- Etapa 4: governança, matriz de riscos, privacidade/LGPD, limites de uso do dataset, análise causal hipotética e vídeo STAR.

## Cuidados de governança (rascunho)

Dados públicos históricos de pesquisa não significam autorização automática para decisões de crédito reais. Evitar publicar dados pessoais, credenciais, conjuntos internos ou dados reais de clientes. Dados demográficos do dataset exigem análise de viés e justificativa de finalidade. A base legal e retenção de uma operação real devem ser validadas com jurídico/encarregado, não inferidas da licença do dataset.

## Organização do projeto

- `src/`: ingestão, contrato, treinamento e validação de lotes.
- `tests/`: testes automatizados de contrato.
- `data/`, `models/`, `reports/`: resultados locais ignorados pelo Git.
- `docs/`: documentação complementar da governança e roteiro STAR (futura).
