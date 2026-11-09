# Tech Challenge — Fase 4 | Credit Scoring Reliability

**Projeto acadêmico de confiabilidade e observabilidade de um modelo de classificação de inadimplência.** O objetivo é detectar entradas inválidas, simular mudanças no perfil dos dados, verificar alterações nas previsões e registrar evidências operacionais, com limites de uso e governança documentados.

> **Atenção:** este é um protótipo acadêmico. Não é um sistema validado para aprovação, recusa ou precificação de crédito real.

## 1. Contexto e objetivo

Em um cenário fictício de concessão de crédito, um modelo pode continuar executando sem erros técnicos mesmo quando as características da população atendida mudam. O projeto demonstra quatro camadas de sustentação:

1. **Qualidade:** bloquear lotes incompatíveis com contratos de dados.
2. **Drift:** simular mudanças nas variáveis de entrada e comparar distribuições com uma referência.
3. **Observabilidade:** registrar execução, volume, duração, métricas e alertas em MLflow e logs estruturados.
4. **Governança:** documentar finalidade, limitações, privacidade, riscos de viés e hipóteses causais.

## 2. Dados e separação dos conjuntos

- **Fonte:** UCI Machine Learning Repository — *Default of Credit Card Clients* (I-Cheng Yeh; Taiwan, 2005).
- **Referência:** 30.000 observações; DOI: <https://doi.org/10.24432/C55S3H>.
- **Licença informada pela fonte:** CC BY 4.0. Atribuir a autoria e consultar os termos originais antes de qualquer redistribuição.
- **Limitação de representatividade:** a amostra histórica de Taiwan não representa o mercado brasileiro atual.

| Conjunto | Registros | Finalidade |
|---|---:|---|
| `data/train.csv` | 18.000 (60%) | Ajuste do classificador e referência estatística para drift |
| `data/test.csv` | 6.000 (20%) | Avaliação do baseline sem simulação de drift |
| `data/production_holdout.csv` | 6.000 (20%) | Base intacta para criação de cenários artificiais de produção |

A separação é **aleatória e estratificada, não temporal**. As janelas de “produção” são **simuladas**, e não períodos históricos reais. O identificador `ID` é utilizado para verificações de integridade, não como evidência de identidade pessoal verificada.

## 3. Ambiente e reprodução (WSL/Ubuntu + Poetry)

O desenvolvimento foi validado em **Python 3.12**, com ambiente virtual da **Fase 4**. Execute os comandos a partir de `tech-challenge-fase-04` (não da raiz do monorepositório nem do ambiente virtual da Fase 3).

```bash
cd ~/Machine-Learning-Engineering/tech-challenge-fase-04
poetry install
source .venv/bin/activate
which python
python -m pytest tests/ -v
```

O caminho de `which python` deve terminar em `tech-challenge-fase-04/.venv/bin/python`. Uma alternativa que dispensa ativação manual é usar `poetry run python` em cada comando. O arquivo `pyproject.toml` e o `poetry.lock` são a referência para dependências; `requirements.txt` corresponde ao fluxo inicial do projeto e pode estar desatualizado.

### Fluxo de reprodução

Execute as etapas a seguir **separadamente**, nessa ordem, partindo de um ambiente configurado e com acesso à fonte pública de dados quando necessário:

```bash
# Etapa 1 — preparação, contratos e baseline
python -m src.prepare_data
python -m src.validate_batch --batch valid
python -m src.validate_batch --batch invalid
python -m src.train

# Etapa 2 — cenários, relatórios estatísticos e previsões
python -m src.simulate_drift
python -m src.detect_drift
python -m src.analyze_predictions
python -m src.compare_scenarios

# Etapa 3 — experimentos e monitoramento operacional
python -m src.monitor_experiments
python -m src.monitor_pipeline
python -m pytest tests/ -v

# Interface de rastreamento (enquanto o terminal estiver aberto)
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Abra `http://localhost:5001` e selecione **Model training**. A porta 5001 foi usada porque a 5000 estava ocupada na máquina de desenvolvimento. **Observação:** `validate_batch --batch invalid` deve rejeitar o lote e pode terminar com código de saída `1`; isso faz parte da demonstração do contrato e não impede executar o comando seguinte separadamente. Os scripts de etapas posteriores dependem dos arquivos gerados anteriormente. Os comandos listados documentam o fluxo concebido; revise os scripts caso seus argumentos mudem.

## 4. Etapa 1 — contratos de dados e modelo baseline

### Contratos de qualidade

`src/contracts.py` utiliza **Pandera** para validar, entre outros aspectos: ausência de nulos obrigatórios, unicidade de `ID`, `AGE` entre 18 e 100, `LIMIT_BAL` positivo, valores de pagamento não negativos e faixas esperadas de indicadores de atraso.

O lote inválido foi construído para demonstrar rejeição de idade fora da faixa, limite negativo, pagamento ausente e identificador duplicado. Os resultados de validação são gerados em `reports/validation_valid.json` e `reports/validation_invalid.json`.

### Modelo e resultados

Classificador baseline: **regressão logística**, com escalonamento, imputação e pesos balanceados. Avaliação no conjunto de teste de 6.000 registros:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,6945 |
| Precision (classe positiva: inadimplência) | 0,3847 |
| Recall | 0,6360 |
| F1 | 0,4794 |
| ROC-AUC | 0,7174 |
| Average Precision | 0,4857 |

Matriz de confusão (linhas: rótulo real 0/1; colunas: previsão 0/1):

| | Predito 0 | Predito 1 |
|---|---:|---:|
| Real 0 | 3.323 | 1.350 |
| Real 1 | 483 | 844 |

Os **1.350 falsos positivos** representam risco de classificar como inadimplentes clientes que não pertencem à classe positiva observada no teste; **483 falsos negativos** representam casos positivos não identificados. Esses custos não foram convertidos em impactos financeiros reais. Resultados completos: `reports/baseline_metrics.json`.

## 5. Etapa 2 — simulação e detecção de Data Drift

`src/simulate_drift.py` modifica simultaneamente **duas variáveis** sobre os mesmos 6.000 registros do `production_holdout.csv`:

| Cenário | `LIMIT_BAL` | `AGE` |
|---|---|---|
| `baseline` | Sem mudança | Sem mudança |
| `moderate` | Redução artificial de 15% | Acréscimo artificial de 3 anos |
| `severe` | Redução artificial de 30% | Acréscimo artificial de 7 anos |

O objetivo é testar a resposta do sistema a perturbações controladas de entrada; **não** afirmar que essas transformações refletem evolução demográfica, inflação ou inadimplência reais.

### Método estatístico

`src/detect_drift.py` produz relatórios **Evidently AI** (`DataDriftPreset`) e executa separadamente o teste **Kolmogorov–Smirnov (KS)** em seis features:

`LIMIT_BAL`, `AGE`, `PAY_0`, `PAY_2`, `BILL_AMT1` e `PAY_AMT1`.

- Critério exploratório não corrigido: `p < 0,05` (campo `drift_detected` nos JSONs KS).
- Critério dos **alertas operacionais**: correção de **Bonferroni**, com `α = 0,05 / 6 ≈ 0,00833`.
- A contagem exibida pelo **Evidently** pode usar critérios próprios e não deve ser confundida automaticamente com a contagem KS corrigida do monitoramento.
- O KS clássico exige cautela na interpretação de variáveis discretas ou com muitos valores repetidos.

| Cenário | Features com `p < 0,05` | Features com Bonferroni | Features significativas após correção |
|---|---:|---:|---|
| Baseline | 1 | 0 | Nenhuma |
| Moderate | 2 | 2 | `LIMIT_BAL`, `AGE` |
| Severe | 2 | 2 | `LIMIT_BAL`, `AGE` |

No baseline, `AGE` apresentou `p = 0,00977265`: significativo somente **sem** correção. No cenário severe, a saída `p = 0` para `AGE` reflete limite de precisão numérica, não probabilidade literalmente nula.

### Mudanças nas classificações

As predições dos cenários foram comparadas com as do cenário baseline:

| Cenário | Previsões alteradas | Taxa | Direção das alterações |
|---|---:|---:|---|
| Baseline | 0 | 0% | — |
| Moderate | 211 | 3,52% | 211 para classe positiva |
| Severe | 508 | 8,47% | 508 para classe positiva |

Esses números caracterizam **mudança nas previsões** (*prediction drift*), não mudança observada da taxa real de inadimplência.

**Arquivos de evidência:** `reports/drift_baseline.html`, `reports/drift_moderate.html`, `reports/drift_severe.html`, `reports/drift_ks_*.json`, `reports/prediction_comparison.json` e `reports/classification_changes.json`. Os três HTML foram gerados com Evidently e destinados à demonstração visual; seu conteúdo deve ser inspecionado antes da publicação externa.

### Interpretação dos alertas de drift

No cenário severo, o Evidently identificou drift em duas das seis variáveis analisadas (`AGE` e `LIMIT_BAL`), correspondendo a 33,33% das features monitoradas.

Entretanto, o alerta global de Dataset Drift não foi acionado, pois o relatório utiliza um limite de 50% de colunas com drift.

O pipeline operacional adota um critério diferente: gera um alerta `DATA_DRIFT` quando pelo menos uma variável apresenta significância estatística após correção de Bonferroni.

Assim, os resultados não são contraditórios, mas refletem diferentes níveis de monitoramento e regras de decisão.

## 6. Etapa 3 — observabilidade e monitoramento

O projeto possui dois experimentos no **MLflow**, utilizando armazenamento local SQLite (`mlflow.db`):

| Experimento | Objetivo |
|---|---|
| `fase-04-credit-drift` | Registrar comparação estatística, métricas de risco/previsão e artefatos dos cenários |
| `fase-04-pipeline-monitoring` | Registrar saúde operacional, quantidade de registros, duração, drift, alertas e status |

O script `src/monitor_experiments.py` registra contagens de drift **bruta e corrigida**, parâmetros de significância e artefatos dos experimentos. O script `src/monitor_pipeline.py` aplica o contrato de entrada, lê relatórios estatísticos e de previsão, gera alertas e escreve logs estruturados.

### Métricas e alertas

| Métrica/atributo | Interpretação |
|---|---|
| `records_processed` | Registros processados no cenário |
| `duration_seconds` | Tempo medido até o ponto de observabilidade no script; **não** é benchmark de latência ponta a ponta de uma API |
| `drifted_features_raw` | Features com KS significativo antes da correção |
| `drifted_features` | Features significativas após Bonferroni |
| `classification_change_rate` | Proporção de classificações diferentes da baseline |
| `predicted_positive_rate` | Proporção prevista da classe positiva |
| `alert_count` | Quantidade de regras de alerta acionadas |
| `pipeline_status` | Tag operacional: `SUCCESS`, `WARNING` ou `FAILED` |

Regras experimentais:

- `DATA_DRIFT`: disparado se `drifted_features > 0` após Bonferroni.
- `PREDICTION_CHANGE`: disparado se `classification_change_rate > 0,05` (**mais de 5%**).
- `SUCCESS`: execução concluída sem alerta.
- `WARNING`: execução concluída com um ou mais alertas.
- `FAILED`: erro que impediu a execução, com `error_type` e `error_message` registrados quando disponíveis.

| Cenário | Status operacional | Features corrigidas | Taxa de mudança | Alertas |
|---|---|---:|---:|---|
| Baseline | `SUCCESS` | 0 | 0% | Nenhum |
| Moderate | `WARNING` | 2 | 3,52% | `DATA_DRIFT` |
| Severe | `WARNING` | 2 | 8,47% | `DATA_DRIFT`, `PREDICTION_CHANGE` |

O status **Finished** da run no MLflow significa que o registro técnico terminou normalmente; ele não substitui a tag de saúde operacional `pipeline_status=WARNING`. Para falhas, o projeto também suporta o status MLflow `FAILED`.

Os logs locais ficam em `logs/pipeline.jsonl` (uma linha JSON por execução) e `logs/pipeline.log` (log de aplicação). A persistência de uma run `FAILED` foi verificada por teste de integração usando **um banco SQLite temporário**, preservando o banco principal.

### Testes

```bash
python -m pytest tests/ -v
```

Na validação documentada em **08/11/2026**: **12 testes aprovados** — sete de contratos e cinco de monitoramento (três cenários, tratamento de falha e indisponibilidade do MLflow). Foi realizado separadamente um teste de integração real de falha com resultado `FAILED` tanto no pipeline quanto no MLflow. Os testes automatizados usam isolamento/simulações para evitar alterar o banco principal.

**Limites:** não há medição contínua de CPU, memória ou disponibilidade da infraestrutura, nem monitoramento em produção de uma API. Os limiares são demonstrativos, não calibrados com dados reais ou níveis de serviço. Alertas são identificados e registrados; o protótipo não implementa notificação externa nem resposta operacional automática.

## 7. Etapa 4 — governança, privacidade e causalidade

### Finalidade e privacidade por concepção (*privacy by design/default*)

Este projeto utiliza dados públicos históricos **somente para demonstração acadêmica**. A divulgação pública da fonte não implica autorização ilimitada para tratar dados de pessoas. Antes de qualquer uso real, seriam necessárias avaliação jurídica e documentação da finalidade, minimização de dados, identificação dos controladores/operadores e definição da base legal adequada à operação de crédito conforme a LGPD.

Em uma implantação efetiva, o plano de proteção deve contemplar:

- **Inventário e minimização:** documentar campos utilizados, finalidade e necessidade; evitar coletar ou registrar identificadores e atributos desnecessários.
- **Acesso:** perfis de menor privilégio, autenticação e segregação entre dados de treino, produção e auditoria.
- **Segurança:** criptografia em trânsito e repouso, gestão de segredos, registros de acesso e resposta a incidentes.
- **Logs seguros:** preferir métricas agregadas e IDs técnicos pseudonimizados; não incluir PII em erros, tags ou artefatos exportados.
- **Direitos dos titulares:** processo para informação, correção e revisão/contestação conforme o caso e a legislação aplicável.
- **Avaliação pré-produção:** revisão jurídica, de privacidade e de risco de modelo, inclusive para decisões automatizadas relevantes.

### Base legal e retenção

O projeto **não determina uma base legal universal** para concessão de crédito: em contexto real, o enquadramento deve ser analisado conforme finalidade e operação concreta, com suporte jurídico e do encarregado. A licença do dataset não substitui essa análise.

**Política proposta para o protótipo:** manter dados e artefatos locais somente durante o desenvolvimento e a avaliação acadêmica; depois revisar necessidade de retenção e eliminar cópias desnecessárias de forma segura. Para uma operação real, estabelecer prazos específicos por categoria de informação com justificativa legal, controles de descarte e trilha de auditoria. Os prazos não foram implementados nem validados automaticamente neste protótipo.

### Vieses, erros e mitigação

O dataset inclui atributos demográficos e o modelo utiliza `AGE`, variável que exige cautela por poder influenciar decisões desfavoráveis e estar associada a desigualdades. A amostra também tem deslocamento geográfico e temporal significativo. O modelo apresentou **1.350 falsos positivos** e **483 falsos negativos** no teste, evidenciando riscos distintos para clientes e instituição.

Antes de produção seriam necessários: auditoria por grupos relevantes e legalmente justificáveis, comparação de taxas de erro e calibração, análise de proxies, avaliação de impacto, limiares de decisão adequados ao risco, mecanismos de revisão humana e monitoramento periódico. **Essas avaliações de fairness não foram executadas como comprovação de conformidade.**

### Análise causal — hipóteses e limites

Foram aplicadas duas intervenções artificiais **ao mesmo tempo**: redução de `LIMIT_BAL` e aumento de `AGE`. Isso permite observar como o **modelo treinado** reage ao conjunto dessas perturbações, mas **não identifica o efeito isolado de cada variável**. Tampouco demonstra que envelhecimento, redução do limite ou inflação **causem inadimplência real**.

Para investigar causalidade em um cenário de negócio seriam necessários, conforme o desenho apropriado: um diagrama causal e hipóteses explícitas, controle de fatores de confusão, comparação contrafactual ou intervenção identificável, evidência temporal confiável e avaliação de mecanismos de seleção. Um experimento adicional alterando uma variável por vez ajudaria a decompor a **sensibilidade das predições**, mas ainda não provaria causalidade na inadimplência observada.

### Data Drift, Prediction Drift e Concept Drift

- **Data Drift (`P(X)`):** mudanças na distribuição das características — **demonstradas** nos cenários moderate/severe.
- **Prediction Drift:** mudanças nas saídas do classificador — **demonstradas** pelas 211 e 508 classificações alteradas.
- **Concept Drift (`P(Y|X)`):** mudança na relação entre entradas e desfecho — **não demonstrada** por esta análise, pois não foram observados novos rótulos confiáveis ao longo do tempo.

Não é correto inferir perda de desempenho preditivo em produção apenas a partir de drift de features ou de previsões.

## 8. Estrutura e artefatos

```text
tech-challenge-fase-04/
├── README.md
├── pyproject.toml / poetry.lock
├── src/
│   ├── prepare_data.py / contracts.py / validate_batch.py / train.py
│   ├── simulate_drift.py / detect_drift.py
│   ├── analyze_predictions.py / compare_scenarios.py
│   └── monitor_experiments.py / monitor_pipeline.py
├── tests/
│   ├── test_contracts.py
│   └── test_monitor_pipeline.py
├── data/               # arquivos locais de trabalho
├── models/             # artefatos locais
├── reports/            # JSON de métricas e relatórios HTML
├── logs/               # arquivos locais de execução
└── docs/
    └── plano_de_execucao.md
```

**Controle de versão:** `mlflow.db`, `logs/`, backups e dados locais de produção/simulação são mantidos fora do Git. Os três relatórios HTML do Evidently foram liberados no `.gitignore` para eventual inclusão como evidência. **Antes de publicar**, verificar se contêm dados por linha, campos identificáveis ou conteúdo que não deva ser compartilhado; o HTML pode incorporar dados na própria página. Não executar `git add .` sem revisar a lista de arquivos.

## 9. Evidências e apresentação

Para demonstrar o projeto, apresentar: (a) rejeição do lote inválido; (b) os três HTML do Evidently e as features com drift; (c) as diferenças de classificação dos cenários; (d) execuções e tags no MLflow, incluindo alertas; (e) resultado `12 passed` e a evidência de falha `FAILED` registrada; (f) medidas de privacidade e limites de inferência.

**Vídeo obrigatório:** até **5 minutos**, estruturado pelo método **STAR**:

- **Situation:** risco de degradação silenciosa no crédito.
- **Task:** validar entradas, detectar drift, rastrear saúde e documentar governança.
- **Action:** Pandera, classificador baseline, simulações, Evidently/KS, Bonferroni, MLflow e testes.
- **Result:** alertas baseline/moderate/severe, 211/508 alterações de previsão, testes aprovados, limitações e ações recomendadas.

**Link do vídeo:** _pendente de gravação e inserção antes da entrega_.

## 10. Limitações e uso responsável

Os resultados provêm de uma simulação sobre um dataset histórico. Não há validação externa em mercado brasileiro, evidência de Concept Drift, auditoria completa de fairness, monitoramento contínuo de infraestrutura ou garantia de conformidade legal para operação de crédito. O protótipo **não deve orientar decisões financeiras individuais**.
