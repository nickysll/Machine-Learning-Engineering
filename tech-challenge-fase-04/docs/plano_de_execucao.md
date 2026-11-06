# Plano de execução — 04 a 17/11/2026

## Objetivo
Demonstrar que um pipeline de score de inadimplência consegue rejeitar dados inválidos, sinalizar mudanças distribucionais e apresentar evidências operacionais e de governança.

## Critérios acadêmicos (conforme PDF)
- Qualidade de dados: 25%.
- Detecção de drift: 25%.
- Observabilidade e logs: 20%.
- Governança e LGPD: 15%.
- Vídeo STAR (até 5 min): 15%.

## Marcos
- 04–05/11: estrutura, ambiente, obtenção do dataset, schema, treino inicial.
- 06–08/11: validar exemplos negativos, testes, baseline e evidências.
- 09–11/11: simular duas variáveis, Evidently, testes PSI/KS, HTML, separar drift de dados de concept drift.
- 12–13/11: logs, volume, latência, alerta, painel/MLflow.
- 14–15/11: README final LGPD, vieses, retenção, análise causal e checklist.
- 16/11: roteiro STAR e gravação (até 5 minutos).
- 17/11: checklist, link de vídeo, revisar commits e entregar.

## Critérios de aceite da Etapa 1
- Dataset com mais de 5.000 linhas; fonte e licença citadas.
- `train`, `test`, `production_holdout` sem intersecção de `ID`.
- Baseline treinado somente com `train`; métricas calculadas em `test`.
- Batch correto aprovado; batch com quatro tipos de erros bloqueado.
- Relatórios JSON persistidos; testes pytest verificam regras.
- GitHub: commits semânticos; sem upload de dados/artefatos locais.

## Riscos e mitigação
- Dependências incompatíveis: Python 3.11/3.12 em venv e `pip install`.
- Download indisponível: baixar dataset da UCI e adaptar ingestão local.
- Relatório com apenas mudança de distribuição: separar evidência de Data Drift de Concept Drift, que exige comparação da relação X→Y.
- Pouco tempo para dashboard: priorizar relatório HTML Evidently e logs funcionais.
- Inferência causal indevida: distinguir hipótese explicativa de demonstração causal.
