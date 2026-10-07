---
project: eda
task: classification
dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing
team: []
ai_use: "Codex: elaboração de código, análise e redação; revisão da equipe pendente."
---

# 1. EDA — Bank Marketing

**Equipe:** preencher nomes completos e GitHub antes da entrega.

O relatório completo, com código executado, tabelas, interpretações e nove figuras, está no
[notebook de EDA](index.ipynb), que também é a página principal desta entrega no menu.

## 0. Proposal

Foi selecionado **Bank Marketing**, da [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing),
na versão `bank-full.csv`, licenciada sob CC BY 4.0. A escolha foi registrada na
[conversa de referência](https://chatgpt.com/s/t_6ac6c39017b08191b83c26d6dc3f00dc).
Aprovação pelo docente: confirmar e registrar o status real.

O objetivo é prever adesão a depósito a prazo antes da ligação. A combinação de dados
numéricos e categóricos, histórico de campanhas e desbalanceamento exige preparação
cuidadosa. O dicionário da versão está em `code/bank-names.txt`.

## 1. Initial inspection

O arquivo contém **45,211 registros, 16 entradas (7 numéricas e 9 categóricas) e o alvo `y`**.
Não há NaN ou duplicatas completas. O maior desconhecimento semântico é em
`poutcome`: **81.75%** de `unknown`. `pdays=-1` indica ausência de contato anterior.

A classe `yes` representa **5,289 registros (11.70%)**;
a razão entre classes é **7.55:1**. O baseline sempre `no` tem
**88.30% de acurácia** e recall positivo zero.
O split estratificado, seed 42, reserva **36,168 registros para treino** e
**9,043 para teste**, antes das transformações.

## 2. Univariate analysis

As estatísticas de todas as numéricas e frequências de todas as categóricas estão no notebook.
Caudas longas e escalas distintas motivam compressão logarítmica e padronização.
Categorias raras são sinalizadas; `unknown` é preservado como categoria.
A regra IQR marca **9,757 linhas de treino** nas numéricas do modelo,
mas nenhuma linha é removida ou winsorizada. A regra em contagens com IQR zero exige cautela.

## 3. Bivariate and multivariate analysis

O par mais correlacionado é **pdays × previous (Spearman 0.986)**.
A correlação cai para **−0,099** nos clientes previamente contatados: a sentinela explica
boa parte da associação global. Resultado de campanha anterior diferencia taxas de adesão,
sem demonstrar causalidade.

A duração mediana é **426s em yes** e
**164s em no**. Essa relação é relevante para investigar vazamento,
mas a exclusão se deve à indisponibilidade da variável antes da ligação.

## 4. Preprocessing

### Plano de pré-processamento {#8-plano-de-pre-processamento}

O [pipeline importável](code/preprocessing.py) exclui `duration` e, conservadoramente,
`campaign`; aplica signed-log ao saldo, log1p ao histórico, indicador de nunca contatado,
imputação no treino, StandardScaler e one-hot. `day` recebe encoding categórico.
Categorias inéditas e faltas reais foram verificadas em um caso sintético.

A saída tem **80 features**: treino **(36168, 80)**,
teste **(9043, 80)**, sem NaN ou infinitos. PCA explica
**27.27% com PC1+PC2**; 90% requer **32 componentes**.
t-SNE (perplexity 10/30) e UMAP (n_neighbors 5/30) usam a mesma amostra de 2.000 registros
de treino. As classes permanecem misturadas nas projeções; ilhas não comprovam separabilidade.

## 5. Synthesis

Usar PR-AUC, recall, precision, F1 e balanced accuracy na próxima entrega; reservar validação
dentro do treino e reajustar o pipeline em cada fold. O teste não escolhe parâmetros.
A falta de ano e ID de cliente limita avaliação temporal e por cliente. Confirmar se dia,
mês e canal planejados estão disponíveis no instante da previsão; se não estiverem, removê-los.

### Results summary

| # | Métrica | Valor |
|---|---|---|
| 1 | Dataset, tarefa e alvo | Bank Marketing / bank-full.csv; classificação; y=yes/no |
| 2 | Instâncias × features (numéricas / categóricas) | 45,211 × 16 (7 / 9), mais y |
| 3 | Coluna com mais NaN e % | Todas empatadas: 0 (0,00%) |
| 4 | Maior desconhecimento semântico | poutcome: 81.75% unknown |
| 5 | Colunas removidas e motivo | duration: pós-ligação; campaign: inclui contato atual |
| 6 | Classe minoritária | yes: 11.70% (5,289) |
| 7 | Razão de desbalanceamento | 7.55:1 |
| 8 | Baseline sempre no | Acurácia 88.30%; recall positivo 0% |
| 9 | Treino / teste | 36,168 / 9,043 |
| 10 | Par numérico mais correlacionado | pdays × previous: Spearman 0.986 |
| 11 | Maior |correlação numérica–alvo| | duration: 0.343 (inclui features excluídas) |
| 12 | Duplicatas / inconsistências nas regras | 0 / 0 |
| 13 | Linhas com flags IQR nas numéricas do modelo | 9,757 (treino) |
| 14 | Flags nas caudas comprimidas por log | 9,565 linhas; removidas 0; winsorizadas 0 |
| 15 | Total de linhas com transformação log relevante | 33,725 (treino) |
| 16 | Variância PC1 + PC2 | 27.27% |
| 17 | Features antes / após encoding | 16 brutas → 14 retidas + indicador → 80 finais |
| 18 | Shape após pipeline (treino / teste) | (36168, 80) / (9043, 80) |
| 19 | NaN / infinitos após pipeline | 0 / 0 em ambas as partições |

## Reprodução e referências

Instale `code/requirements.txt` e execute todas as células do [notebook](index.ipynb).
O CSV oficial acompanha esta entrega e é validado por SHA-256; há download automático de fallback.
O notebook registra versões das bibliotecas. Os resultados também estão em
`code/results-summary.csv` e `code/results-summary.json`.

- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- Moro, S.; Laureano, R.; Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing:
  An Application of the CRISP-DM Methodology*. ESM'2011, pp. 117–121.
- [Site da disciplina](https://insper.github.io/ann-dl/): Projects → EDA.

Preencher nomes, confirmar aprovação e revisar hipóteses e declaração de uso de IA antes da entrega.
