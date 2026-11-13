# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! info "Identificação da equipe"

    A equipe abaixo responde pelas entregas do projeto. Os nomes também aparecem no
    cabeçalho da EDA e no campo `site_author` do `mkdocs.yml`.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Vitor Hideki Pereira Katakura | <vitorhpk@al.insper.edu.br> | [@thevitorhideki](https://github.com/thevitorhideki) |
| Antonio Anderson de Araujo Julião | <antonioaaj@al.insper.edu.br> | [@andersonjuIiao](https://github.com/andersonjuIiao) |

Times de 2 a 3 pessoas. Repita esses nomes no cabeçalho de cada entrega — quem corrige pode
abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.ipynb) |
| 2 | Classificação | [Classificação](classification/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

!!! info "Tarefa escolhida: classificação"

    A segunda entrega continua a classificação binária definida no EDA. A opção de regressão
    foi removida da estrutura e da navegação do site.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

| | |
|---|---|
| **Nome** | UCI Bank Marketing — versão `bank-full.csv` |
| **Fonte (URL)** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Licença / termos de uso** | CC BY 4.0; documentação da versão em `eda/code/bank-names.txt` |
| **Amostras** | 45.211 registros |
| **Features** | 16 entradas brutas: 7 numéricas e 9 categóricas |
| **Variável alvo** | `y`: contratação de depósito a prazo (`yes` / `no`) |
| **Tarefa escolhida** | Classificação binária |

O dataset combina perfil do cliente, dados bancários e histórico de campanhas. A classe
positiva minoritária, as escalas heterogêneas e categorias desconhecidas exigem preparação
cuidadosa para redes neurais. O cenário é a previsão antes da ligação, com exclusão de
`duration` e, conservadoramente, `campaign`. A disponibilidade de dia, mês e canal planejados
é uma hipótese a confirmar. Aprovação pelo docente: preencher com o status real.

## Status

- [ ] **1. EDA**
- [ ] **2. Classificação**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 01/11/2026 | Entrega 2 é **classificação**; pasta e item de menu de regressão removidos | O alvo `y` é binário e o EDA já foi feito para classificação |
| 01/11/2026 | Validação = 15% do treino do EDA, estratificada, seed 42; teste do EDA intocado até o passo 5 | O enunciado exige o mesmo split do EDA e a validação tirada do treino |
| 01/11/2026 | Métrica principal PR-AUC (`yes`); critério: rede ≥ 1,05 × melhor baseline no teste e diferença maior que um desvio padrão entre seeds | Com 11,70% de positivos, a acurácia isolada mascara o desempenho na classe minoritária |
| 01/11/2026 | Limiar escolhido na validação pelo custo FP:FN = 1:5 | Perder um contratante custa mais que uma ligação a mais (hipótese da equipe) |
| 01/11/2026 | Rede final 128-64-32, dropout 0,3, weight decay 0,05; sem `pos_weight` | Melhor PR-AUC de validação entre 16 configurações; `pos_weight` piorou todas |
| 13/11/2026 | Integração e conferência dos resultados salvos, alinhamento do critério automatizado e correção do resumo de subgrupos | Preservar a seleção original por validação e distinguir maior taxa de erro de pior recall |
