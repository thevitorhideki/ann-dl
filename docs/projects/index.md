# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Se os nomes não estiverem aqui, não há
    como atribuir a nota — e o mesmo vale para o `mkdocs.yml`, cujo `site_author` deve
    listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| | | |
| | | |
| | | |

Times de 2 a 3 pessoas. Repita esses nomes no cabeçalho de cada entrega — quem corrige pode
abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.ipynb) |
| 2 | Classificação **ou** Regressão | [Classificação](classification/index.md) · [Regressão](regression/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

!!! warning "Escolha uma: classificação ou regressão"

    A segunda entrega é **uma das duas**, não as duas. Este template traz as duas pastas
    para você escolher; depois de decidir, apague a que não vai usar — da pasta `docs/projects/`
    **e** da `nav` no `mkdocs.yml`.

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
- [ ] **2. Classificação ou Regressão**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| | | |
