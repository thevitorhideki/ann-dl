"""Pré-processamento do bank-full.csv para prever adesão antes da ligação.

Uso: pipeline = build_preprocessor(); pipeline.fit_transform(X_train).
Estatísticas aprendidas (imputação, escala e categorias) vêm apenas do treino.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAW_NUMERIC = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
RAW_CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"
]
DROPPED_COLUMNS = ["duration", "campaign"]
MODEL_NUMERIC = ["age", "balance", "pdays", "previous"]
MODEL_CATEGORICAL = RAW_CATEGORICAL + ["day"]


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Transformações determinísticas: sem aprender estatísticas ou usar o alvo."""
    result = X.drop(columns=DROPPED_COLUMNS, errors="ignore").copy()
    # Saldo pode ser negativo: log com sinal preserva dívida e comprime a cauda.
    result["balance"] = np.sign(result["balance"]) * np.log1p(np.abs(result["balance"]))
    result["previous"] = np.log1p(result["previous"])
    # -1 representa ausência estrutural de contato, não uma duração negativa.
    result["never_contacted"] = (result["pdays"] == -1).astype(float)
    result["pdays"] = np.log1p(result["pdays"].mask(result["pdays"] == -1))
    # Dia do mês não deve impor uma distância linear ao modelo.
    result["day"] = result["day"].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    for col in RAW_CATEGORICAL:
        # unknown permanece categoria explícita; ausência real vira missing.
        result[col] = result[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return result


def build_preprocessor() -> Pipeline:
    """Retorna pipeline novo, ainda não ajustado, com saída densa para a rede."""
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    columns = ColumnTransformer([
        ("num", numerical, MODEL_NUMERIC),
        ("cat", categorical, MODEL_CATEGORICAL),
        ("flag", "passthrough", ["never_contacted"]),
    ], remainder="drop")
    return Pipeline([
        ("prepare", FunctionTransformer(prepare_features)),
        ("columns", columns),
    ])
