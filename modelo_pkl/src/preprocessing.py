import unicodedata
from typing import Any, List, Optional, Sequence

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
import logging


logger = logging.getLogger(__name__)


NUMERIC_FEATURES: List[str] = [
    "coeficiente_rendimento",
    "disciplinas_aprovadas",
    "disciplinas_reprovadas_nota",
    "periodo",
    "ano_ingresso",
    "semestre_ingresso",
    "nota_enem_humanas",
    "nota_enem_linguagem",
    "nota_enem_matematica",
    "nota_enem_natureza",
    "nota_enem_redacao",
    "nota_vestibular_biologia",
    "nota_vestibular_filosofia_sociologia",
    "nota_vestibular_fisica",
    "nota_vestibular_geografia",
    "nota_vestibular_historia",
    "nota_vestibular_literatura_brasileira",
    "nota_vestibular_lingua_estrangeira",
    "nota_vestibular_lingua_portuguesa",
    "nota_vestibular_matematica",
    "nota_vestibular_quimica",
    "idade",
    "calouro",
]


CATEGORICAL_FEATURES: List[str] = [
    "sexo",
    "municipio_residencia",
    "curso",
    "campus",
    "turno",
    "modalidade_ingresso",
    "tipo_cota",
]


ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "aprovacao_ratio",
    "nota_enem_total",
    "nota_vestibular_total",
]


class EnsureColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        numeric_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "EnsureColumnsTransformer":
        if self.numeric_features is None:
            self.numeric_features_ = []
        else:
            self.numeric_features_ = list(self.numeric_features)
        if self.categorical_features is None:
            self.categorical_features_ = []
        else:
            self.categorical_features_ = list(self.categorical_features)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["numeric_features_", "categorical_features_"])
        X_in = X
        X = X.copy()
        if not hasattr(X, "columns"):
            return X
        for col in self.numeric_features_:
            if col not in X.columns:
                X[col] = np.nan
        for col in self.categorical_features_:
            if col not in X.columns:
                X[col] = np.nan
        for col in self.numeric_features_:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        for col in self.categorical_features_:
            if col in X.columns:
                X[col] = X[col].astype(object)
        X = X.reindex(index=X_in.index)
        return X


class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, string_columns: Optional[Sequence[str]] = None) -> None:
        self.string_columns = string_columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataCleaningTransformer":
        if self.string_columns is None:
            self.string_columns_ = None
        else:
            self.string_columns_ = list(self.string_columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(X, "columns"):
            return X
        if self.string_columns_ is None:
            columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
        else:
            columns = [c for c in self.string_columns_ if c in X.columns]
        placeholders = {"na", "n/a", "-", ""}
        for col in columns:
            series = X[col].astype("string")
            series = series.str.strip()
            lower = series.str.lower()
            mask_placeholder = lower.isin(placeholders)
            series = series.mask(mask_placeholder)
            series = series.apply(self._normalize_string)
            series = series.astype(object)
            series[pd.isna(series)] = np.nan
            X[col] = series
        return X

    def _normalize_string(self, value: Any) -> Any:
        if value is None or pd.isna(value):
            return value
        s = str(value).lower()
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        return s


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, enem_prefix: str = "nota_enem_", vestibular_prefix: str = "nota_vestibular_") -> None:
        self.enem_prefix = enem_prefix
        self.vestibular_prefix = vestibular_prefix

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineeringTransformer":
        if hasattr(X, "shape"):
            self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if hasattr(X, "columns"):
            X = self._add_enem_total(X)
            X = self._add_vestibular_total(X)
            X = self._add_approval_ratio(X)
        return X

    def _add_enem_total(self, X: pd.DataFrame) -> pd.DataFrame:
        enem_cols = [c for c in X.columns if c.startswith(self.enem_prefix)]
        if not enem_cols:
            if "nota_enem_total" not in X.columns:
                X["nota_enem_total"] = np.nan
            return X
        data = {c: pd.to_numeric(X[c], errors="coerce") for c in enem_cols}
        frame = pd.DataFrame(data, index=X.index)
        all_nan_mask = frame.isna().all(axis=1)
        filled = frame.fillna(0.0)
        total = filled.sum(axis=1)
        total[all_nan_mask] = np.nan
        X["nota_enem_total"] = total
        return X

    def _add_vestibular_total(self, X: pd.DataFrame) -> pd.DataFrame:
        vestibular_cols = [c for c in X.columns if c.startswith(self.vestibular_prefix)]
        if not vestibular_cols:
            if "nota_vestibular_total" not in X.columns:
                X["nota_vestibular_total"] = np.nan
            return X
        data = {c: pd.to_numeric(X[c], errors="coerce") for c in vestibular_cols}
        frame = pd.DataFrame(data, index=X.index)
        all_nan_mask = frame.isna().all(axis=1)
        filled = frame.fillna(0.0)
        total = filled.sum(axis=1)
        total[all_nan_mask] = np.nan
        X["nota_vestibular_total"] = total
        return X

    def _add_approval_ratio(self, X: pd.DataFrame) -> pd.DataFrame:
        approvals = pd.to_numeric(X.get("disciplinas_aprovadas"), errors="coerce")
        fails_grade = pd.to_numeric(X.get("disciplinas_reprovadas_nota"), errors="coerce")
        
        total = approvals + fails_grade
        
        denom = total.replace(0, np.nan)
        ratio = approvals / denom
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        X["aprovacao_ratio"] = ratio
        return X


class AdaptiveCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        max_onehot_cardinality: int = 20,
        smoothing: float = 10.0,
        min_samples_leaf: int = 20
    ) -> None:
        self.max_onehot_cardinality = max_onehot_cardinality
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AdaptiveCategoricalEncoder":
        X = self._ensure_dataframe(X)
        self.columns_ = list(X.columns)
        nunique = X.nunique(dropna=True)
        self.low_card_cols_ = [
            c for c in self.columns_ if nunique.get(c, 0) <= self.max_onehot_cardinality
        ]
        self.high_card_cols_ = [
            c for c in self.columns_ if nunique.get(c, 0) > self.max_onehot_cardinality
        ]
        if self.low_card_cols_:
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.onehot_.fit(X[self.low_card_cols_])
        else:
            self.onehot_ = None
        if self.high_card_cols_:
            if y is None:
                raise ValueError(
                    "Target y is required for high-cardinality columns using TargetEncoder. "
                    "Ensure you are calling fit(X, y) with a valid target vector."
                )
            
            # Garante alinhamento de índices resetando ambos
            X_high = X[self.high_card_cols_].reset_index(drop=True)
            y_series = pd.Series(y).reset_index(drop=True)
            
            self.global_mean_ = float(y_series.mean())
            self.target_encoder_ = ce.TargetEncoder(
                cols=self.high_card_cols_,
                handle_unknown="value",
                handle_missing="value",
                smoothing=self.smoothing,
                min_samples_leaf=self.min_samples_leaf,
            )
            self.target_encoder_.fit(X_high, y_series)
        else:
            self.target_encoder_ = None
            self.global_mean_ = 0.0
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        X = self._ensure_dataframe(X)
        check_is_fitted(self, ["columns_", "low_card_cols_", "high_card_cols_", "global_mean_"])
        parts: List[np.ndarray] = []
        if self.low_card_cols_ and getattr(self, "onehot_", None) is not None:
            data_low = X[self.low_card_cols_].astype("category")
            arr_low = self.onehot_.transform(data_low)
            parts.append(arr_low)
        if self.high_card_cols_:
            if getattr(self, "target_encoder_", None) is not None:
                data_high = X[self.high_card_cols_]
                arr_high_df = self.target_encoder_.transform(data_high)
                arr_high_df = arr_high_df.fillna(self.global_mean_)
                arr_high = np.asarray(arr_high_df, dtype=float)
            else:
                n_rows = X.shape[0]
                n_cols = len(self.high_card_cols_)
                arr_high = np.full((n_rows, n_cols), float(self.global_mean_), dtype=float)
            parts.append(arr_high)
        if not parts:
            return np.empty((X.shape[0], 0))
        return np.hstack(parts)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, ["columns_", "low_card_cols_", "high_card_cols_"])
        names = []
        if self.low_card_cols_ and getattr(self, "onehot_", None) is not None:
            names.extend(self.onehot_.get_feature_names_out(self.low_card_cols_))
        if self.high_card_cols_:
            names.extend(self.high_card_cols_)
        return np.array(names, dtype=object)

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class DropTechnicalColumns(BaseEstimator, TransformerMixin):
    """
    Remove colunas técnicas e identificadores que não devem ser usados como features,
    mas que precisam estar presentes até o momento do split.
    """
    def __init__(self, cols_to_drop: Optional[List[str]] = None):
        if cols_to_drop is None:
            self.cols_to_drop = [
                "codigo_aluno", 
                "ano_referencia", 
                "periodo_referencia", 
                "situacao", 
                "target_evasao", 
                "is_target_valid"
            ]
        else:
            self.cols_to_drop = cols_to_drop

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropTechnicalColumns":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(X, "columns"):
            return X
        
        # Interseção entre colunas existentes e as que queremos remover
        existing_cols = [c for c in self.cols_to_drop if c in X.columns]
        if existing_cols:
            return X.drop(columns=existing_cols)
        return X

def get_preprocessor_pipeline(
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
) -> Pipeline:
    """
    Cria o pipeline de pré-processamento com imputação determinística e remoção de colunas técnicas.
    
    Estratégia de Imputação:
    - Numéricos: Mediana (SimpleImputer strategy='median')
    - Categóricos: Constante "missing" (SimpleImputer strategy='constant')
    """
    base_numeric = list(numeric_features) if numeric_features is not None else list(NUMERIC_FEATURES)
    base_categorical = (
        list(categorical_features) if categorical_features is not None else list(CATEGORICAL_FEATURES)
    )
    engineered_numeric = list(ENGINEERED_NUMERIC_FEATURES)
    numeric_for_ensure = base_numeric + engineered_numeric
    numeric_for_model = numeric_for_ensure
    
    # Adicionando passo de remoção de colunas técnicas antes do processamento
    
    # 1. Pipelines Específicos por Tipo de Dado
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("scaler", StandardScaler()),
        ]
    )
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing").set_output(transform="pandas")),
            ("encoder", AdaptiveCategoricalEncoder()),
        ]
    )

    # 2. ColumnTransformer para aplicar pipelines específicos
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_for_model),
            ("cat", categorical_transformer, base_categorical),
        ]
    )

    # 3. Pipeline Base (Limpeza e Feature Engineering Geral)
    base_pipeline = Pipeline(
        steps=[
            ("drop_technical", DropTechnicalColumns()),
            (
                "ensure_columns",
                EnsureColumnsTransformer(
                    numeric_features=numeric_for_ensure,
                    categorical_features=base_categorical,
                ),
            ),
            ("cleaning", DataCleaningTransformer()),
            ("features", FeatureEngineeringTransformer()),
        ]
    )

    # 4. Pipeline Final
    return Pipeline(
        steps=[
            ("base", base_pipeline),
            ("preprocess", column_transformer),
        ]
    )

