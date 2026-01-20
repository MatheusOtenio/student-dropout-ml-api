import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing import (
    EnsureColumnsTransformer,
    DataCleaningTransformer,
    FeatureEngineeringTransformer,
    AdaptiveCategoricalEncoder,
)


class TestEnsureColumnsTransformer:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        t = EnsureColumnsTransformer(
            numeric_features=["a", "missing_num"],
            categorical_features=["missing_cat"],
        )
        result = t.fit_transform(df)
        assert "missing_num" in result.columns
        assert "missing_cat" in result.columns
        assert result["missing_num"].isna().all()
        assert result["missing_cat"].isna().all()

    def test_converts_string_numeric_to_float(self):
        df = pd.DataFrame({"a": ["1", "2.5", "x"]})
        t = EnsureColumnsTransformer(numeric_features=["a"])
        result = t.fit_transform(df)
        assert result["a"].dtype == float
        assert result["a"].iloc[0] == pytest.approx(1.0)
        assert result["a"].iloc[1] == pytest.approx(2.5)
        assert np.isnan(result["a"].iloc[2])

    def test_preserves_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
        t = EnsureColumnsTransformer(numeric_features=["a", "b"])
        result = t.fit_transform(df)
        assert list(result.index) == [10, 20]
        assert result.loc[10, "a"] == 1
        assert result.loc[20, "a"] == 2


class TestDataCleaningTransformer:
    def test_removes_whitespace_and_normalizes(self):
        df = pd.DataFrame(
            {
                "col": ["  João  ", "  Árvore  ", " TESTE "],
            }
        )
        t = DataCleaningTransformer()
        result = t.fit_transform(df.copy())
        assert list(result["col"]) == ["joao", "arvore", "teste"]

    def test_converts_placeholders_to_nan(self):
        df = pd.DataFrame(
            {
                "col": ["na", "NA", " n/a ", "-", ""],
            }
        )
        t = DataCleaningTransformer()
        result = t.fit_transform(df.copy())
        assert result["col"].isna().all()

    def test_respects_string_columns_parameter(self):
        df = pd.DataFrame(
            {
                "text": ["  João  "],
                "other": ["  Valor  "],
            }
        )
        t = DataCleaningTransformer(string_columns=["text"])
        result = t.fit_transform(df.copy())
        assert result["text"].iloc[0] == "joao"
        assert result["other"].iloc[0] == "  Valor  "


class TestFeatureEngineeringTransformer:
    def test_creates_enem_total_correctly(self):
        df = pd.DataFrame(
            {
                "nota_enem_humanas": [600, np.nan],
                "nota_enem_linguagem": [650, 650],
                "nota_enem_matematica": [700, np.nan],
                "nota_enem_natureza": [620, 620],
                "nota_enem_redacao": [720, 720],
                "disciplinas_aprovadas": [0, 0],
                "disciplinas_reprovadas_nota": [0, 0],
                "disciplinas_reprovadas_frequencia": [0, 0],
            }
        )
        t = FeatureEngineeringTransformer()
        result = t.fit_transform(df.copy())
        assert "nota_enem_total" in result.columns
        assert result["nota_enem_total"].iloc[0] == pytest.approx(600 + 650 + 700 + 620 + 720)
        expected_second = 0 + 650 + 0 + 620 + 720
        assert result["nota_enem_total"].iloc[1] == pytest.approx(expected_second)

    def test_enem_total_all_nan_becomes_nan(self):
        df = pd.DataFrame(
            {
                "nota_enem_humanas": [np.nan],
                "nota_enem_linguagem": [np.nan],
                "nota_enem_matematica": [np.nan],
                "nota_enem_natureza": [np.nan],
                "nota_enem_redacao": [np.nan],
                "disciplinas_aprovadas": [0],
                "disciplinas_reprovadas_nota": [0],
                "disciplinas_reprovadas_frequencia": [0],
            }
        )
        t = FeatureEngineeringTransformer()
        result = t.fit_transform(df.copy())
        assert np.isnan(result["nota_enem_total"].iloc[0])

    def test_creates_vestibular_total_correctly(self):
        df = pd.DataFrame(
            {
                "nota_vestibular_matematica": [8.5, np.nan],
                "nota_vestibular_lingua_portuguesa": [9.0, 8.1],
                "nota_vestibular_biologia": [7.5, 6.8],
                "disciplinas_aprovadas": [0, 0],
                "disciplinas_reprovadas_nota": [0, 0],
                "disciplinas_reprovadas_frequencia": [0, 0],
            }
        )
        t = FeatureEngineeringTransformer()
        result = t.fit_transform(df.copy())
        assert "nota_vestibular_total" in result.columns
        assert result["nota_vestibular_total"].iloc[0] == pytest.approx(8.5 + 9.0 + 7.5)
        expected_second = 0 + 8.1 + 6.8
        assert result["nota_vestibular_total"].iloc[1] == pytest.approx(expected_second)

    def test_vestibular_total_all_nan_becomes_nan(self):
        df = pd.DataFrame(
            {
                "nota_vestibular_matematica": [np.nan],
                "nota_vestibular_lingua_portuguesa": [np.nan],
                "nota_vestibular_biologia": [np.nan],
                "disciplinas_aprovadas": [0],
                "disciplinas_reprovadas_nota": [0],
                "disciplinas_reprovadas_frequencia": [0],
            }
        )
        t = FeatureEngineeringTransformer()
        result = t.fit_transform(df.copy())
        assert np.isnan(result["nota_vestibular_total"].iloc[0])

    def test_aprovacao_ratio_basic(self):
        df = pd.DataFrame(
            {
                "disciplinas_aprovadas": [8],
                "disciplinas_reprovadas_nota": [2],
                "disciplinas_reprovadas_frequencia": [0],
            }
        )
        t = FeatureEngineeringTransformer()
        result = t.fit_transform(df.copy())
        assert "aprovacao_ratio" in result.columns
        assert result["aprovacao_ratio"].iloc[0] == pytest.approx(8 / (8 + 2 + 0))

    def test_aprovacao_ratio_zero_total_yields_nan(self):
        df = pd.DataFrame(
            {
                "disciplinas_aprovadas": [0],
                "disciplinas_reprovadas_nota": [0],
                "disciplinas_reprovadas_frequencia": [0],
            }
        )
        t = FeatureEngineeringTransformer()
        result = t.fit_transform(df.copy())
        assert np.isnan(result["aprovacao_ratio"].iloc[0])

    def test_aprovacao_ratio_no_inf_values(self):
        df = pd.DataFrame(
            {
                "disciplinas_aprovadas": [8, 0],
                "disciplinas_reprovadas_nota": [2, 0],
                "disciplinas_reprovadas_frequencia": [0, 0],
            }
        )
        t = FeatureEngineeringTransformer()
        result = t.fit_transform(df.copy())
        ratio = result["aprovacao_ratio"]
        assert ratio.iloc[0] == pytest.approx(8 / (8 + 2 + 0))
        assert np.isnan(ratio.iloc[1])
        assert not np.isinf(ratio).any()


class TestAdaptiveCategoricalEncoder:
    def test_onehot_for_low_cardinality(self):
        df = pd.DataFrame({"col": ["a", "b", "a"]})
        y = pd.Series([0, 1, 0])
        enc = AdaptiveCategoricalEncoder(max_onehot_cardinality=20)
        enc.fit(df, y)
        assert enc.low_card_cols_ == ["col"]
        assert enc.high_card_cols_ == []
        arr = enc.transform(df)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == 3
        assert arr.shape[1] == 2

    def test_target_encoding_for_high_cardinality(self):
        values = [f"cat_{i}" for i in range(30)]
        df = pd.DataFrame({"col": values})
        y = pd.Series([0, 1] * 15)
        enc = AdaptiveCategoricalEncoder(max_onehot_cardinality=5)
        enc.fit(df, y)
        assert enc.low_card_cols_ == []
        assert enc.high_card_cols_ == ["col"]
        arr = enc.transform(df)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (30, 1)

    def test_transform_without_y_uses_trained_encoder(self):
        values = [f"cat_{i}" for i in range(30)]
        df_train = pd.DataFrame({"col": values})
        y = pd.Series([0, 1] * 15)
        enc = AdaptiveCategoricalEncoder(max_onehot_cardinality=5)
        enc.fit(df_train, y)
        df_test = pd.DataFrame({"col": ["cat_0", "cat_1", "cat_29"]})
        arr = enc.transform(df_test)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 1)
