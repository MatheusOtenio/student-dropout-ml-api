"""
Testes unitários para validação de dados de evasão universitária.
Verifica exclusividade ENEM/VESTIBULAR e integridade do pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from validate_dataset_and_pipeline import analyze_dataset


@pytest.fixture
def sample_enem_only():
    """Dataset com apenas notas ENEM preenchidas."""
    return pd.DataFrame({
        'nota_enem_humanas': [500.0, 600.0],
        'nota_enem_linguagem': [550.0, 620.0],
        'nota_enem_matematica': [480.0, 590.0],
        'nota_enem_natureza': [510.0, 610.0],
        'nota_enem_redacao': [700.0, 800.0],
        'nota_vestibular_matematica': [0.0, 0.0],
        'nota_vestibular_lingua_portuguesa': [0.0, 0.0],
        'nota_vestibular_biologia': [0.0, 0.0],
        'nota_vestibular_historia': [0.0, 0.0],
        'situacao': ['Desistente', 'Regular']
    })


@pytest.fixture
def sample_vest_only():
    """Dataset com apenas notas VESTIBULAR preenchidas."""
    return pd.DataFrame({
        'nota_enem_humanas': [0.0, 0.0],
        'nota_enem_linguagem': [0.0, 0.0],
        'nota_enem_matematica': [0.0, 0.0],
        'nota_enem_natureza': [0.0, 0.0],
        'nota_enem_redacao': [0.0, 0.0],
        'nota_vestibular_matematica': [8.5, 7.2],
        'nota_vestibular_lingua_portuguesa': [9.0, 8.1],
        'nota_vestibular_biologia': [7.5, 6.8],
        'nota_vestibular_historia': [8.0, 7.5],
        'situacao': ['Regular', 'Desistente']
    })


@pytest.fixture
def sample_mixed_invalid():
    """Dataset INVÁLIDO com ENEM e VESTIBULAR misturados."""
    return pd.DataFrame({
        'nota_enem_humanas': [500.0, 0.0],
        'nota_enem_linguagem': [550.0, 0.0],
        'nota_vestibular_matematica': [8.5, 7.2],
        'nota_vestibular_lingua_portuguesa': [0.0, 8.1],
        'situacao': ['Desistente', 'Regular']
    })


class TestENEMVestibularExclusivity:
    """Testes para garantir exclusividade entre ENEM e VESTIBULAR."""
    
    def test_enem_only_valid(self, sample_enem_only):
        """Verifica que dataset somente ENEM é válido."""
        df = sample_enem_only
        
        enem_cols = [c for c in df.columns if c.startswith('nota_enem_')]
        vest_cols = [c for c in df.columns if c.startswith('nota_vestibular_')]
        
        # Verifica que há notas ENEM não-zero
        has_enem = (df[enem_cols] > 0).any(axis=1)
        # Verifica que notas VEST são todas zero
        has_vest = (df[vest_cols] > 0).any(axis=1)
        
        assert has_enem.all(), "Todas linhas devem ter ENEM > 0"
        assert not has_vest.any(), "Nenhuma linha deve ter VESTIBULAR > 0"
    
    def test_vest_only_valid(self, sample_vest_only):
        """Verifica que dataset somente VESTIBULAR é válido."""
        df = sample_vest_only
        
        enem_cols = [c for c in df.columns if c.startswith('nota_enem_')]
        vest_cols = [c for c in df.columns if c.startswith('nota_vestibular_')]
        
        has_enem = (df[enem_cols] > 0).any(axis=1)
        has_vest = (df[vest_cols] > 0).any(axis=1)
        
        assert not has_enem.any(), "Nenhuma linha deve ter ENEM > 0"
        assert has_vest.all(), "Todas linhas devem ter VESTIBULAR > 0"
    
    def test_mixed_invalid(self, sample_mixed_invalid):
        """Verifica que dataset misto VIOLA a regra de exclusividade."""
        df = sample_mixed_invalid
        
        enem_cols = [c for c in df.columns if c.startswith('nota_enem_')]
        vest_cols = [c for c in df.columns if c.startswith('nota_vestibular_')]
        
        has_enem = (df[enem_cols] > 0).any(axis=1)
        has_vest = (df[vest_cols] > 0).any(axis=1)
        
        # Verifica violação: linhas com AMBOS preenchidos
        both = has_enem & has_vest
        
        # Dataset de exemplo tem violação na linha 0
        assert both.any(), "Dataset deve ter pelo menos uma violação (ENEM + VEST)"
        assert both.iloc[0], "Linha 0 deve ter violação"
    
    def test_exclusivity_check_function(self):
        """Testa função genérica de checagem de exclusividade."""
        
        def check_enem_vest_exclusivity(df: pd.DataFrame) -> dict:
            """Verifica exclusividade ENEM vs VESTIBULAR."""
            enem_cols = [c for c in df.columns if c.startswith('nota_enem_')]
            vest_cols = [c for c in df.columns if c.startswith('nota_vestibular_')]
            
            has_enem = (df[enem_cols] > 0).any(axis=1)
            has_vest = (df[vest_cols] > 0).any(axis=1)
            
            only_enem = has_enem & ~has_vest
            only_vest = has_vest & ~has_enem
            both = has_enem & has_vest
            neither = ~has_enem & ~has_vest
            
            return {
                'only_enem': only_enem.sum(),
                'only_vest': only_vest.sum(),
                'both': both.sum(),  # VIOLAÇÃO se > 0
                'neither': neither.sum(),
                'violations': both[both].index.tolist()
            }
        
        # Teste com dataset válido
        df_valid = pd.DataFrame({
            'nota_enem_humanas': [500.0, 0.0],
            'nota_vestibular_matematica': [0.0, 8.5]
        })
        result = check_enem_vest_exclusivity(df_valid)
        assert result['both'] == 0, "Não deve haver violações"
        assert len(result['violations']) == 0
        
        # Teste com dataset inválido
        df_invalid = pd.DataFrame({
            'nota_enem_humanas': [500.0, 600.0],
            'nota_vestibular_matematica': [8.5, 0.0]
        })
        result = check_enem_vest_exclusivity(df_invalid)
        assert result['both'] == 1, "Deve haver 1 violação"
        assert result['violations'] == [0]


class TestTargetColumnIntegrity:
    """Testes para validar integridade da coluna target."""
    
    def test_target_exists(self, sample_enem_only):
        """Verifica que coluna 'situacao' existe."""
        assert 'situacao' in sample_enem_only.columns
    
    def test_target_no_missing(self, sample_enem_only):
        """Verifica que target não tem valores faltantes."""
        assert not sample_enem_only['situacao'].isna().any()
    
    def test_target_valid_values(self):
        """Verifica que valores do target são válidos."""
        valid_values = {
            'desistente', 'trancado', 'afastado',  # Evasão
            'regular', 'formado', 'transferido'     # Não-evasão
        }
        
        df = pd.DataFrame({
            'situacao': ['Desistente', 'Regular', 'Formado', 'Trancado']
        })
        
        normalized = df['situacao'].str.lower().str.strip()
        assert normalized.isin(valid_values).all()


class TestMissingValuesHandling:
    """Testes para tratamento de valores faltantes."""
    
    def test_critical_column_not_empty(self):
        """Verifica que coluna crítica não está 100% vazia."""
        df = pd.DataFrame({
            'situacao': ['Regular', 'Desistente', 'Formado'],
            'cor_raca': [np.nan, np.nan, np.nan]  # 100% missing
        })
        
        # situacao não deve estar vazia
        assert not df['situacao'].isna().all()
        
        # cor_raca pode estar vazia (warning, não erro)
        if df['cor_raca'].isna().all():
            import warnings
            warnings.warn("Coluna 'cor_raca' está 100% vazia")


class TestDataLeakagePrevention:
    """Testes para prevenir vazamento de dados."""
    
    def test_total_semestres_not_in_features(self):
        """Verifica que 'total_semestres_cursados' NÃO está em features."""
        # Simula as listas de features do código
        NUMERIC_FEATURES = [
            'coeficiente_rendimento', 'disciplinas_aprovadas',
            'periodo', 'ano_ingresso', 'idade'
        ]
        
        CATEGORICAL_FEATURES = [
            'sexo', 'curso', 'campus', 'turno'
        ]
        
        all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        
        # total_semestres_cursados NÃO deve estar nas features
        assert 'total_semestres_cursados' not in all_features


class TestAnalyzeDatasetENEMVestibular:
    def test_enem_vestibular_check_counts(self, tmp_path):
        df = pd.DataFrame({
            'nota_enem_humanas': [500.0, 0.0, 0.0, 600.0],
            'nota_vestibular_matematica': [0.0, 8.5, 0.0, 7.2],
        })
        csv_path = tmp_path / "enem_vest_sample.csv"
        df.to_csv(csv_path, index=False)
        result = analyze_dataset(str(csv_path))
        check = result['enem_vestibular_check']
        assert check['only_enem']['count'] == 1
        assert check['only_vestibular']['count'] == 1
        assert check['both']['count'] == 1
        assert check['neither']['count'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
