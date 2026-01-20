#!/usr/bin/env python3
"""
Script de validação automática para dataset e pipeline de evasão universitária.
Executa checagens objetivas e retorna resultados em JSON.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np


def analyze_dataset(csv_path: str) -> Dict[str, Any]:
    """Analisa dataset e retorna métricas objetivas."""
    
    df = pd.read_csv(csv_path)
    
    # 1. Sumário básico
    shape = df.shape
    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    
    # 2. Valores faltantes
    missing_counts = df.isna().sum()
    missing_dict = {col: int(count) for col, count in missing_counts.items() if count > 0}
    missing_indices = {
        col: df[df[col].isna()].index.tolist()
        for col, count in missing_counts.items() if count > 0
    }
    
    # 3. Colunas totalmente vazias (100% NaN)
    totally_empty = [col for col in df.columns if df[col].isna().all()]
    
    # 4. Coluna target
    target_col = 'situacao' if 'situacao' in df.columns else None
    target_distribution = {}
    if target_col:
        counts = df[target_col].value_counts()
        total = len(df)
        target_distribution = {
            str(k): {
                'count': int(v),
                'percentage': round(100 * v / total, 2)
            }
            for k, v in counts.items()
        }
    
    # 5. Checagem ENEM x VESTIBULAR
    enem_cols = [c for c in df.columns if c.startswith('nota_enem_')]
    vest_cols = [c for c in df.columns if c.startswith('nota_vestibular_')]
    
    enem_mask = df[enem_cols].notna().any(axis=1) if enem_cols else pd.Series([False] * len(df))
    vest_mask = df[vest_cols].notna().any(axis=1) if vest_cols else pd.Series([False] * len(df))
    
    # Considerando zeros como ausência de dados
    if enem_cols:
        enem_nonzero = (df[enem_cols] != 0).any(axis=1)
        enem_mask = enem_mask & enem_nonzero
    
    if vest_cols:
        vest_nonzero = (df[vest_cols] != 0).any(axis=1)
        vest_mask = vest_mask & vest_nonzero
    
    only_enem = enem_mask & ~vest_mask
    only_vest = vest_mask & ~enem_mask
    both = enem_mask & vest_mask
    neither = ~enem_mask & ~vest_mask
    
    enem_vest_check = {
        'enem_columns': enem_cols,
        'vestibular_columns': vest_cols,
        'only_enem': {
            'count': int(only_enem.sum()),
            'indices': only_enem[only_enem].index.tolist()
        },
        'only_vestibular': {
            'count': int(only_vest.sum()),
            'indices': only_vest[only_vest].index.tolist()
        },
        'both': {
            'count': int(both.sum()),
            'indices': both[both].index.tolist()
        },
        'neither': {
            'count': int(neither.sum()),
            'indices': neither[neither].index.tolist()
        }
    }
    
    # 6. Cardinalidade de categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    cardinality = {}
    for col in categorical_cols:
        if col == target_col:
            continue
        counts = df[col].value_counts().head(20)
        cardinality[col] = {
            'unique_count': int(df[col].nunique()),
            'top_20_values': {str(k): int(v) for k, v in counts.items()}
        }
    
    # 7. Features numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return {
        'summary': {
            'shape': {'rows': int(shape[0]), 'columns': int(shape[1])},
            'columns': columns,
            'dtypes': dtypes
        },
        'missing_values': {
            'counts': missing_dict,
            'indices': missing_indices,
            'totally_empty_columns': totally_empty
        },
        'target': {
            'column': target_col,
            'distribution': target_distribution
        },
        'enem_vestibular_check': enem_vest_check,
        'feature_types': {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'cardinality': cardinality
        }
    }


def validate_code_repository(repo_path: str = '.') -> Dict[str, Any]:
    """Valida implementações no repositório de código."""
    
    repo = Path(repo_path)
    results = {
        'preprocessing_pipeline': False,
        'column_transformer': False,
        'numeric_imputer': False,
        'categorical_imputer': False,
        'feature_engineering': False,
        'adaptive_encoding': False,
        'stratified_cv': False,
        'roc_auc_metric': False,
        'class_balancing': False,
        'calibration': False,
        'joblib_save': False,
        'evidence': {}
    }
    
    # Procurar arquivo preprocessing.py
    preproc_file = repo / 'src' / 'preprocessing.py'
    if not preproc_file.exists():
        preproc_file = repo / 'preprocessing.py'
    
    if preproc_file.exists():
        content = preproc_file.read_text()
        
        if 'ColumnTransformer' in content:
            results['column_transformer'] = True
            results['evidence']['column_transformer'] = 'preprocessing.py: ColumnTransformer found'
        
        if 'SimpleImputer' in content and 'median' in content:
            results['numeric_imputer'] = True
            results['evidence']['numeric_imputer'] = 'preprocessing.py: SimpleImputer(strategy="median")'
        
        if 'SimpleImputer' in content and 'constant' in content:
            results['categorical_imputer'] = True
            results['evidence']['categorical_imputer'] = 'preprocessing.py: SimpleImputer(strategy="constant")'
        
        if 'FeatureEngineeringTransformer' in content:
            results['feature_engineering'] = True
            results['evidence']['feature_engineering'] = 'preprocessing.py: FeatureEngineeringTransformer class'
        
        if 'AdaptiveCategoricalEncoder' in content:
            results['adaptive_encoding'] = True
            results['evidence']['adaptive_encoding'] = 'preprocessing.py: AdaptiveCategoricalEncoder class'
    
    # Procurar arquivo trainer.py
    trainer_file = repo / 'src' / 'trainer.py'
    if not trainer_file.exists():
        trainer_file = repo / 'trainer.py'
    
    if trainer_file.exists():
        content = trainer_file.read_text()
        
        if 'StratifiedKFold' in content:
            results['stratified_cv'] = True
            results['evidence']['stratified_cv'] = 'trainer.py: StratifiedKFold found'
        
        if 'roc_auc_score' in content:
            results['roc_auc_metric'] = True
            results['evidence']['roc_auc_metric'] = 'trainer.py: roc_auc_score metric'
        
        if 'class_weight="balanced"' in content or "class_weight='balanced'" in content:
            results['class_balancing'] = True
            results['evidence']['class_balancing'] = 'trainer.py: class_weight="balanced"'
        
        if 'CalibratedClassifierCV' in content:
            results['calibration'] = True
            results['evidence']['calibration'] = 'trainer.py: CalibratedClassifierCV found'
        
        if 'joblib.dump' in content:
            results['joblib_save'] = True
            results['evidence']['joblib_save'] = 'trainer.py: joblib.dump method'
    
    results['preprocessing_pipeline'] = (
        results['column_transformer'] and 
        results['numeric_imputer'] and 
        results['categorical_imputer']
    )
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Uso: python validate_dataset_and_pipeline.py <caminho_csv> [caminho_repo]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    repo_path = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    print("=" * 80)
    print("VALIDAÇÃO DE DATASET E PIPELINE - EVASÃO UNIVERSITÁRIA")
    print("=" * 80)
    
    # Análise do dataset
    print(f"\n[1/2] Analisando dataset: {csv_path}")
    try:
        dataset_results = analyze_dataset(csv_path)
    except Exception as e:
        print(f"ERRO ao analisar dataset: {e}")
        dataset_results = {'error': str(e)}
    
    # Validação do código
    print(f"\n[2/2] Validando repositório: {repo_path}")
    try:
        code_results = validate_code_repository(repo_path)
    except Exception as e:
        print(f"ERRO ao validar código: {e}")
        code_results = {'error': str(e)}
    
    # Resultados finais
    final_results = {
        'dataset_analysis': dataset_results,
        'code_validation': code_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Salvar JSON
    output_file = 'validation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"Resultados salvos em: {output_file}")
    print(f"{'=' * 80}\n")
    
    # Exibir resumo
    print(json.dumps(final_results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()