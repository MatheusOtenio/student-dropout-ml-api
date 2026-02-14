from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List
import unicodedata

import gc
import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import logging

from src.preprocessing import get_preprocessor_pipeline, DropTechnicalColumns


logger = logging.getLogger(__name__)


def _normalize_text(text: Any) -> str:
    """Normaliza strings removendo acentos e espaços."""
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def _map_situacao_to_binary(situacao: str) -> Optional[int]:
    """Mapeia situação para 0 (Sucesso) ou 1 (Evasão)."""
    s = _normalize_text(situacao)
    if s in ["formado", "concluido"]:
        return 0
    if s in ["desistente", "trancado", "evadido"]:
        return 1
    return None


CONSERVATIVE_LGBM_PARAMS = {
    "max_depth": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}




def _build_base_model(model_type: str, random_state: int, params: Dict[str, Any]) -> Any:
    if model_type == "lightgbm":
        return LGBMClassifier(
            class_weight="balanced",
            random_state=random_state,
            **params,
        )
    base_params = dict(params)
    max_iter = base_params.pop("max_iter", 5000)
    solver = base_params.pop("solver", "saga")
    return LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight="balanced",
        random_state=random_state,
        **base_params,
    )


# Remover a função _wrap_with_calibration que não é mais usada
# (Ou deixar se for utilitário, mas o prompt pediu para limpar código obsoleto)
# Vou removê-la para atender ao requisito de "código limpo".



def _build_model(config: Dict[str, Any], override_params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Constrói o modelo base (classificador) sem calibração externa.
    A calibração agora é aplicada sobre o pipeline completo.
    """
    model_type = config.get("model_type", "logreg")
    random_state = config.get("random_state", 42)
    base_params = dict(config.get("model_params", {}))
    if override_params:
        base_params.update(override_params)
    
    # Retorna apenas o classificador base (LGBM ou LogReg)
    return _build_base_model(model_type, random_state, base_params)


def _normalize_split_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza a configuração de split para garantir o uso de listas para anos.
    Trata compatibilidade retroativa (val_year -> val_years).
    """
    norm = config.copy()

    if "val_years" not in norm:
        if "val_year" in norm:
            norm["val_years"] = [norm["val_year"]]
        else:
            norm["val_years"] = []
            
    if "test_years" not in norm:
        if "test_year" in norm:
            norm["test_years"] = [norm["test_year"]]
        else:
            norm["test_years"] = []
            
    return norm


def _validate_split_config(config: Dict[str, Any], years_available: List[int]) -> None:
    """
    Valida as regras de negócio para o split temporal configurável.
    """
    if not config:
        return

    norm_config = _normalize_split_config(config)

    train_range = norm_config.get("train_range")
    val_years = norm_config.get("val_years")
    test_years = norm_config.get("test_years")

    if not train_range:
        raise ValueError("Incomplete split_config. Must provide train_range.")
        
    if not val_years:
        raise ValueError("Incomplete split_config. Must provide val_years (or val_year).")

    if not test_years:
        raise ValueError("Incomplete split_config. Must provide test_years (or test_year).")

    train_start, train_end = train_range

    val_years = sorted(val_years)
    test_years = sorted(test_years)
    
    min_val, max_val = val_years[0], val_years[-1]
    min_test, max_test = test_years[0], test_years[-1]

    if not (train_end < min_val):
        raise ValueError(f"Invalid temporal sequence: Train End ({train_end}) must be < First Val Year ({min_val}).")
        
    if not (max_val < min_test):
        raise ValueError(f"Invalid temporal sequence: Last Val Year ({max_val}) must be < First Test Year ({min_test}).")

    train_years_set = set(range(train_start, train_end + 1))
    available_set = set(years_available)
    
    if not train_years_set.intersection(available_set):
        raise ValueError(f"No data found for Train Range {train_range}. Available years: {sorted(list(available_set))}")

    for vy in val_years:
        if vy not in available_set:
            raise ValueError(f"No data found for Validation Year {vy}. Available years: {sorted(list(available_set))}")

    for ty in test_years:
        if ty not in available_set:
            raise ValueError(f"No data found for Test Year {ty}. Available years: {sorted(list(available_set))}")


def _get_time_based_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    random_state: int = 42,
    split_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Realiza o split temporal dos dados considerando Ano E Semestre.
    """
    
    if "ano_ingresso" not in X.columns:
        logger.warning("Coluna 'ano_ingresso' não encontrada. Usando split aleatório.")
        return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    if "semestre_ingresso" in X.columns:
        semestre = pd.to_numeric(X["semestre_ingresso"], errors="coerce").fillna(1)
    else:
        semestre = 1

    time_index = X["ano_ingresso"] + (semestre / 10.0)

    if split_config:
        norm_config = _normalize_split_config(split_config)
        
        train_start, train_end = norm_config["train_range"]
        val_years = norm_config["val_years"]

        mask_train = (X["ano_ingresso"] >= train_start) & (X["ano_ingresso"] <= train_end)
        
        mask_val = X["ano_ingresso"].isin(val_years)
        
        n_train = mask_train.sum()
        n_val = mask_val.sum()
        
        if n_train == 0 or n_val == 0:
             raise ValueError(f"Split Config resulted in empty sets: Train={n_train}, Val={n_val}")

        X_train = X[mask_train].reset_index(drop=True)
        y_train = y[mask_train].reset_index(drop=True)
        X_val = X[mask_val].reset_index(drop=True)
        y_val = y[mask_val].reset_index(drop=True)
        
        return X_train, y_train, X_val, y_val

    unique_times = sorted(time_index.unique())
    
    if len(unique_times) < 2:
        return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        
    cutoff_time = unique_times[-2] if len(unique_times) >= 4 else unique_times[-1]
    
    if len(unique_times) <= 2:
        cutoff_time = unique_times[-1]
        
    mask_val = time_index >= cutoff_time
    mask_train = ~mask_val
    
    n_train = mask_train.sum()
    n_val = mask_val.sum()
    
    if n_train < 50 or n_val < 50:
         return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    logger.info(f"Split Temporal Automático: Cutoff={cutoff_time}, Train={n_train}, Val={n_val}")
    
    X_train = X[mask_train].reset_index(drop=True)
    y_train = y[mask_train].reset_index(drop=True)
    X_val = X[mask_val].reset_index(drop=True)
    y_val = y[mask_val].reset_index(drop=True)
    
    return X_train, y_train, X_val, y_val


def _validate_time_based(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    split_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executa uma validação simples (Hold-out) baseada no tempo.
    Agora recebe um modelo/pipeline completo que lida com o pré-processamento internamente.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    
    X_train, y_train, X_val, y_val = _get_time_based_split(X, y, random_state, split_config)

    unique_train = set(y_train.unique())
    if len(unique_train) < 2:
         raise ValueError(
             f"Conjunto de TREINO inválido: contém apenas a classe {list(unique_train)}. "
             f"Verifique se o intervalo de anos selecionado contém exemplos de evasão (1) e não evasão (0)."
         )
         
    unique_val = set(y_val.unique())
    if len(unique_val) < 2:
         logger.warning(
             f"Conjunto de VALIDAÇÃO contém apenas a classe {list(unique_val)}. "
             "O cálculo de AUC será inválido (NaN)."
         )

    # Clona o modelo completo (pode ser CalibratedClassifierCV(Pipeline(...)) ou apenas Pipeline)
    fold_model = clone(model)
    
    # Treina no conjunto de treino (o pré-processamento interno cuidará do target encoding sem leakage)
    fold_model.fit(X_train, y_train)
    
    # Prediz no conjunto de validação
    # Se for CalibratedClassifierCV, predict_proba já retorna a média calibrada
    proba = fold_model.predict_proba(X_val)[:, 1]
    
    try:
        roc_auc = roc_auc_score(y_val, proba)
    except Exception:
        roc_auc = float("nan")
        
    brier = brier_score_loss(y_val, proba)

    logger.info(
        "Resultado da Validação Temporal: roc_auc=%.4f, brier_score=%.4f",
        roc_auc,
        brier,
    )

    roc_auc_val = float(roc_auc)
    if np.isnan(roc_auc_val):
        roc_auc_val = None

    return {
        "roc_auc": roc_auc_val,
        "brier_score": float(brier),
    }


def save_artifact(artifact: Dict[str, Any], path: str) -> None:
    joblib.dump(artifact, path)


def _get_pipeline_details(pipeline: Any) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    """Extrai nomes das features e importâncias de um pipeline treinado."""
    feature_names = None
    importances = None
    
    # 1. Tenta extrair nomes das features do preprocessor
    try:
        if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
            # Este é o Pipeline de pré-processamento retornado por get_preprocessor_pipeline
            # Ele contém: ("base", base_pipeline) e ("preprocess", column_transformer)
            preprocessor_main = pipeline.named_steps["preprocess"]
            
            # A estratégia mais segura é ir direto no ColumnTransformer final,
            # pois os transformers anteriores (FeatureEngineering) podem não ter get_feature_names_out implementado,
            # o que faria a chamada no pipeline pai falhar.
            if hasattr(preprocessor_main, "named_steps") and "preprocess" in preprocessor_main.named_steps:
                col_transformer = preprocessor_main.named_steps["preprocess"]
                if hasattr(col_transformer, "get_feature_names_out"):
                    feature_names = col_transformer.get_feature_names_out()
            
            # Fallback: Tenta chamar no pipeline principal se a estrutura interna for diferente
            if feature_names is None and hasattr(preprocessor_main, "get_feature_names_out"):
                feature_names = preprocessor_main.get_feature_names_out()

    except Exception as e:
        logger.warning(f"Erro ao extrair nomes das features: {e}")

    # 2. Tenta extrair importâncias do modelo
    try:
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            model = pipeline.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = model.coef_
                # Se for matriz (1, n), pega a primeira linha
                if hasattr(importances, "ndim") and importances.ndim > 1:
                    importances = importances[0]
    except Exception as e:
        logger.warning(f"Erro ao extrair importâncias: {e}")
        
    return feature_names, importances


def _metadata_from_pipeline(
    model_wrapper: Any,
    model_type: str,
    metrics: Dict[str, Any],
    X: pd.DataFrame,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    # Dicionário para agregar importâncias por nome da feature
    feature_importance_map: Dict[str, List[float]] = {}
    
    # 1. Coleta estimadores (folds ou único)
    estimators_to_process = []
    
    if isinstance(model_wrapper, CalibratedClassifierCV):
        if hasattr(model_wrapper, "calibrated_classifiers_") and model_wrapper.calibrated_classifiers_:
            for clf in model_wrapper.calibrated_classifiers_:
                if hasattr(clf, "estimator"):
                    estimators_to_process.append(clf.estimator)
    else:
        estimators_to_process.append(model_wrapper)
        
    # 2. Extrai dados de cada estimador e agrega por nome
    for estimator in estimators_to_process:
        f_names, f_imps = _get_pipeline_details(estimator)
        
        if f_names is not None and f_imps is not None:
            f_names = list(f_names)
            f_imps = np.array(f_imps).flatten()
            
            if len(f_names) == len(f_imps):
                for name, imp in zip(f_names, f_imps):
                    if name not in feature_importance_map:
                        feature_importance_map[name] = []
                    feature_importance_map[name].append(float(imp))
    
    # 3. Consolida Feature Importance (Média)
    final_mapped_importance = []
    for name, imps in feature_importance_map.items():
        final_mapped_importance.append({
            "feature": str(name),
            "importance": float(np.mean(imps))
        })
    
    final_mapped_importance.sort(key=lambda x: abs(x["importance"]), reverse=True)
    
    feature_importance = {}
    if final_mapped_importance:
        feature_importance["mapped"] = final_mapped_importance

    # 4. Reconstrói Feature Map (Original -> Derivadas)
    feature_map = {}
    original_cols = list(X.columns)
    
    for orig_col in original_cols:
         derived_features = []
         for item in final_mapped_importance:
             f_name = item["feature"]
             # Lógica heurística para associar derivada à original
             if f_name == orig_col or \
                f_name.startswith(f"{orig_col}_") or \
                f"__{orig_col}_" in f_name:
                 derived_features.append(f_name)
         
         if derived_features:
             feature_map[orig_col] = derived_features

    # 5. Recupera hiperparâmetros
    params = {}
    if estimators_to_process:
        est = estimators_to_process[0]
        if hasattr(est, "named_steps") and "model" in est.named_steps:
             params = est.named_steps["model"].get_params()

    metadata: Dict[str, Any] = {
        "model_type": model_type,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": params,
        "validation_strategy": "time_based_split",
        "class_mapping": {
            "negative": ["formado"],
            "positive": ["desistente", "trancado"],
        },
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_importance": feature_importance, 
        "feature_map": feature_map,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _objective_lightgbm(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    base_config: Dict[str, Any],
) -> float:
    params = dict(CONSERVATIVE_LGBM_PARAMS)
    params.update({
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
    })
    
    random_state = base_config.get("random_state", 42)
    split_config = base_config.get("split_config")
    
    model = _build_base_model("lightgbm", random_state, params)
    
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    
    X_train, y_train, X_val, y_val = _get_time_based_split(X, y, random_state, split_config)
    
    preprocessor = get_preprocessor_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_val_proc = preprocessor.transform(X_val)
    
    model.fit(
        X_train_proc, 
        y_train,
        eval_set=[(X_val_proc, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    proba = model.predict_proba(X_val_proc)[:, 1]
    
    try:
        score = roc_auc_score(y_val, proba)
    except Exception:
        score = float("nan")
        
    if np.isnan(score):
        return 0.5  
        
    return 1.0 - float(score)


def tune_and_train(
    df: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    if "target_evasao" not in df.columns or "is_target_valid" not in df.columns:
        raise ValueError("Colunas 'target_evasao' e 'is_target_valid' não encontradas. Execute o ETL prospectivo antes.")

    valid_mask = df["is_target_valid"] == True
    y = df.loc[valid_mask, "target_evasao"]
    X = df.loc[valid_mask].copy()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)
    
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Target inválido no Tuning: {set(y.unique())}")
        
    # Garante limpeza de colunas técnicas (pipeline consistency)
    dropper = DropTechnicalColumns()
    X = dropper.transform(X)
        
    logger.info("Iniciando Tuning com %d amostras (Target Prospectivo).", len(X))

    model_type = config.get("model_type", "lightgbm")
    if model_type != "lightgbm":
        return train_model(df, config, optimize=False)
        
    optimize_trials = int(config.get("optimize_trials", 20))
    study_n_trials = n_trials if n_trials is not None else optimize_trials
    
    logger.info("Iniciando otimização com %s trials (Split Temporal).", study_n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective_lightgbm(trial, X, y, config),
        n_trials=study_n_trials,
    )
    best_params = study.best_params
    logger.info("Melhores parâmetros: %s", best_params)
    
    tuned_config = dict(config)
    tuned_config["model_params"] = dict(tuned_config.get("model_params", {}))
    tuned_config["model_params"].update(best_params)
    
    artifact = train_model(df, tuned_config, optimize=False, best_params=best_params)
    return artifact


def train_model(
    df: pd.DataFrame,
    config: Dict[str, Any],
    optimize: bool = False,
    best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logger.info(f"Iniciando treinamento com configuração: {config}")

    if "target_evasao" not in df.columns or "is_target_valid" not in df.columns:
        raise ValueError("Colunas 'target_evasao' e 'is_target_valid' não encontradas. Execute o ETL prospectivo antes.")

    valid_mask = df["is_target_valid"] == True
    
    if valid_mask.sum() == 0:
        raise ValueError("Nenhum dado válido para treino (is_target_valid=True).")

    split_config = config.get("split_config")
    if split_config:
        if "ano_ingresso" not in df.columns:
             raise ValueError("Column 'ano_ingresso' is required for temporal split.")
             
        available_years = sorted(df.loc[valid_mask, "ano_ingresso"].dropna().unique().astype(int).tolist())
        _validate_split_config(split_config, available_years)
        
        norm_config = _normalize_split_config(split_config)
        train_start, train_end = norm_config["train_range"]
        val_years = norm_config["val_years"]
        test_years = norm_config["test_years"]

        split_config = norm_config

        allowed_years = list(range(train_start, train_end + 1)) + val_years + test_years
        
        year_mask = df["ano_ingresso"].isin(allowed_years)
        final_mask = valid_mask & year_mask
        
        logger.info(
            "Split Config Filter: Keeping years %s. Rows: %d -> %d",
            allowed_years, valid_mask.sum(), final_mask.sum()
        )
        
        if final_mask.sum() == 0:
             raise ValueError("Dataset empty after split configuration filtering.")
             
        valid_mask = final_mask
        
    X = df.loc[valid_mask].copy()
    y = df.loc[valid_mask, "target_evasao"].astype(int)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    unique_targets = set(y.unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target inválido: {unique_targets}")

    df_filtered = X.copy()

    if optimize:
        return tune_and_train(df_filtered, config)

    preprocessor = get_preprocessor_pipeline()
    base_model = _build_model(config, override_params=best_params)

    # Cria o pipeline interno: Pré-processamento -> Modelo Base
    inner_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )
    
    # Aplica Calibração sobre o Pipeline Completo (Correção de Leakage)
    calibrate = config.get("calibrate", True)
    if calibrate:
        method = config.get("calibration_method", "isotonic")
        cv = int(config.get("calibration_cv", 5))
        logger.info(f"Aplicando Calibração (Pipeline-Level): method={method}, cv={cv}")
        
        final_model = CalibratedClassifierCV(
            estimator=inner_pipeline,
            cv=cv,
            method=method,
        )
    else:
        logger.warning("Calibração desativada.")
        final_model = inner_pipeline

    random_state = config.get("random_state", 42)

    # Validação (agora usa o final_model que encapsula tudo)
    metrics = _validate_time_based(
        model=final_model,
        X=X,
        y=y,
        random_state=random_state,
        split_config=split_config,
    )

    # Treinamento Final (fit em tudo)
    final_model.fit(X, y)

    model_type = config.get("model_type", "logreg")
    extra_meta: Dict[str, Any] = {}
    if best_params is not None:
        extra_meta["best_params"] = best_params
    
    if split_config:
        extra_meta["split_config"] = split_config
        
    metadata = _metadata_from_pipeline(
        model_wrapper=final_model,
        model_type=model_type,
        metrics=metrics,
        X=X,
        extra=extra_meta,
    )

    feature_importance = metadata.get("feature_importance", {})
    if "mapped" in feature_importance and len(feature_importance["mapped"]) > 0:
        top_features = feature_importance["mapped"][0:5]
        top_features_str = ", ".join([f"{f['feature']} ({f['importance']:.4f})" for f in top_features])
        logger.info(f"Top 5 Features by Importance: {top_features_str}")
    elif "raw_values" in feature_importance:
        logger.warning("Feature importance is available but not mapped to feature names; cannot log top features")
    else:
        logger.warning("Feature importance not available for logging")

    del X
    del y
    gc.collect()

    artifact = {
        "model": final_model, # O modelo agora é auto-contido (CalibratedClassifierCV ou Pipeline)
        "metadata": metadata,
        "version": config.get("version", "1.0.0"),
    }

    artifact_path = config.get("artifact_path")
    if artifact_path:
        save_artifact(artifact, artifact_path)

    return artifact
