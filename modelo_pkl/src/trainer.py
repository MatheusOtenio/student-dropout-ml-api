from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import gc
import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import logging

from src.preprocessing import get_preprocessor_pipeline


logger = logging.getLogger(__name__)


def _normalize_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text


def _map_situacao_to_binary(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    normalized = series.map(_normalize_text)
    positive = {"desistente", "trancado", "afastado"}
    negative = {"regular", "formado", "transferido"}

    def mapper(v: Any) -> float:
        if v in positive:
            return 1.0
        if v in negative:
            return 0.0
        return np.nan

    mapped = normalized.map(mapper)
    valid_mask = mapped.notna()
    if valid_mask.sum() == 0:
        raise ValueError("No valid situacao values found for mapping.")
    return mapped[valid_mask].astype(int), valid_mask


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


def _wrap_with_calibration(model: Any, config: Dict[str, Any]) -> Any:
    calibrate = config.get("calibrate", False)
    if not calibrate:
        return model
    return CalibratedClassifierCV(
        estimator=model,
        cv=config.get("calibration_cv", 5),
        method=config.get("calibration_method", "isotonic"),
    )


def _build_model(config: Dict[str, Any], override_params: Optional[Dict[str, Any]] = None) -> Any:
    model_type = config.get("model_type", "logreg")
    random_state = config.get("random_state", 42)
    base_params = dict(config.get("model_params", {}))
    if override_params:
        base_params.update(override_params)
    base_model = _build_base_model(model_type, random_state, base_params)
    return _wrap_with_calibration(base_model, config)


def _cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int,
) -> Dict[str, Any]:
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    min_samples = int(y.value_counts().min())
    requested_splits = int(n_splits)
    actual_splits = min(requested_splits, min_samples)
    if actual_splits < 2:
        logger.warning(
            "Not enough samples per class (%s) for cross-validation; "
            "training on full data without CV.",
            min_samples,
        )
        pipeline.fit(X, y)
        return {
            "roc_auc": 0.5,
            "brier_score": 0.25,
        }
    skf = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_probs = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)

        fold_preprocessor = get_preprocessor_pipeline()
        X_train_proc = fold_preprocessor.fit_transform(X_train, y_train)
        X_val_proc = fold_preprocessor.transform(X_val)

        if isinstance(pipeline.named_steps["model"], CalibratedClassifierCV):
            base_model = pipeline.named_steps["model"].estimator
        else:
            base_model = pipeline.named_steps["model"]

        fold_model = clone(base_model)
        fold_model.fit(X_train_proc, y_train)
        proba = fold_model.predict_proba(X_val_proc)[:, 1]
        oof_probs[val_idx] = proba

    roc_auc = roc_auc_score(y, oof_probs)
    brier = brier_score_loss(y, oof_probs)

    return {
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
    }


def save_artifact(artifact: Dict[str, Any], path: str) -> None:
    joblib.dump(artifact, path)


def _metadata_from_pipeline(
    pipeline: Pipeline,
    model_type: str,
    metrics: Dict[str, Any],
    X: pd.DataFrame,
    n_splits: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = pipeline.named_steps["model"].get_params()
    metadata: Dict[str, Any] = {
        "model_type": model_type,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": params,
        "cv_splits": n_splits,
        "class_mapping": {
            "negative": ["regular", "formado", "transferido"],
            "positive": ["desistente", "trancado", "afastado"],
        },
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
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
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    model = _build_base_model("lightgbm", base_config.get("random_state", 42), params)
    preprocessor = get_preprocessor_pipeline()
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    min_samples = int(y.value_counts().min())
    requested_splits = int(base_config.get("cv_splits", 5))
    n_splits = min(requested_splits, min_samples)
    random_state = base_config.get("random_state", 42)
    if n_splits < 2:
        logger.warning(
            "Not enough samples per class (%s) for LightGBM tuning CV; "
            "using single fit with default score.",
            min_samples,
        )
        pipeline.fit(X, y)
        return 0.5
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_probs = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in skf.split(X, y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        proba = fold_pipeline.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = proba
    score = roc_auc_score(y, oof_probs)
    return 1.0 - float(score)


def tune_and_train(
    df: pd.DataFrame,
    config: Dict[str, Any],
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    if "situacao" not in df.columns:
        raise ValueError("Column 'situacao' is required for training.")
    y, valid_mask = _map_situacao_to_binary(df["situacao"])
    X = df.loc[valid_mask].drop(columns=["situacao"])
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)
    model_type = config.get("model_type", "lightgbm")
    if model_type != "lightgbm":
        return train_model(df, config, optimize=False)
    optimize_trials = int(config.get("optimize_trials", 20))
    study_n_trials = n_trials if n_trials is not None else optimize_trials
    logger.info("Iniciando otimização de hiperparâmetros com %s trials", study_n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective_lightgbm(trial, X, y, config),
        n_trials=study_n_trials,
    )
    best_params = study.best_params
    logger.info("Otimização de hiperparâmetros concluída. Melhor conjunto de parâmetros: %s", best_params)
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
    if "situacao" not in df.columns:
        raise ValueError("Column 'situacao' is required for training.")

    y, valid_mask = _map_situacao_to_binary(df["situacao"])
    X = df.loc[valid_mask].drop(columns=["situacao"])
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)

    if optimize:
        return tune_and_train(df, config)

    preprocessor = get_preprocessor_pipeline()
    model = _build_model(config, override_params=best_params)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    n_splits = config.get("cv_splits", 5)
    random_state = config.get("random_state", 42)

    metrics = _cross_validate(
        pipeline=pipeline,
        X=X,
        y=y,
        n_splits=n_splits,
        random_state=random_state,
    )

    pipeline.fit(X, y)

    model_type = config.get("model_type", "logreg")
    extra_meta: Dict[str, Any] = {}
    if best_params is not None:
        extra_meta["best_params"] = best_params
    metadata = _metadata_from_pipeline(
        pipeline=pipeline,
        model_type=model_type,
        metrics=metrics,
        X=X,
        n_splits=n_splits,
        extra={
            "class_mapping": {
                "negative": ["regular", "formado", "transferido"],
                "positive": ["desistente", "trancado", "afastado"],
            },
            **extra_meta,
        },
    )

    del X
    del y
    gc.collect()

    artifact = {
        "preprocessor": preprocessor,
        "model": pipeline,
        "metadata": metadata,
        "version": config.get("version", "1.0.0"),
    }

    artifact_path = config.get("artifact_path")
    if artifact_path:
        save_artifact(artifact, artifact_path)

    return artifact
