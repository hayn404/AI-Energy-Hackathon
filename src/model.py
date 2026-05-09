"""LightGBM load forecaster."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


BASE_PARAMS = dict(
    objective='regression',
    learning_rate=0.01,
    max_depth=10,
    num_leaves=255,
    min_child_samples=25,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.05,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def train(df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          feature_cols: list[str],
          target: str = 'load_p') -> lgb.LGBMRegressor:
    """Train LightGBM with early stopping on val. Returns fitted model."""
    model = lgb.LGBMRegressor(n_estimators=5000, metric='rmse', **BASE_PARAMS)
    model.fit(
        df_train[feature_cols], df_train[target],
        eval_set=[(df_val[feature_cols], df_val[target])],
        callbacks=[lgb.early_stopping(200, verbose=False),
                   lgb.log_evaluation(period=500)],
    )
    return model


def retrain_full(df_full: pd.DataFrame,
                 feature_cols: list[str],
                 best_iteration: int,
                 target: str = 'load_p') -> lgb.LGBMRegressor:
    """Retrain on the full dataset using the best_iteration from early stopping."""
    model = lgb.LGBMRegressor(n_estimators=best_iteration, **BASE_PARAMS)
    model.fit(df_full[feature_cols], df_full[target])
    return model


def predict(model: lgb.LGBMRegressor,
            df: pd.DataFrame,
            feature_cols: list[str]) -> np.ndarray:
    """Return non-negative predictions."""
    preds = model.predict(df[feature_cols])
    return np.clip(preds, 0, None)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = '') -> dict:
    """Compute RMSE, MAE, NRMSE."""
    rmse  = mean_squared_error(y_true, y_pred) ** 0.5
    mae   = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / y_true.mean() * 100
    tag = f"[{label}] " if label else ""
    print(f"  {tag}RMSE={rmse:.4f} kW  MAE={mae:.4f} kW  NRMSE={nrmse:.2f}%")
    return dict(RMSE=rmse, MAE=mae, NRMSE=nrmse)
