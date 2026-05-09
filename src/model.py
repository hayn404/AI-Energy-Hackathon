"""LightGBM load forecaster."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train(df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          feature_cols: list[str],
          target: str = 'load_p') -> lgb.LGBMRegressor:
    """Train LightGBM on train, early-stop on val. Returns fitted model."""
    params = dict(
        objective='regression',
        metric='rmse',
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        df_train[feature_cols], df_train[target],
        eval_set=[(df_val[feature_cols], df_val[target])],
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
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
