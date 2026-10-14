import json
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

NAIVE_FEATURES = [
    'bed', 'bath', 'log_house_size', 'acre_lot',
    'bed_missing', 'bath_missing', 'house_size_missing', 'acre_lot_missing',
    'years_since_last_sale', 'no_prior_sale',
]
AWARE_FEATURES = NAIVE_FEATURES + ['state_encoded', 'city_encoded', 'zip_code_encoded']


def get_feature_cols(model_type: str) -> list[str]:
    """Return feature column list for 'naive' or 'aware' model."""
    if model_type == 'naive':
        return list(NAIVE_FEATURES)
    if model_type == 'aware':
        return list(AWARE_FEATURES)
    raise ValueError(f"model_type must be 'naive' or 'aware', got {model_type!r}")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor with early stopping evaluated on dev set."""
    model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, n_jobs=-1, verbose=-1)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=callbacks,
    )
    return model


def compute_shap_artifacts(
    model: lgb.LGBMRegressor,
    sample_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    output_dir: Path,
) -> None:
    """Compute SHAP values on sample_df; save parquet (for Altair) and summary PNG."""
    X_sample = sample_df[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)   # shape: (n_samples, n_features)

    shap_df = X_sample.copy()
    shap_df['state'] = sample_df['state'].values
    for i, col in enumerate(feature_cols):
        shap_df[f'shap_{col}'] = shap_values[:, i]
    shap_df.to_parquet(output_dir / f'shap_values_{model_type}.parquet', index=False)

    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f'shap_summary_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()


def compute_metrics(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    y_true: pd.Series,
) -> dict:
    """RMSE, MAE, R² on log scale and original price scale."""
    y_pred_log = model.predict(X)
    y_true_price = np.expm1(y_true)
    y_pred_price = np.expm1(y_pred_log)
    return {
        'rmse_log': float(np.sqrt(mean_squared_error(y_true, y_pred_log))),
        'mae_log': float(mean_absolute_error(y_true, y_pred_log)),
        'r2_log': float(r2_score(y_true, y_pred_log)),
        'rmse_price': float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        'mae_price': float(mean_absolute_error(y_true_price, y_pred_price)),
        'r2_price': float(r2_score(y_true_price, y_pred_price)),
    }


def run(
    data_dir: str | Path = 'artifacts/data',
    output_dir: str | Path = 'artifacts/page1',
    shap_sample_size: int = 5000,
) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    dev_df = pd.read_parquet(data_dir / 'dev.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')

    metrics_out: dict = {}
    for model_type in ['naive', 'aware']:
        feature_cols = get_feature_cols(model_type)
        X_train, y_train = train_df[feature_cols], train_df['log_price']
        X_dev, y_dev = dev_df[feature_cols], dev_df['log_price']
        X_test, y_test = test_df[feature_cols], test_df['log_price']

        print(f'Training {model_type} model...')
        model = train_model(X_train, y_train, X_dev, y_dev)
        model.booster_.save_model(str(output_dir / f'{model_type}_model.lgbm'))

        print(f'Computing SHAP for {model_type} model...')
        sample_df = train_df.sample(shap_sample_size, random_state=42)
        compute_shap_artifacts(model, sample_df, feature_cols, model_type, output_dir)

        print(f'Computing metrics for {model_type} model...')
        metrics_out[model_type] = {
            'dev': compute_metrics(model, X_dev, y_dev),
            'test': compute_metrics(model, X_test, y_test),
        }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print('Done.')


if __name__ == '__main__':
    run()
