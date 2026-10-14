"""Compute residuals using the pre-trained confounding-aware model and save to artifacts/page3/."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

AWARE_FEATURES = [
    'bed', 'bath', 'log_house_size', 'acre_lot',
    'bed_missing', 'bath_missing', 'house_size_missing', 'acre_lot_missing',
    'years_since_last_sale', 'no_prior_sale',
    'state_encoded', 'city_encoded', 'zip_code_encoded',
]


def run(
    model_path: str | Path = 'artifacts/page1/aware_model.lgbm',
    data_path: str | Path = 'artifacts/data/test.parquet',
    output_dir: str | Path = 'artifacts/page3',
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading model and data...')
    booster = lgb.Booster(model_file=str(model_path))
    df = pd.read_parquet(data_path)

    print('Computing residuals...')
    pred_log = booster.predict(df[AWARE_FEATURES])
    df = df.copy()
    df['predicted_price'] = np.expm1(pred_log)
    df['residual'] = df['price'] - df['predicted_price']
    df['zip_code'] = df['zip_code'].astype(str).str.split('.').str[0]

    print(f'Saving to {output_dir / "residuals.parquet"}...')
    df.to_parquet(output_dir / 'residuals.parquet', index=False)
    print(f'Done. {len(df):,} rows with residuals.')


if __name__ == '__main__':
    run()
