import json  # used in run()
import numpy as np  # used in engineer_features()
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split  # used in split_and_encode()
from sklearn.preprocessing import LabelEncoder  # used in engineer_features()

REFERENCE_DATE = pd.Timestamp('2024-01-01')


def load_and_filter(csv_path: str | Path) -> pd.DataFrame:
    """Load raw CSV, preserve original row index as data_id, filter to sold, remove outliers."""
    df = pd.read_csv(csv_path)
    df['data_id'] = df.index

    df = df[df['status'] == 'sold'].copy()
    df = df[df['bed'].isna() | (df['bed'] <= 20)]
    df = df[df['bath'].isna() | (df['bath'] <= 20)]

    price_cap = df['price'].quantile(0.99)
    df = df[df['price'] <= price_cap]

    # Remove states with too few records to support stratified splitting
    state_counts = df['state'].value_counts()
    min_state_size = 10  # Ensures at least 2 members per state after 80/20 split
    valid_states = state_counts[state_counts >= min_state_size].index
    df = df[df['state'].isin(valid_states)].copy()

    return df.reset_index(drop=True)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing indicators and impute with state-grouped median (fallback: global median)."""
    df = df.copy()
    for col in ['bed', 'bath', 'house_size', 'acre_lot']:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        state_medians = df.groupby('state')[col].transform('median')
        global_median = df[col].median()
        df[col] = df[col].fillna(state_medians).fillna(global_median)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Add log transforms, price_per_sqft, temporal features, and state label encoding."""
    df = df.copy()

    df['log_price'] = np.log1p(df['price'])
    df['log_house_size'] = np.log1p(df['house_size'])
    df['price_per_sqft'] = df['price'] / df['house_size'].replace(0, np.nan)

    df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
    df['no_prior_sale'] = df['prev_sold_date'].isna().astype(int)
    df['years_since_last_sale'] = (
        (REFERENCE_DATE - df['prev_sold_date']).dt.days / 365.25
    ).fillna(0.0)

    le = LabelEncoder()
    df['state_encoded'] = le.fit_transform(df['state'])

    return df, le


def split_and_encode(
    df: pd.DataFrame, le: LabelEncoder
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """80/10/10 stratified split by state, then fit target encoding on train only."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['state'], random_state=42
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['state'], random_state=42
    )

    global_mean = float(train_df['log_price'].mean())
    encoders: dict = {
        'global_price_mean': global_mean,
        'state_classes': list(le.classes_),
        'city_target_means': {},
        'zip_code_target_means': {},
    }

    for col in ['city', 'zip_code']:
        means = train_df.groupby(col)['log_price'].mean()
        encoders[f'{col}_target_means'] = {str(k): float(v) for k, v in means.items()}
        for split in [train_df, dev_df, test_df]:
            split[f'{col}_encoded'] = split[col].map(means).fillna(global_mean)

    return train_df, dev_df, test_df, encoders


def run(
    csv_path: str | Path = 'dataset/realtor-data.zip.csv',
    output_dir: str | Path = 'artifacts/data',
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading and filtering...')
    df = load_and_filter(csv_path)
    print(f'  {len(df):,} sold listings after filtering')

    print('Handling missing values...')
    df = handle_missing(df)

    print('Engineering features...')
    df, le = engineer_features(df)

    print('Splitting and encoding...')
    train_df, dev_df, test_df, encoders = split_and_encode(df, le)
    print(f'  train={len(train_df):,}  dev={len(dev_df):,}  test={len(test_df):,}')

    print('Saving...')
    df.to_parquet(output_dir / 'cleaned_data.parquet', index=False)
    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    dev_df.to_parquet(output_dir / 'dev.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)
    with open(output_dir / 'encoders.json', 'w') as f:
        json.dump(encoders, f, indent=2)
    print('Done.')


if __name__ == '__main__':
    run()
