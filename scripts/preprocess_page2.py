"""
Preprocessing script for Page 2: Geospatial & External Factor Analysis.
Run once before launching the Streamlit app: python scripts/preprocess_page2.py
Reads the raw real estate CSV, merges with embedded ACS 2022 5-year median
household income estimates by state, and saves state-level and city-level
summary parquets to artifacts/page2/.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ACS 2022 5-Year Estimates: Median Household Income by State，Source: U.S. Census Bureau, ACS 5-Year Estimates, Table B19013
STATE_MEDIAN_INCOME = {
    'Alabama': 54943, 'Alaska': 77790, 'Arizona': 72581,
    'Arkansas': 52528, 'California': 84097, 'Colorado': 87598,
    'Connecticut': 83771, 'Delaware': 72724, 'Florida': 63062,
    'Georgia': 65030, 'Hawaii': 88005, 'Idaho': 66688,
    'Illinois': 72563, 'Indiana': 63195, 'Iowa': 65429,
    'Kansas': 64124, 'Kentucky': 55573, 'Louisiana': 53571,
    'Maine': 64767, 'Maryland': 94384, 'Massachusetts': 89645,
    'Michigan': 63202, 'Minnesota': 80441, 'Mississippi': 48716,
    'Missouri': 61847, 'Montana': 60560, 'Nebraska': 68116,
    'Nevada': 66274, 'New Hampshire': 90300, 'New Jersey': 93183,
    'New Mexico': 51945, 'New York': 75157, 'North Carolina': 62891,
    'North Dakota': 68131, 'Ohio': 62689, 'Oklahoma': 56956,
    'Oregon': 73648, 'Pennsylvania': 67587, 'Rhode Island': 74008,
    'South Carolina': 59318, 'South Dakota': 65875, 'Tennessee': 59695,
    'Texas': 67321, 'Utah': 79133, 'Vermont': 72431,
    'Virginia': 80615, 'Washington': 87648, 'West Virginia': 50884,
    'Wisconsin': 67080, 'Wyoming': 68002,
    'District of Columbia': 101027, 'Puerto Rico': 21058,
}
# FIPS codes for Altair choropleth join with us-10m TopoJSON
STATE_FIPS = {
    'Alabama': 1, 'Alaska': 2, 'Arizona': 4, 'Arkansas': 5,
    'California': 6, 'Colorado': 8, 'Connecticut': 9, 'Delaware': 10,
    'District of Columbia': 11, 'Florida': 12, 'Georgia': 13,
    'Hawaii': 15, 'Idaho': 16, 'Illinois': 17, 'Indiana': 18,
    'Iowa': 19, 'Kansas': 20, 'Kentucky': 21, 'Louisiana': 22,
    'Maine': 23, 'Maryland': 24, 'Massachusetts': 25, 'Michigan': 26,
    'Minnesota': 27, 'Mississippi': 28, 'Missouri': 29, 'Montana': 30,
    'Nebraska': 31, 'Nevada': 32, 'New Hampshire': 33, 'New Jersey': 34,
    'New Mexico': 35, 'New York': 36, 'North Carolina': 37, 'North Dakota': 38,
    'Ohio': 39, 'Oklahoma': 40, 'Oregon': 41, 'Pennsylvania': 42,
    'Rhode Island': 44, 'South Carolina': 45, 'South Dakota': 46,
    'Tennessee': 47, 'Texas': 48, 'Utah': 49, 'Vermont': 50,
    'Virginia': 51, 'Washington': 53, 'West Virginia': 54,
    'Wisconsin': 55, 'Wyoming': 56,
}


def run(
    csv_path: str | Path = 'dataset/realtor-data.zip.csv',
    output_dir: str | Path = 'artifacts/page2',
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Loading and filtering...')
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'sold'].copy()
    df = df[df['bed'].isna() | (df['bed'] <= 20)]
    df = df[df['bath'].isna() | (df['bath'] <= 20)]
    price_cap = df['price'].quantile(0.99)
    df = df[(df['price'] > 0) & (df['price'] <= price_cap)].copy()
    print(f'  {len(df):,} sold listings after filtering')
    df['price_per_sqft'] = df['price'] / df['house_size'].replace(0, np.nan)
    print('Computing state summary...')
    state_df = df.groupby('state').agg(
        median_price=('price', 'median'),
        mean_price=('price', 'mean'),
        median_price_per_sqft=('price_per_sqft', 'median'),
        listing_count=('price', 'count'),
    ).reset_index()

    state_df['median_income'] = state_df['state'].map(STATE_MEDIAN_INCOME)
    state_df['fips'] = state_df['state'].map(STATE_FIPS)
    state_df = state_df.dropna(subset=['median_income'])
    state_df['income_price_ratio'] = (
        state_df['median_price'] / state_df['median_income']
    ).round(2)

    state_df.to_parquet(output_dir / 'state_summary.parquet', index=False)
    print(f'  Saved state_summary.parquet  ({len(state_df)} states)')
    print('Computing city summary...')
    city_df = df.groupby(['state', 'city']).agg(
        median_price=('price', 'median'),
        median_price_per_sqft=('price_per_sqft', 'median'),
        listing_count=('price', 'count'),
    ).reset_index()

    city_df['median_income'] = city_df['state'].map(STATE_MEDIAN_INCOME)
    city_df = city_df.dropna(subset=['median_income'])
    city_df['income_price_ratio'] = (
        city_df['median_price'] / city_df['median_income']
    ).round(2)

    city_df.to_parquet(output_dir / 'city_summary.parquet', index=False)
    print(f'  Saved city_summary.parquet   ({len(city_df)} cities)')

    print('Done.')


if __name__ == '__main__':
    run()
