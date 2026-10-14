import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Union

alt.data_transformers.disable_max_rows()

UNDERPRICED_THRESHOLD = -0.10
OVERPRICED_THRESHOLD = 0.10
SCATTER_SAMPLE_SIZE = 2500
TOP_OPPORTUNITY_ROWS = 15
TOP_VOLATILE_ROWS = 20
RESIDUAL_CATEGORY_ORDER = ['underpriced', 'fair value', 'overpriced']
RESIDUAL_CATEGORY_COLORS = ['#2a9d8f', '#adb5bd', '#e76f51']


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet('artifacts/page3/residuals.parquet').copy()

    numeric_cols = ['price', 'predicted_price', 'residual', 'house_size', 'bed', 'bath']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['state', 'city']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    df['zip_code'] = (
        df['zip_code']
        .fillna('Unknown')
        .astype(str)
        .str.split('.')
        .str[0]
        .replace({'nan': 'Unknown', 'None': 'Unknown', '<NA>': 'Unknown'})
    )

    valid_predicted = df['predicted_price'].where(df['predicted_price'] > 0)
    df['residual_pct'] = (df['price'] - df['predicted_price']) / valid_predicted
    df['residual_pct'] = df['residual_pct'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['price', 'predicted_price', 'residual', 'residual_pct'])

    df['residual_category'] = pd.Categorical(
        np.select(
            [
                df['residual_pct'] <= UNDERPRICED_THRESHOLD,
                df['residual_pct'] >= OVERPRICED_THRESHOLD,
            ],
            ['underpriced', 'overpriced'],
            default='fair value',
        ),
        categories=RESIDUAL_CATEGORY_ORDER,
        ordered=True,
    )
    return df


@st.cache_data(show_spinner=False)
def get_filter_ranges(df: pd.DataFrame) -> Dict[str, Union[int, Tuple[int, int]]]:
    house_sizes = df['house_size'].dropna()
    prices = df['price'].dropna()
    zip_listing_max = int(df.groupby('zip_code').size().max())

    return {
        'house_size': (int(house_sizes.min()), int(house_sizes.max())),
        'price': (int(prices.min()), int(prices.max())),
        'zip_listing_max': max(zip_listing_max, 1),
    }


@st.cache_data(show_spinner=False)
def filter_properties(
    df: pd.DataFrame,
    selected_state: str,
    selected_city: str,
    selected_zip: str,
    house_size_range: Tuple[int, int],
    price_range: Tuple[int, int],
    residual_categories: Tuple[str, ...],
) -> pd.DataFrame:
    filtered = df

    if selected_state != 'All':
        filtered = filtered[filtered['state'] == selected_state]
    if selected_city != 'All':
        filtered = filtered[filtered['city'] == selected_city]
    if selected_zip != 'All':
        filtered = filtered[filtered['zip_code'] == selected_zip]

    filtered = filtered[filtered['price'].between(price_range[0], price_range[1])]
    filtered = filtered[
        filtered['house_size'].fillna(-1).between(house_size_range[0], house_size_range[1])
    ]

    if residual_categories:
        filtered = filtered[filtered['residual_category'].isin(residual_categories)]

    return filtered.copy()


@st.cache_data(show_spinner=False)
def build_zip_summary(df: pd.DataFrame, min_zip_listings: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                'zip_code', 'state', 'city', 'zip_count', 'zip_mean_price',
                'zip_std_price', 'cv', 'mean_residual_pct', 'underpriced_share',
            ]
        )

    zip_summary = (
        df.groupby('zip_code')
        .agg(
            state=('state', 'first'),
            city=('city', 'first'),
            zip_count=('price', 'size'),
            zip_mean_price=('price', 'mean'),
            zip_std_price=('price', 'std'),
            mean_residual_pct=('residual_pct', 'mean'),
            underpriced_share=('residual_category', lambda x: (x == 'underpriced').mean()),
        )
        .reset_index()
    )

    zip_summary['zip_std_price'] = zip_summary['zip_std_price'].fillna(0.0)
    zip_summary['cv'] = zip_summary['zip_std_price'] / zip_summary['zip_mean_price'].replace(0, np.nan)
    zip_summary['cv'] = zip_summary['cv'].fillna(0.0)

    return zip_summary[zip_summary['zip_count'] >= min_zip_listings].copy()


@st.cache_data(show_spinner=False)
def build_kpi_summary(df: pd.DataFrame, zip_summary: pd.DataFrame) -> Dict[str, Union[str, int, float]]:
    underpriced_count = int((df['residual_category'] == 'underpriced').sum())
    zip_count = int(df['zip_code'].nunique())
    median_residual_pct = float(df['residual_pct'].median())
    underpriced_share = float((df['residual_category'] == 'underpriced').mean())

    if zip_summary.empty:
        best_zip = 'N/A'
        best_zip_delta = 'No ZIP meets listing threshold'
    else:
        best_zip_row = zip_summary.sort_values(
            ['mean_residual_pct', 'cv', 'zip_count'],
            ascending=[True, True, False],
        ).iloc[0]
        best_zip = str(best_zip_row['zip_code'])
        best_zip_delta = f"{best_zip_row['mean_residual_pct']:.1%} avg. residual"

    return {
        'property_count': len(df),
        'zip_count': zip_count,
        'underpriced_count': underpriced_count,
        'underpriced_share': underpriced_share,
        'median_residual_pct': median_residual_pct,
        'best_zip': best_zip,
        'best_zip_delta': best_zip_delta,
    }
st.title('Investment Opportunity & Anomaly Detection')
st.markdown(
    'This page turns model residuals into a screening workflow. Residuals compare each '
    'sold price to the confounding-aware fair-value estimate from Page 1, so negative '
    'residual percentages may indicate homes that sold below model expectations.'
)

df_with_residual = load_data()
filter_ranges = get_filter_ranges(df_with_residual)

st.subheader('Shared Filters')
st.caption(
    'Start with state, then narrow to city and ZIP. Every chart and table below updates '
    'from the same filter state.'
)

states = ['All'] + sorted(df_with_residual['state'].unique().tolist())
state_col, city_col, zip_col = st.columns(3)

with state_col:
    selected_state = st.selectbox('State', states)

if selected_state == 'All':
    state_filtered = df_with_residual
    available_cities = ['All']
    available_zips = ['All']
else:
    state_filtered = df_with_residual[df_with_residual['state'] == selected_state]
    available_cities = ['All'] + sorted(state_filtered['city'].unique().tolist())

with city_col:
    selected_city = st.selectbox('City', available_cities)

if selected_state == 'All':
    city_filtered = state_filtered
    available_zips = ['All']
elif selected_city == 'All':
    city_filtered = state_filtered
    available_zips = ['All'] + sorted(state_filtered['zip_code'].unique().tolist())
else:
    city_filtered = state_filtered[state_filtered['city'] == selected_city]
    available_zips = ['All'] + sorted(city_filtered['zip_code'].unique().tolist())

with zip_col:
    selected_zip = st.selectbox('ZIP Code', available_zips)

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    price_range = st.slider(
        'Price Range ($)',
        min_value=filter_ranges['price'][0],
        max_value=filter_ranges['price'][1],
        value=filter_ranges['price'],
        step=10_000,
    )
with filter_col2:
    house_size_range = st.slider(
        'House Size Range (sqft)',
        min_value=filter_ranges['house_size'][0],
        max_value=filter_ranges['house_size'][1],
        value=filter_ranges['house_size'],
        step=100,
    )
with filter_col3:
    residual_categories = st.multiselect(
        'Residual Category',
        RESIDUAL_CATEGORY_ORDER,
        default=RESIDUAL_CATEGORY_ORDER,
    )

listing_min, _, _ = st.columns([1, 0.08, 1.92])
with listing_min:
    min_zip_listings = st.slider(
        'Minimum ZIP Listings',
        min_value=1,
        max_value=int(filter_ranges['zip_listing_max']),
        value=min(10, int(filter_ranges['zip_listing_max'])),
    )

if not residual_categories:
    st.info('Select at least one residual category to populate the dashboard.')
    st.stop()

filtered_df = filter_properties(
    df_with_residual,
    selected_state,
    selected_city,
    selected_zip,
    house_size_range,
    price_range,
    tuple(residual_categories),
)

if filtered_df.empty:
    st.info('No properties match the current filters. Try widening the ranges or selecting fewer filters.')
    st.stop()

zip_summary = build_zip_summary(filtered_df, min_zip_listings)
kpis = build_kpi_summary(filtered_df, zip_summary)

st.subheader('Opportunity Snapshot')
st.markdown(
    'Use residual percentage as the main valuation signal: values below `-10%` are treated '
    'as underpriced, values between `-10%` and `+10%` are fair-value territory, and values '
    'above `+10%` are treated as overpriced.'
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric(
        'Filtered Sales',
        f"{kpis['property_count']:,}",
        f"{kpis['zip_count']:,} ZIPs",
        delta_color='off',
    )
with kpi_col2:
    st.metric(
        'Underpriced Share',
        f"{kpis['underpriced_share']:.1%}",
        f"{kpis['underpriced_count']:,} homes",
        delta_color='off',
    )
with kpi_col3:
    st.metric(
        'Median Residual %',
        f"{kpis['median_residual_pct']:.1%}",
        'vs fair value',
        delta_color='off',
    )
with kpi_col4:
    st.metric(
        'Best Opportunity ZIP',
        kpis['best_zip'],
        str(kpis['best_zip_delta']),
        delta_color='off',
    )

st.markdown('**Residual Distribution**')

residual_min = float(filtered_df['residual_pct'].min())
residual_max = float(filtered_df['residual_pct'].max())
default_lo = max(residual_min, -1.0)
default_hi = min(residual_max, 1.0)
x_range = st.slider(
    'X-axis range (Residual %)',
    min_value=residual_min,
    max_value=residual_max,
    value=(default_lo, default_hi),
    format='%.0f%%',
    help='Adjust to zoom in on the main distribution or zoom out to see outliers',
)

hist_df = filtered_df[
    filtered_df['residual_pct'].between(x_range[0], x_range[1])
]

histogram = (
    alt.Chart(hist_df)
    .mark_bar(opacity=0.85)
    .encode(
        x=alt.X(
            'residual_pct:Q',
            bin=alt.Bin(maxbins=60),
            title='Residual % vs Predicted Fair Value',
            axis=alt.Axis(format='.0%'),
            scale=alt.Scale(domain=[x_range[0], x_range[1]]),
        ),
        y=alt.Y('count():Q', title='Property Count'),
        color=alt.Color(
            'residual_category:N',
            sort=RESIDUAL_CATEGORY_ORDER,
            title='Category',
            scale=alt.Scale(domain=RESIDUAL_CATEGORY_ORDER, range=RESIDUAL_CATEGORY_COLORS),
        ),
    )
)
threshold_rules = alt.Chart(
    pd.DataFrame(
        {
            'threshold': [UNDERPRICED_THRESHOLD, OVERPRICED_THRESHOLD],
            'label': ['Underpriced cutoff', 'Overpriced cutoff'],
        }
    )
).mark_rule(color='#495057', strokeDash=[6, 4]).encode(
    x='threshold:Q',
    tooltip=['label:N', alt.Tooltip('threshold:Q', format='.0%')],
)
st.altair_chart(
    (histogram + threshold_rules).properties(height=320),
    use_container_width=True,
)
st.caption(
    'Negative residual percentages suggest sold prices below the model estimate. '
    'The dashed lines mark the `-10%` and `+10%` screening thresholds. '
    'Use the slider above to zoom in/out on the x-axis.'
)

st.divider()

st.subheader('Property Screener')
st.markdown(
    'Scan how individual properties compare to their predicted fair value, then review the '
    'most discounted candidates within the current filter set.'
)

if len(filtered_df) > SCATTER_SAMPLE_SIZE:
    scatter_df = filtered_df.sample(SCATTER_SAMPLE_SIZE, random_state=42)
else:
    scatter_df = filtered_df.copy()

line_min = float(scatter_df['predicted_price'].min())
line_max = float(scatter_df['predicted_price'].max())
if line_min == line_max:
    line_min = max(0.0, line_min - 1.0)
    line_max = line_max + 1.0

property_scatter = (
    alt.Chart(scatter_df)
    .mark_circle(opacity=0.45, size=55)
    .encode(
        x=alt.X('predicted_price:Q', title='Predicted Fair Price ($)', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('price:Q', title='Actual Sold Price ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color(
            'residual_pct:Q',
            title='Residual %',
            scale=alt.Scale(scheme='redblue', domainMid=0),
            legend=alt.Legend(format='.0%'),
        ),
        tooltip=[
            alt.Tooltip('state:N', title='State'),
            alt.Tooltip('city:N', title='City'),
            alt.Tooltip('zip_code:N', title='ZIP Code'),
            alt.Tooltip('bed:Q', title='Beds', format='.0f'),
            alt.Tooltip('bath:Q', title='Baths', format='.0f'),
            alt.Tooltip('house_size:Q', title='House Size', format=',.0f'),
            alt.Tooltip('price:Q', title='Actual Price', format='$,.0f'),
            alt.Tooltip('predicted_price:Q', title='Predicted Price', format='$,.0f'),
            alt.Tooltip('residual:Q', title='Residual ($)', format='$,.0f'),
            alt.Tooltip('residual_pct:Q', title='Residual %', format='.1%'),
        ],
    )
    .properties(height=380)
)
diagonal = alt.Chart(
    pd.DataFrame({'x': [line_min, line_max]})
).mark_line(color='black', strokeDash=[4, 4]).encode(
    alt.X('x:Q'),
    alt.Y('x:Q'),
)
st.altair_chart((property_scatter + diagonal).interactive(), use_container_width=True)
st.caption(
    'Points below the diagonal sold below predicted fair value. The scatter is sampled when '
    'needed to keep the page responsive.'
)

opportunities = filtered_df.sort_values(['residual_pct', 'price'], ascending=[True, True]).head(
    TOP_OPPORTUNITY_ROWS
)
opportunity_table = opportunities[
    ['state', 'city', 'zip_code', 'bed', 'bath', 'house_size', 'price', 'predicted_price', 'residual', 'residual_pct']
].rename(
    columns={
        'state': 'State',
        'city': 'City',
        'zip_code': 'ZIP Code',
        'bed': 'Beds',
        'bath': 'Baths',
        'house_size': 'House Size (sqft)',
        'price': 'Actual Price',
        'predicted_price': 'Predicted Fair Price',
        'residual': 'Actual - Predicted ($)',
        'residual_pct': 'Residual %',
    }
)

st.markdown('**Top Opportunity Candidates**')
st.caption('Sorted by lowest residual percentage first, then by lower absolute price.')
st.dataframe(
    opportunity_table.style.format(
        {
            'Beds': '{:.0f}',
            'Baths': '{:.0f}',
            'House Size (sqft)': '{:,.0f}',
            'Actual Price': '${:,.0f}',
            'Predicted Fair Price': '${:,.0f}',
            'Actual - Predicted ($)': '${:,.0f}',
            'Residual %': '{:.1%}',
        },
        na_rep='-',
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader('Market Risk View')
st.markdown(
    'ZIP-level aggregation adds market context to the property screener. Look for ZIPs with '
    'negative average residual percentages without landing in the highest-volatility corner of '
    'the chart.'
)

if zip_summary.empty:
    st.info('No ZIP codes meet the current minimum listing threshold. Lower the slider to inspect ZIP-level risk.')
else:
    risk_scatter = (
        alt.Chart(zip_summary)
        .mark_circle(opacity=0.85, stroke='white', strokeWidth=0.5)
        .encode(
            x=alt.X(
                'mean_residual_pct:Q',
                title='Average Residual % by ZIP',
                axis=alt.Axis(format='.0%'),
            ),
            y=alt.Y(
                'cv:Q',
                title='Coefficient of Variation (Price Volatility)',
                axis=alt.Axis(format='.2f'),
            ),
            size=alt.Size(
                'zip_count:Q',
                title='# Listings',
                scale=alt.Scale(range=[80, 1600]),
            ),
            color=alt.Color(
                'underpriced_share:Q',
                title='Underpriced Share',
                scale=alt.Scale(scheme='teals'),
                legend=alt.Legend(format='.0%'),
            ),
            tooltip=[
                alt.Tooltip('zip_code:N', title='ZIP Code'),
                alt.Tooltip('state:N', title='State'),
                alt.Tooltip('city:N', title='City'),
                alt.Tooltip('zip_count:Q', title='Listings', format=','),
                alt.Tooltip('zip_mean_price:Q', title='Average Price', format='$,.0f'),
                alt.Tooltip('cv:Q', title='Volatility (CV)', format='.2f'),
                alt.Tooltip('mean_residual_pct:Q', title='Average Residual %', format='.1%'),
                alt.Tooltip('underpriced_share:Q', title='Underpriced Share', format='.1%'),
            ],
        )
        .properties(height=380)
    )
    risk_rules = alt.Chart(
        pd.DataFrame(
            {
                'threshold': [UNDERPRICED_THRESHOLD, 0.0],
                'label': ['Underpriced cutoff', 'Fair value'],
            }
        )
    ).mark_rule(color='#495057', strokeDash=[6, 4]).encode(
        x='threshold:Q',
        tooltip=['label:N', alt.Tooltip('threshold:Q', format='.0%')],
    )
    st.altair_chart((risk_scatter + risk_rules).interactive(), use_container_width=True)
    st.caption(
        'ZIPs farther left look cheaper relative to the model on average, while higher values on '
        'the y-axis indicate more price volatility.'
    )

    volatile_table = zip_summary.sort_values(['cv', 'zip_count'], ascending=[False, False]).head(
        TOP_VOLATILE_ROWS
    )[
        ['zip_code', 'state', 'city', 'zip_count', 'zip_mean_price', 'cv', 'mean_residual_pct', 'underpriced_share']
    ].rename(
        columns={
            'zip_code': 'ZIP Code',
            'state': 'State',
            'city': 'City',
            'zip_count': 'Listings',
            'zip_mean_price': 'Average Price',
            'cv': 'Volatility (CV)',
            'mean_residual_pct': 'Average Residual %',
            'underpriced_share': 'Underpriced Share',
        }
    )

    st.markdown('**Most Volatile ZIP Codes in Current Filter**')
    st.dataframe(
        volatile_table.style.format(
            {
                'Listings': '{:,.0f}',
                'Average Price': '${:,.0f}',
                'Volatility (CV)': '{:.2f}',
                'Average Residual %': '{:.1%}',
                'Underpriced Share': '{:.1%}',
            },
            na_rep='-',
        ),
        use_container_width=True,
        hide_index=True,
    )
