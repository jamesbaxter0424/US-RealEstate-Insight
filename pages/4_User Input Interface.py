import json
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

REFERENCE_DATE = pd.Timestamp('2024-01-01')

AWARE_FEATURES = [
    'bed', 'bath', 'log_house_size', 'acre_lot',
    'bed_missing', 'bath_missing', 'house_size_missing', 'acre_lot_missing',
    'years_since_last_sale', 'no_prior_sale',
    'state_encoded', 'city_encoded', 'zip_code_encoded',
]


@st.cache_resource
def load_model():
    return lgb.Booster(model_file='artifacts/page1/aware_model.lgbm')


@st.cache_data
def load_encoders():
    with open('artifacts/data/encoders.json') as f:
        return json.load(f)


@st.cache_data
def load_dropdown_data():
    """Build state->city->zip mapping and state medians from training data."""
    train = pd.read_parquet('artifacts/data/train.parquet')
    train = train.dropna(subset=['city', 'zip_code'])
    train['zip_code'] = train['zip_code'].astype(int).astype(str)

    state_cities = (
        train.groupby('state')['city']
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )
    city_zips = (
        train.groupby(['state', 'city'])['zip_code']
        .apply(lambda x: sorted(x.unique().tolist()))
        .to_dict()
    )
    state_medians = (
        train.groupby('state')[['house_size', 'acre_lot']]
        .median()
        .to_dict(orient='index')
    )
    return state_cities, city_zips, state_medians


booster = load_model()
encoders = load_encoders()
state_cities, city_zips, state_medians = load_dropdown_data()

st.title('User Input Interface')
st.markdown('##### Enter housing information below to estimate the predicted price')

col_input, col_spacer, col_output = st.columns([1, 0.1, 1])

with col_input:
    # ── Location ─────────────────────────────────────────────
    st.markdown('**Location**')

    states = sorted(encoders['state_classes'])
    selected_state = st.selectbox('State:', states)

    cities = state_cities.get(selected_state, [])
    selected_city = st.selectbox('City:', cities)

    zips = city_zips.get((selected_state, selected_city), [])
    zip_options = zips + ['Other (custom ZIP)']
    selected_zip_option = st.selectbox('ZIP Code:', zip_options)

    if selected_zip_option == 'Other (custom ZIP)':
        custom_zip = st.text_input('Enter ZIP code:')
    else:
        custom_zip = None

    # ── Property Features ────────────────────────────────────
    st.markdown('**Property Features**')

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        bed = st.number_input('# Beds', min_value=0, max_value=20, value=3)
        house_size = st.number_input('House Size (sqft)', min_value=0, value=0,
                                      help='Leave 0 if unknown')
    with feat_col2:
        bath = st.number_input('# Baths', min_value=0, max_value=20, value=2)
        acre_lot = st.number_input('Lot Size (acres)', min_value=0.0, value=0.0,
                                    format='%.2f', help='Leave 0 if unknown')

    # ── History & Listing ────────────────────────────────────
    st.markdown('**History & Listing**')

    has_prior_sale = st.checkbox('Property was previously sold')
    if has_prior_sale:
        last_sold_date = st.date_input('Last Sold Date')
    else:
        last_sold_date = None

    asking_price = st.number_input('Asking Price ($)', min_value=0, value=0,
                                    help='Optional — leave 0 to skip Buy/Hold recommendation')

# ── Encode inputs and predict ────────────────────────────────

state_classes = encoders['state_classes']
state_encoded = state_classes.index(selected_state)

global_mean = encoders['global_price_mean']

city_means = encoders['city_target_means']
city_encoded = city_means.get(selected_city, global_mean)

zip_means = encoders['zip_code_target_means']
if custom_zip is not None:
    zip_encoded = zip_means.get(custom_zip, global_mean)
else:
    zip_encoded = zip_means.get(selected_zip_option, global_mean)

medians = state_medians.get(selected_state, {'house_size': 1500, 'acre_lot': 0.25})

if house_size == 0:
    house_size_val = medians['house_size']
    house_size_missing = 1
else:
    house_size_val = float(house_size)
    house_size_missing = 0

if acre_lot == 0.0:
    acre_lot_val = medians['acre_lot']
    acre_lot_missing = 1
else:
    acre_lot_val = float(acre_lot)
    acre_lot_missing = 0

log_house_size = np.log1p(house_size_val)

if last_sold_date is not None:
    sold_ts = pd.Timestamp(last_sold_date)
    years_since_last_sale = (REFERENCE_DATE - sold_ts).days / 365.25
    no_prior_sale = 0
else:
    years_since_last_sale = 0.0
    no_prior_sale = 1

features = pd.DataFrame([{
    'bed': float(bed),
    'bath': float(bath),
    'log_house_size': log_house_size,
    'acre_lot': acre_lot_val,
    'bed_missing': 0,
    'bath_missing': 0,
    'house_size_missing': house_size_missing,
    'acre_lot_missing': acre_lot_missing,
    'years_since_last_sale': years_since_last_sale,
    'no_prior_sale': no_prior_sale,
    'state_encoded': state_encoded,
    'city_encoded': city_encoded,
    'zip_code_encoded': zip_encoded,
}])

predicted_log = booster.predict(features[AWARE_FEATURES])[0]
predicted_price = np.expm1(predicted_log)

with col_output:
    st.markdown('')
    st.markdown('')

    st.markdown(
        f'<div style="background-color:#262730;border-radius:8px;padding:20px;text-align:center;margin-bottom:16px;">'
        f'<div style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Predicted Fair Price</div>'
        f'<div style="font-size:36px;font-weight:700;color:#a6e3a1;">${predicted_price:,.0f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if asking_price > 0:
        diff = asking_price - predicted_price
        if asking_price < predicted_price:
            signal = 'BUY'
            color = '#a6e3a1'
            msg = f'Asking ${asking_price:,.0f} is **${abs(diff):,.0f} below** predicted fair value'
        else:
            signal = 'HOLD'
            color = '#f9e2af'
            msg = f'Asking ${asking_price:,.0f} is **${abs(diff):,.0f} above** predicted fair value'

        st.markdown(
            f'<div style="background-color:#262730;border-radius:8px;padding:20px;text-align:center;margin-bottom:16px;">'
            f'<div style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Recommendation</div>'
            f'<div style="font-size:32px;font-weight:700;color:{color};">{signal}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(msg)

    st.markdown(
        '<div style="background-color:#262730;border-radius:8px;padding:14px;font-size:13px;color:#a6adc8;">'
        '<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">How it works</div>'
        'Uses the confounding-aware LightGBM model from Page 1 (R² = 0.814). '
        'Location is target-encoded; unknown ZIP codes use global-mean fallback. '
        'If asking price is below predicted → <span style="color:#a6e3a1;">Buy</span>. '
        'If at or above → <span style="color:#f9e2af;">Hold</span>.'
        '</div>',
        unsafe_allow_html=True,
    )
