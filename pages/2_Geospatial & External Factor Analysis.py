import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

STATE_CENTROIDS = {
    'Alabama': (32.78, -86.83), 'Alaska': (64.07, -153.28), 'Arizona': (34.27, -111.66),
    'Arkansas': (34.89, -92.44), 'California': (37.18, -119.47), 'Colorado': (39.00, -105.55),
    'Connecticut': (41.62, -72.73), 'Delaware': (38.99, -75.51),
    'District of Columbia': (38.91, -77.01), 'Florida': (28.63, -82.45),
    'Georgia': (32.64, -83.44), 'Hawaii': (20.29, -156.37), 'Idaho': (44.35, -114.61),
    'Illinois': (40.04, -89.20), 'Indiana': (39.89, -86.28), 'Iowa': (42.08, -93.50),
    'Kansas': (38.49, -98.38), 'Kentucky': (37.53, -85.30), 'Louisiana': (31.07, -92.00),
    'Maine': (45.37, -69.24), 'Maryland': (39.06, -76.79), 'Massachusetts': (42.26, -71.81),
    'Michigan': (44.35, -85.41), 'Minnesota': (46.28, -94.31), 'Mississippi': (32.74, -89.67),
    'Missouri': (38.36, -92.46), 'Montana': (46.88, -110.36), 'Nebraska': (41.54, -99.80),
    'Nevada': (39.33, -116.63), 'New Hampshire': (43.68, -71.58), 'New Jersey': (40.19, -74.67),
    'New Mexico': (34.41, -106.11), 'New York': (42.95, -75.53),
    'North Carolina': (35.56, -79.39), 'North Dakota': (47.45, -100.47),
    'Ohio': (40.29, -82.79), 'Oklahoma': (35.59, -97.49), 'Oregon': (43.93, -120.56),
    'Pennsylvania': (40.88, -77.80), 'Rhode Island': (41.68, -71.56),
    'South Carolina': (33.92, -80.90), 'South Dakota': (44.44, -100.23),
    'Tennessee': (35.86, -86.35), 'Texas': (31.48, -99.33), 'Utah': (39.31, -111.09),
    'Vermont': (44.07, -72.67), 'Virginia': (37.52, -78.85), 'Washington': (47.38, -120.45),
    'West Virginia': (38.64, -80.62), 'Wisconsin': (44.62, -89.99), 'Wyoming': (43.00, -107.55),
    'Puerto Rico': (18.22, -66.59),
}

st.title("Geospatial & External Factor Analysis")
s = """While property-level features explain part of housing price variation,
they do not capture broader regional and socio-economic influences.
In this analysis, we aim to understand whether differences in housing prices
can be explained by underlying regional characteristics such as purchasing power
(or income) and environmental factors."""
st.markdown(s)

@st.cache_data
def load_data():
    state_df = pd.read_parquet("artifacts/page2/state_summary.parquet")
    city_df = pd.read_parquet("artifacts/page2/city_summary.parquet")
    return state_df, city_df


try:
    state_df, city_df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.error(
        "Artifact files not found. "
        "Please run `python scripts/preprocess_page2.py` first to generate them."
    )
    st.stop()
    data_loaded = False

if data_loaded:
    price_min = int(state_df['median_price'].min())
    price_max = int(state_df['median_price'].max())

    col1, col2 = st.columns(2)
    with col1:
        price_range = st.slider(
            "Filter by median price:",
            price_min, price_max, (price_min, price_max),
            step=10_000, format="$%d",
        )
    with col2:
        st.markdown("**Map:** bubble size = # listings · color = Price-Income Ratio")
        st.markdown("Use the dropdowns below to drill down into the Impact Analysis.")
    filtered_state = state_df[
        (state_df['median_price'] >= price_range[0]) &
        (state_df['median_price'] <= price_range[1])
    ].copy()
    bubble_df = filtered_state.copy()
    bubble_df['lat'] = bubble_df['state'].map(
        lambda s: STATE_CENTROIDS.get(s, (None, None))[0]
    )
    bubble_df['lon'] = bubble_df['state'].map(
        lambda s: STATE_CENTROIDS.get(s, (None, None))[1]
    )
    bubble_df = bubble_df.dropna(subset=['lat', 'lon'])
    US_STATES_URL = "https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/us-10m.json"
    background = (
        alt.Chart(alt.topo_feature(US_STATES_URL, 'states'))
        .mark_geoshape(fill='#e0e0e0', stroke='white', strokeWidth=0.5)
    )
    bubbles = (
        alt.Chart(bubble_df)
        .mark_circle(opacity=0.85, stroke='white', strokeWidth=0.5)
        .encode(
            longitude='lon:Q',
            latitude='lat:Q',
            size=alt.Size(
                'listing_count:Q',
                title='# Listings',
                scale=alt.Scale(range=[80, 3000]),
                legend=alt.Legend(format=','),
            ),
            color=alt.Color(
                'income_price_ratio:Q',
                title='Price / Annual Income',
                scale=alt.Scale(scheme='orangered'),
                legend=alt.Legend(format='.1f'),
            ),
            tooltip=[
                alt.Tooltip('state:N', title='State'),
                alt.Tooltip('median_price:Q', title='Median Price ($)', format='$,.0f'),
                alt.Tooltip('median_income:Q', title='Median Income ($)', format='$,.0f'),
                alt.Tooltip('income_price_ratio:Q', title='Price/Income Ratio', format='.2f'),
                alt.Tooltip('listing_count:Q', title='# Listings', format=','),
            ],
        )
    )
    map_chart = (
        (background + bubbles)
        .project('albersUsa')
        .properties(width=700, height=420)
    )

    st.subheader("Property Distribution & Price-Income Ratio")
    st.markdown(
        "Each bubble represents a state. Adjust the **price slider** above to "
        "filter which states appear on the map."
    )
    st.altair_chart(map_chart, use_container_width=True)

    st.markdown("")
    st.markdown("")
    st.subheader("Impact Analysis")
    st.markdown(
        "Drill down by state or city to explore how median household income "
        "relates to housing prices at a finer granularity."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        all_states = ["All"] + sorted(state_df['state'].unique().tolist())
        selected_state = st.selectbox("Filter by state:", all_states)
    with col_s2:
        if selected_state != "All":
            available_cities = ["All"] + sorted(
                city_df[city_df['state'] == selected_state]['city'].unique().tolist()
            )
        else:
            available_cities = ["All"]
        selected_city = st.selectbox("Filter by city:", available_cities)
    if selected_state != "All":
        scatter_df = city_df[city_df['state'] == selected_state].copy()
        if selected_city != "All":
            scatter_df = scatter_df[scatter_df['city'] == selected_city]
        point_label = 'city'
        subtitle = f"City-level breakdown — {selected_state}"
    else:
        scatter_df = filtered_state.copy()
        point_label = 'state'
        subtitle = "State-level overview — all states in price filter"
    st.markdown(
        f"**Median Household Income vs. Median House Price** ({subtitle}). "
        "Points above the trend line indicate markets expensive relative to local income."
    )
    if scatter_df.empty:
        st.info("No data available for the current selection.")
    else:
        base = alt.Chart(scatter_df)

        scatter = (
            base.mark_circle(opacity=0.75)
            .encode(
                x=alt.X(
                    'median_income:Q',
                    title='Median Household Income ($)',
                    axis=alt.Axis(format='$,.0f'),
                ),
                y=alt.Y(
                    'median_price:Q',
                    title='Median House Price ($)',
                    axis=alt.Axis(format='$,.0f'),
                ),
                color=alt.Color(
                    'income_price_ratio:Q',
                    title='Price/Income Ratio',
                    scale=alt.Scale(scheme='orangered'),
                ),
                size=alt.Size(
                    'listing_count:Q',
                    title='# Listings',
                    scale=alt.Scale(range=[60, 600]),
                ),
                tooltip=[
                    alt.Tooltip(f'{point_label}:N', title=point_label.capitalize()),
                    alt.Tooltip('median_income:Q', title='Median Income ($)', format='$,.0f'),
                    alt.Tooltip('median_price:Q', title='Median Price ($)', format='$,.0f'),
                    alt.Tooltip('income_price_ratio:Q', title='Price/Income Ratio', format='.2f'),
                    alt.Tooltip('listing_count:Q', title='# Listings', format=','),
                ],
            )
            .interactive()
            .properties(height=420)
        )

        if scatter_df['median_income'].nunique() >= 2:
            regression = (
                base.mark_line(color='steelblue', strokeDash=[6, 4], opacity=0.6)
                .transform_regression('median_income', 'median_price')
            )
            final_chart = scatter + regression
        else:
            final_chart = scatter

        st.altair_chart(final_chart, use_container_width=True)
    st.subheader("Regional Affordability Summary")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        row = filtered_state.nlargest(1, 'income_price_ratio')
        if not row.empty:
            st.metric(
                "Least Affordable State",
                row['state'].values[0],
                f"Ratio: {row['income_price_ratio'].values[0]:.1f}x",
            )
    with col_b:
        row = filtered_state.nsmallest(1, 'income_price_ratio')
        if not row.empty:
            st.metric(
                "Most Affordable State",
                row['state'].values[0],
                f"Ratio: {row['income_price_ratio'].values[0]:.1f}x",
            )
    with col_c:
        avg = filtered_state['income_price_ratio'].mean()
        st.metric("National Avg. Ratio", f"{avg:.1f}x income")

    display_df = filtered_state[
        ['state', 'median_price', 'median_income', 'income_price_ratio', 'listing_count']
    ].sort_values('income_price_ratio', ascending=False).reset_index(drop=True)

    display_df.columns = [
        'State', 'Median Price ($)', 'Median Income ($)',
        'Price/Income Ratio', '# Listings',
    ]

    st.dataframe(
        display_df.style.format({
            'Median Price ($)': '${:,.0f}',
            'Median Income ($)': '${:,.0f}',
            'Price/Income Ratio': '{:.2f}',
            '# Listings': '{:,}',
        }),
        use_container_width=True,
        hide_index=True,
    )
