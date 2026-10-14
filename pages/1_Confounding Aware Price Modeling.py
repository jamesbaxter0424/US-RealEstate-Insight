import json
import altair as alt
import pandas as pd
import streamlit as st

st.title("Confounding-Aware Price Modeling")
st.markdown(
    "Housing prices are driven by both property characteristics (size, beds, baths) and "
    "location (state, city). Because these factors are correlated — urban properties tend to "
    "be smaller yet more expensive — naïve models misattribute location effects to property "
    "features. This page compares a naïve model (property features only) against a "
    "confounding-aware model (property + location) to reveal these hidden effects."
)


@st.cache_data
def load_artifacts():
    with open("artifacts/page1/metrics.json") as f:
        metrics = json.load(f)
    naive_shap_df = pd.read_parquet("artifacts/page1/shap_values_naive.parquet")
    aware_shap_df = pd.read_parquet("artifacts/page1/shap_values_aware.parquet")
    return metrics, naive_shap_df, aware_shap_df


metrics, naive_shap_df, aware_shap_df = load_artifacts()

# ── Section A: Model Comparison ──────────────────────────────────────────────
st.subheader("Model Comparison")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Naïve Model** *(property features only)*")
    st.metric("Test R²",   f"{metrics['naive']['test']['r2_price']:.3f}")
    st.metric("Test RMSE", f"${metrics['naive']['test']['rmse_price']:,.0f}")
    st.metric("Test MAE",  f"${metrics['naive']['test']['mae_price']:,.0f}")
with col2:
    st.markdown("**Confounding-Aware Model** *(property + location)*")
    delta_r2   = metrics['aware']['test']['r2_price']   - metrics['naive']['test']['r2_price']
    delta_rmse = metrics['aware']['test']['rmse_price'] - metrics['naive']['test']['rmse_price']
    delta_mae  = metrics['aware']['test']['mae_price']  - metrics['naive']['test']['mae_price']
    st.metric("Test R²",   f"{metrics['aware']['test']['r2_price']:.3f}",   delta=f"{delta_r2:+.3f}")
    st.metric("Test RMSE", f"${metrics['aware']['test']['rmse_price']:,.0f}", delta=f"${delta_rmse:+,.0f}", delta_color="inverse")
    st.metric("Test MAE",  f"${metrics['aware']['test']['mae_price']:,.0f}",  delta=f"${delta_mae:+,.0f}",  delta_color="inverse")

bar_data = pd.DataFrame({
    "Model":  ["Naïve", "Naïve", "Naïve", "Confounding-Aware", "Confounding-Aware", "Confounding-Aware"],
    "Metric": ["R² (×100)",    "RMSE ($K)", "MAE ($K)", "R² (×100)", "RMSE ($K)", "MAE ($K)"],
    "Value":  [
        metrics['naive']['test']['r2_price'] * 100,
        metrics['naive']['test']['rmse_price'] / 1000,
        metrics['naive']['test']['mae_price'] / 1000,
        metrics['aware']['test']['r2_price'] * 100,
        metrics['aware']['test']['rmse_price'] / 1000,
        metrics['aware']['test']['mae_price'] / 1000,
    ],
})
bar_chart = (
    alt.Chart(bar_data)
    .mark_bar()
    .encode(
        x=alt.X("Model:N", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Value:Q"),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(domain=["Naïve", "Confounding-Aware"], range=["orange", "purple"]),
        ),
        column=alt.Column("Metric:N", header=alt.Header(labelOrient="bottom")),
        tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".2f")],
    )
    .properties(width=120)
)
st.altair_chart(bar_chart)

# ── Section B: Static SHAP Summary Plots ─────────────────────────────────────
st.subheader("SHAP Feature Importance")
st.markdown(
    "Each bar is the mean |SHAP value| for a feature — the average magnitude of "
    "its contribution to model output. Compare how importances shift when location is added."
)
col_b1, col_b2 = st.columns(2)
with col_b1:
    st.markdown("**Naïve Model**")
    st.image("artifacts/page1/shap_summary_naive.png", use_container_width=True)
with col_b2:
    st.markdown("**Confounding-Aware Model**")
    st.image("artifacts/page1/shap_summary_aware.png", use_container_width=True)

# ── Sections C & D: Shared Controls ──────────────────────────────────────────
st.subheader("Interactive SHAP Dependence Plot")
st.markdown(
    "Explore how each feature's SHAP contribution varies with house size and location. "
    "Filtering by state reveals whether confounding by location changes feature importance."
)

all_states = sorted(aware_shap_df['state'].unique())
selected_states = st.multiselect("Filter by state:", all_states, default=[])

naive_features = [c.replace('shap_', '') for c in naive_shap_df.columns if c.startswith('shap_')]
default_feature = 'log_house_size' if 'log_house_size' in naive_features else naive_features[0]
selected_feature = st.selectbox(
    "Feature to examine (Y-axis — SHAP contribution):",
    naive_features,
    index=naive_features.index(default_feature),
)

size_min = float(naive_shap_df['log_house_size'].min())
size_max = float(naive_shap_df['log_house_size'].max())
size_range = st.slider("Filter by log house size:", size_min, size_max, (size_min, size_max))


def filter_shap(df: pd.DataFrame) -> pd.DataFrame:
    if selected_states:
        df = df[df['state'].isin(selected_states)]
    return df[
        (df['log_house_size'] >= size_range[0]) &
        (df['log_house_size'] <= size_range[1])
    ]


def make_dependence_chart(df: pd.DataFrame, feature: str) -> alt.Chart:
    shap_col = f'shap_{feature}'
    if shap_col not in df.columns or df.empty:
        return alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_point()
    plot_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df
    return (
        alt.Chart(plot_df)
        .mark_circle(size=15, opacity=0.5)
        .encode(
            x=alt.X('log_house_size:Q', title='Log House Size'),
            y=alt.Y(f'{shap_col}:Q', title=f'SHAP value — {feature}'),
            color=alt.Color('state:N', title='State'),
            tooltip=['state', 'log_house_size', shap_col],
        )
        .interactive()
        .properties(height=350)
    )


# ── Section C: Tab-switched dependence plots ──────────────────────────────────
tab_naive, tab_aware = st.tabs(["Naïve Model", "Confounding-Aware Model"])

with tab_naive:
    filtered_naive = filter_shap(naive_shap_df)
    st.altair_chart(make_dependence_chart(filtered_naive, selected_feature), use_container_width=True)
    st.caption(f"{len(filtered_naive):,} points shown")

with tab_aware:
    filtered_aware = filter_shap(aware_shap_df)
    st.altair_chart(make_dependence_chart(filtered_aware, selected_feature), use_container_width=True)
    st.caption(f"{len(filtered_aware):,} points shown")

# ── Section D: Feature Importance Shift Table ─────────────────────────────────
st.subheader("Feature Importance Shift: Naïve vs Confounding-Aware")
st.markdown(
    "Mean |SHAP| per feature computed on the filtered subset. "
    "A large shift indicates a feature's apparent importance changes when location is controlled for."
)

aware_features = [c.replace('shap_', '') for c in aware_shap_df.columns if c.startswith('shap_')]

def importance_table(naive_df: pd.DataFrame, aware_df: pd.DataFrame) -> pd.DataFrame:
    naive_imp = {
        f: naive_df[f'shap_{f}'].abs().mean()
        for f in naive_features if f'shap_{f}' in naive_df.columns
    }
    aware_imp = {
        f: aware_df[f'shap_{f}'].abs().mean()
        for f in aware_features if f'shap_{f}' in aware_df.columns
    }
    rows = []
    for feat in sorted(set(naive_features) | set(aware_features)):
        n_val = naive_imp.get(feat)
        a_val = aware_imp.get(feat)
        rows.append({
            'Feature': feat,
            'Naïve mean |SHAP|': round(n_val, 4) if n_val is not None else '—',
            'Aware mean |SHAP|': round(a_val, 4) if a_val is not None else '—',
            'Shift (Aware − Naïve)': (
                round(a_val - n_val, 4)
                if n_val is not None and a_val is not None else '—'
            ),
        })
    return pd.DataFrame(rows)

st.dataframe(importance_table(filtered_naive, filtered_aware), use_container_width=True, hide_index=True)
