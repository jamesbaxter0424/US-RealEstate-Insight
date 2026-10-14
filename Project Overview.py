import streamlit as st

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.header("US-RealEstate Insight: Multi-Dimensional Valuation & Investment Decision Support System")

st.markdown(
    "A multi-page dashboard for U.S. real estate valuation, geographic market analysis, and investment decision support."
)

st.subheader("Domain & Dataset")
st.markdown(
    """
    This project focuses on the U.S. residential real estate market, where housing prices are
    shaped by both property characteristics and local market context. A house may appear expensive
    because it is large, but it may also be expensive because it is located in a high-demand city
    or neighborhood. This makes real estate a useful domain for studying how structural features
    and geography work together in price formation.
    """
)
st.markdown(
    """
    Our primary dataset is the **USA Real Estate Dataset** from Kaggle, which contains more than
    2.2 million Realtor.com listings across the United States. The data includes features such as
    price, bedrooms, bathrooms, house size, lot size, city, state, ZIP code, and previous sale
    history. For modeling, we focus on **sold** properties so that the target reflects actual
    transaction prices rather than listing prices. We also incorporate **state-level median
    household income** from the U.S. Census ACS to provide broader socioeconomic context.
    """
)

st.subheader("Project Goals")
st.markdown(
    """
    The main goal of this project is to build a multi-page decision-support dashboard for U.S.
    housing analysis. Instead of only predicting price, we want to understand how price changes
    once location is taken into account and how those predictions can support market exploration.
    """
)
st.markdown(
    """
    Specifically, our project aims to:

    - compare a **property-only baseline model** with a **confounding-aware model** that includes location features
    - show how geographic context changes both predictive performance and interpretation
    - analyze regional affordability through geospatial and income-based summaries
    - identify potentially underpriced properties and volatile ZIP-code markets using residual-based screening
    - provide an interactive user interface where a user can enter housing features and receive a fair-price estimate and simple buy/hold recommendation
    """
)

st.markdown("Please choose a dashboard using the sidebar on the left.")
