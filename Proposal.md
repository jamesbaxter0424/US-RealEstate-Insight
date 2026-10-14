
# Project Proposal: US-RealEstate Insight: Multi-Dimensional Valuation & Investment Decision Support System

**GitHub Repo:** [https://github.com/CMU-IDS-Assignments/ids-s26-final-project-group-7](https://github.com/CMU-IDS-Assignments/ids-s26-final-project-group-7)

**Team Members:**
James Fu, Chyi-Jiunn Lin, Chenkai Liu, Adrian Liu


## 1. Problem Statement

The U.S. residential real estate market is characterized by extreme price volatility and geographical heterogeneity. Prospective buyers and investors often struggle to determine whether a property’s listing price is justified by its intrinsic features or driven by external socio-economic factors. Traditional valuation models often focus solely on house-specific attributes (e.g., bedrooms, square footage) while ignoring the profound impact of the broader local context.

This project addresses a critical data science problem: **How can we build a robust, multi-dimensional framework to find house price determinants and identify market inefficiencies (i.e. underpriced/overpriced assets) across the U.S.?**

By integrating property-level data with external socio-economic indicators, we aim to move beyond simple price prediction. We seek to solve the technical challenges of:
- **Quantifying the Location Premium**, i.e. how much buyers overpay for better house location, through augmentation with external data.
- **Detecting pricing anomalies (residuals)** to provide actionable investment signals.
- **Visualizing complex spatial-economic relationships** for non-expert stakeholders via an interactive interface.


## 2. Main Dataset

We will utilize the **USA Real Estate Dataset** from Kaggle [1], authored by Ahmed Shahriar Sakib. 
**Link:** [https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)

### 2.1 Overview and Provenance
This dataset contains over 2.2 million real estate listings across the United States. The data was originally collected from Realtor.com. Its size and geographic diversity make it a good fundamental for modeling spatial-economic connections, allowing us to evaluate market features nationwide instead of just local housing trends.

### 2.2 Core Feature Dimensions
The dataset consists of 12 primary columns (1 target variable and 11 features) that capture the essential physical, geographic, and temporal traits of each property:
* **Financial Metric (Target Variable):** `price`, representing either the current listing price or the recently sold price.
* **Structural Attributes:** `bed` (number of beds), `bath`(number of baths), and `house_size`, which denotes the living space in square feet.
* **Land & Status Attributes:** `acre_lot` representing total land area, and `status` indicating if the property is sold, to sale, or ready to build.
* **Geographic Attributes:** `street`, `city`, `state`, and `zip_code`. These features are critical for executing spatial joins with our external socio-economic datasets.
* **Temporal Attributes:** `prev_sold_date`, which tracks the previously sold date and is useful for calculating historical price appreciation.
* **Other:** `brokered_by` indicates agency or broker

Note that `street` and `brokered_by` are categorically encoded due to data privacy policy.

### 2.3 Justification for Selection
Traditional real estate datasets often limit their scope to a single county or state. This dataset's nationwide coverage is a helpful feature when we need to identify and try to solve a data problem. By standardizing features like square footage and bedroom count across all 50 states, we can effectively discover geographic insights to spot market values. This dataset is efficient for finding economic value.

### 2.4 Anticipated Data Quality & Preprocessing Challenges
This is a scraped and large scale dataset, we target the following data quality issues:
* **Missing Values:** Attributes such as `prev_sold_date` and `acre_lot` contain missing values. We will need to apply targeted row dropping or replacing strategies.
* **Extreme Outliers:** Real estate data is heavily right-skewed. We will need to filter out data entry errors, such as listings that says over 100 bedrooms or bathrooms. Also, we could apply log transformations to the `price` and `house_size` variables to normalize their distributions.
* **Processing:** The `zip_code` and `city` variables contain thousands of unique categories. We will utilize Target Encoding or embedding techniques rather than standard one-hot encoding to prevent our models from overfitting on location sparsity.


## 3. Augmenting Data Sources

To enhance analytical depth and novelty, we plan to integrate external datasets to provide context beyond the physical structure and plain location of the houses. In this section, we will introduce the data sources which may be helpful in our future analyses, expecially the Analysis Dimension 2.

### Data 1: U.S. Census Socio-Economic Data
We integrate data from the U.S. Census Bureau’s American Community Survey (ACS) [2], focusing on key indicators such as 
* Median household income
* Demographic indicators like population size and demographic structure

These variables serve as proxies for local purchasing power and economic conditions, which are essential for understanding regional differences in housing prices.

The integration of external data enables us to:
* Compare housing prices relative to local economic conditions
* Identify regions with potential affordability gaps
* Provide interpretable explanations for geographic price differences

These augmented features directly support Analysis Dimension 2, where we investigate how external factors contribute to housing price variation.

---
Besides, We also surveyed several related data sources as backups. We will use them if futher insights are required for our future analyses:
### Backup Data 1: Market Liquidity & Yield (Zillow Research Data)

- **Metrics:** Zillow Home Value Index (ZHVI) and Observed Rent Index (ZORI) [3].
- **Significance:** Allows us to calculate price-to-rent ratios, distinguishing between speculative bubbles and genuine high-yield investment zones.
- Link: zillow.com/research/data/

### Backup Data 2: Socio-Environmental Quality (EPA & NCES)

- **Metrics:** National Walkability Index (EPA) [4] and School District Ratings (NCES) [5].
- **Significance:** Quantifies the lifestyle premium by measuring how much buyers overpay for walkability and top-tier education.
- EPA Link: https://catalog.data.gov/dataset/walkability-index8
- NCES Link: https://nces.ed.gov/programs/edge/Geographic/DistrictBoundaries

### Backup Data 3: Economic Resilience (FRED & Opportunity Atlas)

- **Metrics:** Regional Job Growth (FRED) [6] and Social Mobility Scores (Opportunity Atlas) [7].
- **Significance:** Identifies undervalued areas with high upward social mobility and job growth that haven't yet seen a corresponding spike in real estate prices.
- FRED Link: fred.stlouisfed.org
- Opportunity Atlas Link: opportunityatlas.org



## 4. Proposed Solution & Methodology

The project is structured into three analytical pillars plus one integrated interactive system, delivered via a Streamlit application.

### Analysis Dimension 1: Confounding-Aware Price Modeling & Interpretable Feature Analysis (Chyi-Jiunn)
Housing prices are influenced by both property characteristics (e.g., size, number of rooms) and location (e.g., state, city). However, these factors are often correlated rather than independent. For example, properties in high-demand urban areas tend to be smaller yet more expensive, while larger homes are more common in lower-cost regions.
This correlation introduces a confounding effect, where the influence of property features may be overestimated if geographic context is ignored.

In this analysis, we seek to disentangle the effects of property characteristics and location, and to reveal how naïve interpretations can be misleading due to confounding.

**Methodology** 
We plan to adopt a comparative modeling approach to explicitly expose confounding effects:
* **Naïve Baseline Model:** We first train a regression model using only property-level features, excluding location. This model assumes that price is driven solely by structural attributes.
* **Confounding-Aware Model:** We then extend the model by incorporating location variables (e.g., state, city, or zip code). This allows us to account for geographic heterogeneity.
* **Interpret by comparison:** We compare the two models to understand how feature importance changes
  * Use SHAP values or feature importance scores to quantify contributions
  * Compare how the importance of `house_size`, `bed`, and `bath` shifts after including location
  * Analyze whether location absorbs the feature contribution previously attributed to property features

**Visualization** 
* Side-by-side bar charts for Naïve model and Confounding-aware model
* SHAP Dependence / Partial Effect Visualization, where x-axis is house size and y-axis is feature contribution, and the data is color-encoded by states
* Add interactive filtering to disentangle confounding features. Such as:
    * Dropdown to select states
    * Slider to filter house size range


### Analysis Dimension 2: Geospatial & External Factor Analysis (Chenkai)

While property-level features explain part of housing price variation, they do not capture broader regional and socio-economic influences. In this analysis, we aim to understand whether differences in housing prices can be explained by underlying regional characteristics such as purchasing power (or income) and environmental factors.

**Methodology** 
* **Spatial Data Integration:** Merge the Kaggle dataset with augmenting data described in Section 3 to create a unified spatial-economic profile.
* **Impact Analysis:** Determine the Income-to-House-Price Ratio across regions to identify over-leveraged markets.
* **Visualization:** Develop interactive geospatial layers in Streamlit showing the Price-Income Gap to highlight regional affordability.

### Analysis Dimension 3: Investment Opportunity & Anomaly Detection (James)
While previous analyses focus on understanding price determinants, this analysis shifts the focus toward practical decision-making. This allows us to move from descriptive analysis to actionable insights.

**Methodology** 
* **The Underpricing Detector:** Use the residuals from the predictive model ($Residual = Actual - Predicted$) to identify outliers that are statistically cheaper than their features suggest.
* **Market Segmentation:** Cluster properties based on investment profiles (e.g., High-Growth Speculative vs. Stable-Value Residential).
* **Risk Assessment:** Analyze price variance within the same Zip Code to identify market volatility and potential hidden gems.

### Strategic Overview & Streamlit Integration (Adrian)
This part is to integrate the insights from all three analyses into a interactive decision-support system, enabling users to explore housing market patterns and derive actionable insights.

* **Executive Dashboard:** Organize findings into a cohesive visual story, providing a U.S. Market Snapshot with high-level KPIs.
* **The Decision Engine:** Build a User Input interface where users can simulate a property (e.g., "3 bed, 2 bath in Boston") and receive a real-time valuation and Buy/Hold recommendation based on combined model insights.
* **Integration:** Ensure seamless data flow and UI consistency across the 4-page Streamlit application.

## Sketches and Data Analysis

### Data Processing

The dataset contains over 2.2 million real estate listings with several data quality challenges, including missing values, heterogeneous listing statuses, and potential inconsistencies in price semantics.


#### Missing Data Handling

Several important features (e.g., `bed`, `bath`, `house_size`, `acre_lot`) have missing values ranging from 15% to 33%. Instead of dropping large portions of data, we adopt the following strategy:

- **Median imputation** grouped by geographic region (e.g., ZIP code or state)
- **Missing indicator variables** to preserve information about missingness

For `prev_sold_date`, we derive a new feature:

- `years_since_last_sale`

Missing values in this feature are treated as a separate category to capture potential differences between properties with and without transaction history.

The figure below shows the data whose `bed` (col D), `bath` (col E), or `house_size` (col K) features are missing.
![image](https://hackmd.io/_uploads/Sk-g_Snjbl.png)


#### Outlier Removal and Transformation

From the histogram of each features, we found that many features like `price`, `bed`, `bath`, `acre_lot` have skewed distribution and may include unreasonable large values (473 beds?!).

To address extreme values and skewed distributions:

- Remove unrealistic entries (e.g., bed/bath > 20)
- Apply log transformation to `price` and `house_size`
- Trim top 1% of extreme price values

![image](https://hackmd.io/_uploads/BJB4LB3oWe.png)



#### Feature Engineering

We derive several additional features that will help our investigation:

- `price_per_sqft`
- `years_since_last_sale`
- `income_price_ratio` (after merging ACS data)

#### Separate Listing Price and Transaction Price

We describe the details in the section [Dataset-Definition-and-Preprocessing](#Dataset-Definition-and-Preprocessing).

#### Spatial Data Integration (ACS)

We describe the details in the section [Spatial Data Integration with ACS](#Spatial-Data-Integration-with-ACS).


---

### System Design

We propose an interactive Streamlit-based system that integrates the four main analytical components developed by each team member. Each view corresponds directly to one analytical module, ensuring clear mapping between methodology and user interface.

#### 1. Confounding Analysis Dashboard (Chyi-Jiunn)
This view focuses on understanding the relationship between property features and location, and how confounding effects influence price interpretation.

- Side-by-side comparison of models **with and without location features**
- SHAP-based feature contribution visualization to interpret model behavior
- Interactive filters:
  - State selection
  - Time period selection
  - House size range

This module helps users understand whether observed price differences are driven by intrinsic property characteristics or by geographic factors.


![IMG_5455 (2)](https://hackmd.io/_uploads/rJliSBhi-e.jpg)





#### 2. Socio-Economic & Geospatial Analysis (Chenkai)
This module uses socio-economic data from the U.S. Census (ACS) to examine how regional factors, such as income and demographics, affect housing prices beyond basic property features.

- **Interactive Geospatial Visualization:**
    - **Map Interface**: A localized map displaying property distributions. It supports thematic layers such as the Price-Income Ratio to highlight regional affordability gaps.
    - **Reliability Awareness**: display the Median Household Income and its associated Margin of Error (MOE), allowing users to assess the statistical confidence of the data in specific ZIP codes.
- **Drill-down Filters & Controls:**
    - **Price Range Slider**: Enables dynamic filtering of properties within specific budget constraints.
    - **Geographic Selectors**: Synchronized dropdown menus for "State" and "City" to facilitate precise localized market exploration.
- **Methodology & Data Processing:**
    - **Spatial Join Strategy**: Employs ZIP as default method, but will try a ZIP-to-ZCTA join to account for the ~15% mismatch rate by focusing on residential relevance if necessary. A HUD Crosswalk fallback is reserved for regions with high mismatch outliers.
    - **Uncertainty Filtering**: Future logic will utilize the Coefficient of Variation (CV) to flag or exclude ZCTAs with high sampling error.
- **Improvement Roadmap (options):**
    - **Linked Correlation Analysis**: Implementation of a scatter plot mapping ACS income against price-per-square-foot to detect underpriced or "over-leveraged" market anomalies.
    - **Thematic Overlays**: Toggleable layers for population density and socio-environmental quality indicators.

![sketch2](https://hackmd.io/_uploads/HkHr2S3iZg.png)


#### 3. Investment Opportunity Explorer (James)
This view focuses on identifying potential investment opportunities through anomaly detection.


- **Visualization Components**
    - **Residual distribution plot**:Shows where the selected property lies in the market
    - **Comparison panel**: Predicted vs actual price
    - **Volatility indicator**:Based on ZIP-level variance
    
- **Interaction Design**
    - Real-time update when user changes input
    - Filtering by state, ZIP code, and house size
    - Hover interaction to inspect comparable properties

This module translates model outputs into actionable insights by surfacing potentially undervalued properties.

![S__63774723_0](https://hackmd.io/_uploads/SJVTe93oWe.jpg)
![S__63774724_0](https://hackmd.io/_uploads/rJ4aechobl.jpg)



#### 4. Integrated Decision Engine & User Interface (Adrian)
This view integrates insights from all analytical modules into a unified decision-support interface.

- User input panel:
  - Property features (bed, bath, size, location)
- Outputs:
  - Predicted fair market price
  - Overpriced / underpriced signal
- Summary indicators combining:
  - Model prediction
  - Location effects
  - Socio-economic context

This module serves as the main entry point for end users, providing an intuitive interface for exploring and evaluating individual properties.

![upload_8cf18f48a93557d4f61142387743f042](https://hackmd.io/_uploads/ryHGU9hsWl.png)


---



## Addressing TA Feedback: Data Processing and Evaluation Plan
We refine our approach in three areas: dataset definition, spatial data integration, and evaluation. This approach  make our analysis more reliable.


### Dataset Definition and Preprocessing
A major concern mentioned by TA in the dataset is the heterogeneity of the price variable across different listing statuses. Specifically, the dataset includes a mixture of `sold`, `for-sale`, and `ready-to-build` properties, each associated with fundamentally different pricing semantics. Listing prices may be inflated relative to actual transaction values, while ready-to-build prices do not directly reflect market transactions.

To ensure consistency and interpretability, we restrict our predictive modeling to data whose `status = "sold"`. For other analysis, although we will still use other data, we will make sure to discuss them (transaction price vs. listing price) separately.

Although using only sold data reduces the dataset size to approximately 800,000 records (around 36% of the original dataset), it still provides sufficient geographic coverage and statistical power for modeling. More importantly, this filtering ensures that the target variable reflects true market behavior, which is critical for downstream tasks such as anomaly detection and location premium estimation.

### Spatial Data Integration with ACS
To integrate U.S. Census ACS data with our real estate dataset, we must bridge the gap between Census ZCTAs (ZIP Code Tabulation Areas) and USPS ZIP codes. 

* **Primary Approach (Direct Join):** While literature [8,9] indicates that approximately 15% of ZCTAs have non-matching ZIP codes, this mismatch rate is largely driven by non-residential or point-based ZIPs (e.g., PO Boxes, large businesses). For our residential real estate dataset, this mismatch is practically negligible. Prioritizing computational efficiency and convenience, our initial strategy is to perform a direct join using `zip_code` as the primary key.
* **Fallback Strategy:** Should the direct join result in significant data loss or unacceptable anomaly rates during initial evaluations, we will pivot to a more robust mapping method using the HUD USPS ZIP Code to ZCTA Crosswalk file to perform spatial imputations.
* **Margin of Error (MOE) Mitigation:** ACS estimates inherently carry a Margin of Error. To prevent noisy data from skewing our predictive models, we will monitor the Coefficient of Variation (MOE / Estimate). ZCTAs with high uncertainty will either be filtered out or have their MOE included as an explicit feature, allowing the model to penalize low-confidence regions.


### Model Evaluation Framework
To ensure the reliability and generalizability of our predictive models, we will consider evaluating model performance using standard regression metrics:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

In addition, we adopt a geographic cross-validation strategy to prevent overfitting to location-specific patterns. Specifically, we split the dataset by geographic regions (e.g., states), training models on a subset of states and evaluating them on unseen states.

This approach ensures that the model captures generalizable relationships rather than memorizing location-specific price patterns. It is particularly important for our goal of identifying pricing anomalies and estimating location premiums across different regions.

## References
[1] Ahmed Shahriar Sakib. USA Real Estate Dataset. Kaggle.
https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

[2] U.S. Census Bureau. American Community Survey (ACS).
https://www.census.gov/programs-surveys/acs

[3] Zillow Research. Zillow Home Value Index (ZHVI) & Zillow Observed Rent Index (ZORI).
https://www.zillow.com/research/data/

[4] U.S. Environmental Protection Agency (EPA). National Walkability Index.
https://catalog.data.gov/dataset/walkability-index8

[5] National Center for Education Statistics (NCES). School District Boundaries.
https://nces.ed.gov/programs/edge/Geographic/DistrictBoundaries

[6] Federal Reserve Bank of St. Louis. FRED: Federal Reserve Economic Data.
https://fred.stlouisfed.org/

[7] Opportunity Atlas. The Opportunity Atlas: Mapping the Childhood Roots of Social Mobility.
https://www.opportunityatlas.org/

[8] Healthy Cities. (2023). *What is the difference between ZIP Codes and ZCTAs?* 
https://help.healthycities.org/hc/en-us/articles/115006016767-What-is-the-difference-between-ZIP-Codes-and-ZCTAs

[9] PolicyMap. (2016). *What are Zip Code Tabulation Areas?* 
https://www.policymap.com/blog/what-are-zip-code-tabulation-areas
