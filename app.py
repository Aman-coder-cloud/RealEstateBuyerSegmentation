import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f7f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #172554;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e40af);
    padding: 35px;
    border-radius: 15px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 10px;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.section-note {
    color: #64748b;
    font-size: 14px;
}

.insight-box {
    background: #eff6ff;
    padding: 16px;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    margin: 15px 0;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():

    data = pd.read_csv("clients.csv")

    data.columns = data.columns.str.strip().str.lower()

    return data


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "clients.csv not found. "
        "Place it in the same folder as app.py."
    )

    st.stop()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")

    st.stop()


# =====================================================
# DATA VALIDATION
# =====================================================

required_columns = [
    "client_id",
    "client_type",
    "gender",
    "country",
    "region",
    "date_of_birth",
    "acquisition_purpose",
    "satisfaction_score",
    "loan_applied",
    "referral_channel"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing dataset columns: "
        + ", ".join(missing_columns)
    )

    

# =====================================================
# DATE CLEANING AND AGE CALCULATION
# =====================================================

dob = df["date_of_birth"].astype("string").str.strip()

# Format 1: DD-MM-YYYY
hyphen_dates = dob.str.contains("-", na=False)

# Format 2: MM/DD/YYYY
slash_dates = dob.str.contains("/", na=False)

parsed_dob = pd.Series(
    pd.NaT,
    index=df.index,
    dtype="datetime64[ns]"
)

parsed_dob.loc[hyphen_dates] = pd.to_datetime(
    dob.loc[hyphen_dates],
    format="%d-%m-%Y",
    errors="coerce"
)

parsed_dob.loc[slash_dates] = pd.to_datetime(
    dob.loc[slash_dates],
    format="%m/%d/%Y",
    errors="coerce"
)

df["date_of_birth"] = parsed_dob

# Calculate age
today = pd.Timestamp(date.today())

df["age"] = (
    today.year - df["date_of_birth"].dt.year
    - (
        (today.month < df["date_of_birth"].dt.month)
        | (
            (today.month == df["date_of_birth"].dt.month)
            & (today.day < df["date_of_birth"].dt.day)
        )
    )
)

df.loc[
    (df["age"] < 18) | (df["age"] > 100),
    "age"
] = np.nan




# Clean categorical labels
categorical_columns = [
    "client_type",
    "gender",
    "country",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
    "region"
]

for column in categorical_columns:

    df[column] = df[column].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )


# =====================================================
# FINAL K-MEANS MODEL
# =====================================================

model_features = [
    "age",
    "client_type",
    "gender",
    "country",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]


@st.cache_resource
def train_segmentation_model(data):

    # Numerical feature
    numerical_features = [
        "age"
    ]

    # Six categorical features
    categorical_features = [
        "client_type",
        "gender",
        "country",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel"
    ]

    # Total 7 original features
    features = (
        numerical_features
        + categorical_features
    )

    model_data = data[features].copy()

    # Check missing values rather than silently
    # changing the notebook preprocessing.
    if model_data.isna().any().any():

        raise ValueError(
            "Missing values detected in model features. "
            "Apply the same cleaning procedure used "
            "in the final notebook."
        )

    # Feature preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    X = preprocessor.fit_transform(model_data)

    # K-Means model
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # Model evaluation
    score = silhouette_score(
        X,
        labels
    )

    return (
        model,
        preprocessor,
        labels,
        score,
        X.shape
    )


try:

    model, preprocessor, labels, silhouette, feature_shape = (
        train_segmentation_model(df)
    )

    # Attach model results to dataset
    df["cluster"] = labels

except Exception as error:

    st.error(
        f"Model training failed: {error}"
    )

    st.stop()


# =====================================================
# MODEL VERIFICATION
# =====================================================

if feature_shape != (2000, 22):

    st.warning(
        f"Expected feature shape (2000, 22), "
        f"but received {feature_shape}. "
        "Check the dataset and preprocessing."
    )

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🏡 Buyer Intelligence")

st.sidebar.caption(
    "Real Estate Market Analytics"
)

st.sidebar.divider()

st.sidebar.subheader("Dashboard Filters")

countries = sorted(
    df["country"].dropna().unique()
)

regions = sorted(
    df["region"].dropna().unique()
)

purposes = sorted(
    df["acquisition_purpose"].dropna().unique()
)

buyer_types = sorted(
    df["client_type"].dropna().unique()
)

selected_countries = st.sidebar.multiselect(
    "Country",
    countries,
    default=countries
)

selected_regions = st.sidebar.multiselect(
    "Region",
    regions,
    default=regions
)

selected_purposes = st.sidebar.multiselect(
    "Acquisition Purpose",
    purposes,
    default=purposes
)

selected_types = st.sidebar.multiselect(
    "Buyer Type",
    buyer_types,
    default=buyer_types
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Buyer Analysis",
        "Segmentation",
        "Investment Insights",
        "Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Powered by Python, Streamlit and Scikit-learn"
)


# =====================================================
# FILTER DATA
# =====================================================

filtered_df = df[
    df["country"].isin(selected_countries)
    & df["region"].isin(selected_regions)
    & df["acquisition_purpose"].isin(selected_purposes)
    & df["client_type"].isin(selected_types)
].copy()

if filtered_df.empty:

    st.warning(
        "No records found for the selected filters."
    )

    st.stop()


# COMMON HEADER

st.title("🏡 Real Estate Buyer Intelligence")

st.markdown(
    "Buyer segmentation, customer demographics "
    "and investment purpose analysis."
)

st.caption(
    "An interactive analytics platform for exploring "
    "real estate buyer characteristics."
)

st.divider()


# =====================================================
# CHART THEME
# =====================================================

chart_layout = {
    "template": "plotly_white",
    "margin": dict(
        l=15,
        r=15,
        t=45,
        b=15
    )
}


# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "Overview":

    st.header("Market Overview")

    st.markdown(
        '<p class="section-note">'
        'A snapshot of the available buyer dataset.'
        '</p>',
        unsafe_allow_html=True
    )

    total = len(filtered_df)

    individual = (
        filtered_df["client_type"] == "Individual"
    ).sum()

    companies = (
        filtered_df["client_type"] == "Company"
    ).sum()

    investment = (
        filtered_df["acquisition_purpose"] == "Investment"
    ).sum()

    average_age = filtered_df["age"].mean()

    average_satisfaction = (
        filtered_df["satisfaction_score"].mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Buyers",
        f"{total:,}"
    )

    c2.metric(
        "Individual Buyers",
        f"{individual:,}"
    )

    c3.metric(
        "Company Buyers",
        f"{companies:,}"
    )

    c4.metric(
        "Investment Buyers",
        f"{investment:,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Acquisition Purpose")

        purpose_data = (
            filtered_df["acquisition_purpose"]
            .value_counts()
            .reset_index()
        )

        purpose_data.columns = [
            "Purpose",
            "Buyers"
        ]

        fig = px.pie(
            purpose_data,
            names="Purpose",
            values="Buyers",
            hole=0.55
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Buyer Type")

        buyer_data = (
            filtered_df["client_type"]
            .value_counts()
            .reset_index()
        )

        buyer_data.columns = [
            "Buyer Type",
            "Buyers"
        ]

        fig = px.bar(
            buyer_data,
            x="Buyer Type",
            y="Buyers",
            text="Buyers"
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Additional Indicators")

    c1, c2 = st.columns(2)

    c1.metric(
        "Average Buyer Age",
        f"{average_age:.1f} years"
        if pd.notna(average_age) else "N/A"
    )

    c2.metric(
        "Average Satisfaction",
        f"{average_satisfaction:.2f}/5"
        if pd.notna(average_satisfaction) else "N/A"
    )

    st.info(
        "The dataset does not contain income, property "
        "valuation or investment amount. These indicators "
        "should not be interpreted as financial capacity."
    )


# =====================================================
# PAGE 2: BUYER ANALYSIS
# =====================================================

elif page == "Buyer Analysis":

    st.header("Buyer Demographics")

    left, right = st.columns(2)

    with left:

        st.subheader("Gender Distribution")

        gender_data = (
            filtered_df["gender"]
            .value_counts()
            .reset_index()
        )

        gender_data.columns = [
            "Gender",
            "Count"
        ]

        fig = px.bar(
            gender_data,
            x="Gender",
            y="Count",
            text="Count"
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Loan Application Status")

        loan_data = (
            filtered_df["loan_applied"]
            .value_counts()
            .reset_index()
        )

        loan_data.columns = [
            "Loan Status",
            "Count"
        ]

        fig = px.pie(
            loan_data,
            names="Loan Status",
            values="Count",
            hole=0.45
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Referral Channel Analysis")

    referral_data = (
        filtered_df["referral_channel"]
        .value_counts()
        .reset_index()
    )

    referral_data.columns = [
        "Referral Channel",
        "Buyers"
    ]

    fig = px.bar(
        referral_data,
        x="Referral Channel",
        y="Buyers",
        text="Buyers"
    )

    fig.update_layout(**chart_layout)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Age Distribution")

    age_data = filtered_df.dropna(
        subset=["age"]
    )

    fig = px.histogram(
        age_data,
        x="age",
        nbins=20,
        labels={
            "age": "Buyer Age"
        }
    )

    fig.update_layout(**chart_layout)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Top Countries")

    country_data = (
        filtered_df["country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    country_data.columns = [
        "Country",
        "Buyers"
    ]

    fig = px.bar(
        country_data,
        x="Buyers",
        y="Country",
        orientation="h"
    )

    fig.update_layout(**chart_layout)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# PAGE 3: SEGMENTATION
# =====================================================

elif page == "Segmentation":

    st.header("Machine Learning Segmentation")

    st.write(
        "K-Means clustering identifies groups of buyers "
        "with similar demographic and acquisition "
        "characteristics."
    )

    total_buyers = len(df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Buyers",
        f"{total_buyers:,}"
    )

    col2.metric(
        "Number of Clusters",
        "2"
    )

    col3.metric(
        "Silhouette Score",
        f"{silhouette:.4f}"
    )

    st.caption(
        f"Model feature matrix: {feature_shape[0]} records "
        f"and {feature_shape[1]} encoded features."
    )

    st.divider()

    st.subheader("Cluster Distribution")

    cluster_data = (
        cluster_counts
        .rename_axis("Cluster")
        .reset_index(name="Buyers")
    )

    cluster_data["Cluster"] = (
        "Cluster " + cluster_data["Cluster"].astype(str)
    )

    cluster_data["Percentage"] = (
        cluster_data["Buyers"] / len(df) * 100
    )

    fig = px.bar(
        cluster_data,
        x="Cluster",
        y="Buyers",
        text="Percentage",
        labels={
            "Buyers": "Number of Buyers"
        }
    )

    fig.update_layout(**chart_layout)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Cluster Profiles")

    profile = (
        df.groupby("cluster")
        .agg(
            Buyers=("client_id", "count"),
            Average_Age=("age", "mean"),
            Average_Satisfaction=(
                "satisfaction_score",
                "mean"
            )
        )
        .reset_index()
    )

    purpose_profile = pd.crosstab(
        df["cluster"],
        df["acquisition_purpose"],
        normalize="index"
    ) * 100

    profile["Home Purpose (%)"] = (
        profile["cluster"].map(
            purpose_profile["Home"]
        )
        if "Home" in purpose_profile.columns
        else np.nan
    )

    profile["Investment Purpose (%)"] = (
        profile["cluster"].map(
            purpose_profile["Investment"]
        )
        if "Investment" in purpose_profile.columns
        else np.nan
    )

    profile = profile.rename(
        columns={
            "cluster": "Cluster",
            "Average_Age": "Average Age",
            "Average_Satisfaction": "Average Satisfaction"
        }
    )

    st.dataframe(
        profile.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Age Comparison")

    fig = px.box(
        df,
        x="cluster",
        y="age",
        color="cluster",
        labels={
            "cluster": "Cluster",
            "age": "Age"
        }
    )

    fig.update_layout(**chart_layout)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Acquisition Purpose by Cluster")

    purpose_cluster = pd.crosstab(
        df["cluster"],
        df["acquisition_purpose"],
        normalize="index"
    ) * 100

    purpose_cluster.index = [
        f"Cluster {i}"
        for i in purpose_cluster.index
    ]

    fig = px.bar(
        purpose_cluster,
        barmode="group"
    )

    fig.update_layout(
        **chart_layout,
        yaxis_title="Buyers (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "The silhouette score indicates weak cluster "
        "separation. The segments should be treated as "
        "exploratory rather than definitive buyer categories."
    )

    st.info(
        "Model inputs: age, client type, gender, country, "
        "acquisition purpose, loan application and "
        "referral channel. Region and satisfaction score "
        "are excluded from model training."
    )


# =====================================================
# PAGE 4: INVESTMENT INSIGHTS
# =====================================================

elif page == "Investment Insights":

    st.header("Investment Purpose Analysis")

    investment_df = filtered_df[
        filtered_df["acquisition_purpose"] == "Investment"
    ]

    home_df = filtered_df[
        filtered_df["acquisition_purpose"] == "Home"
    ]

    investment_percentage = (
        len(investment_df) / len(filtered_df) * 100
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Investment Buyers",
        f"{len(investment_df):,}"
    )

    c2.metric(
        "Home Buyers",
        f"{len(home_df):,}"
    )

    c3.metric(
        "Investment Share",
        f"{investment_percentage:.2f}%"
    )

    st.divider()

    if not investment_df.empty:

        st.subheader("Investment Buyers by Country")

        country_investment = (
            investment_df["country"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        country_investment.columns = [
            "Country",
            "Buyers"
        ]

        fig = px.bar(
            country_investment,
            x="Buyers",
            y="Country",
            orientation="h"
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("Investment Buyers by Referral Channel")

        referral_investment = (
            investment_df["referral_channel"]
            .value_counts()
            .reset_index()
        )

        referral_investment.columns = [
            "Referral Channel",
            "Buyers"
        ]

        fig = px.pie(
            referral_investment,
            names="Referral Channel",
            values="Buyers",
            hole=0.45
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("Investment Purpose Across Clusters")

        cluster_investment = (
            investment_df["cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_investment.columns = [
            "Cluster",
            "Investment Buyers"
        ]

        cluster_investment["Cluster"] = (
            "Cluster "
            + cluster_investment["Cluster"].astype(str)
        )

        fig = px.bar(
            cluster_investment,
            x="Cluster",
            y="Investment Buyers",
            text="Investment Buyers"
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No investment buyers match the current filters."
        )

    st.divider()

    st.subheader("Business Observations")

    st.markdown("""
    - Acquisition purpose helps distinguish home-oriented
      buyers from investment-oriented buyers.
    - Referral channels provide information about
      customer acquisition sources.
    - Geographic distribution can support further
      market-level analysis.
    - Additional financial information is required
      to assess actual investment capacity.
    """)


# =====================================================
# PAGE 5: DATA EXPLORER
# =====================================================

elif page == "Data Explorer":

    st.header("Dataset Explorer")

    st.write(
        "Explore aggregate information about the "
        "available buyer records."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Filtered Records",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Countries",
        filtered_df["country"].nunique()
    )

    c3.metric(
        "Regions",
        filtered_df["region"].nunique()
    )

    st.divider()

    st.subheader("Dataset Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Total records",
            "Missing values",
            "Duplicate records",
            "Countries represented",
            "Regions represented"
        ],

        "Value": [
            len(filtered_df),
            int(filtered_df.isna().sum().sum()),
            int(filtered_df.duplicated().sum()),
            filtered_df["country"].nunique(),
            filtered_df["region"].nunique()
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Regional Buyer Distribution")

    region_data = (
        filtered_df["region"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    region_data.columns = [
        "Region",
        "Buyers"
    ]

    fig = px.bar(
        region_data,
        x="Buyers",
        y="Region",
        orientation="h"
    )

    fig.update_layout(**chart_layout)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Aggregate Cluster Summary")

    aggregate = (
        filtered_df.groupby("cluster")
        .agg(
            Buyers=("client_id", "count"),
            Average_Age=("age", "mean"),
            Average_Satisfaction=(
                "satisfaction_score",
                "mean"
            )
        )
        .reset_index()
    )

    st.dataframe(
        aggregate.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Aggregate Summary",
        data=aggregate.to_csv(index=False),
        file_name="buyer_cluster_summary.csv",
        mime="text/csv"
    )

  
# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:20px;">

    <h3>Real Estate Buyer Intelligence</h3>

    <p>
    Machine Learning Based Buyer Segmentation
    and Investment Profiling
    </p>

    <p>
    Developed using Python, Streamlit, Pandas,
    Plotly and Scikit-learn.
    </p>

    <small>
    Results are exploratory and depend on dataset quality.
    </small>

    </div>
    """,
    unsafe_allow_html=True
)