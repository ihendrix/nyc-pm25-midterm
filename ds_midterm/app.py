import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="NYC PM2.5 Predictive Dashboard",
    page_icon="🌆",
    layout="wide"
)

page = st.sidebar.selectbox(
    "Select Page",
    ["Background", "Visualizations", "Prediction"]
)


# =========================================================
# LOAD + PREP DATA
# =========================================================
@st.cache_data
def load_data():
    # Air quality
    aqi = pd.read_csv("ad_aqi_tracker_data.csv")
    aqi["Date"] = pd.to_datetime(aqi["Date"])
    aqi = aqi.sort_values("Date").reset_index(drop=True)

    # Previous-day AQI is a derived predictor from the AQI dataset
    aqi["Previous-Day AQI"] = aqi["PM2.5 AQI Value"].shift(1)

    # Traffic
    traffic = pd.read_csv("Automated_Traffic_Volume_Counts_20261005.csv")
    traffic["Date"] = pd.to_datetime(
        dict(
            year=traffic["Yr"],
            month=traffic["M"],
            day=traffic["D"]
        )
    )
    traffic["Traffic Volume"] = (
        traffic["sum_vol"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Weather
    weather = pd.read_csv(
        "open-meteo-40.74N74.04W51m.csv",
        skiprows=3
    )
    weather["time"] = pd.to_datetime(weather["time"])
    weather["Date"] = weather["time"].dt.normalize()

    weather_daily = (
        weather
        .groupby("Date", as_index=False)["dew_point_2m (°C)"]
        .mean()
        .rename(
            columns={
                "dew_point_2m (°C)": "Daily Avg Dew Point (°C)"
            }
        )
    )

    # Combined daily modeling dataset
    merged = (
        aqi[
            [
                "Date",
                "PM2.5 AQI Value",
                "Previous-Day AQI"
            ]
        ]
        .merge(
            traffic[["Date", "Traffic Volume"]],
            on="Date",
            how="inner"
        )
        .merge(
            weather_daily,
            on="Date",
            how="inner"
        )
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return aqi, traffic, weather, weather_daily, merged


aqi, traffic, weather, weather_daily, merged = load_data()


# =========================================================
# MODEL
# =========================================================
FEATURES = [
    "Previous-Day AQI",
    "Traffic Volume",
    "Daily Avg Dew Point (°C)"
]

TARGET = "PM2.5 AQI Value"


@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Standardized coefficients for fair driver comparison
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).ravel()

    standardized_model = LinearRegression()
    standardized_model.fit(X_train_scaled, y_train_scaled)

    standardized_coefs = pd.DataFrame(
        {
            "Variable": FEATURES,
            "Standardized Coefficient": standardized_model.coef_
        }
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        r2,
        mae,
        rmse,
        standardized_coefs
    )


(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    predictions,
    model_r2,
    model_mae,
    model_rmse,
    standardized_coefs
) = train_model(merged)


# =========================================================
# HELPERS
# =========================================================
def aqi_category(value):
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Moderate"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif value <= 200:
        return "Unhealthy"
    elif value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def show_hero():
    hero_html = """
<div style="padding:42px 46px; border-radius:8px; background:linear-gradient(120deg,#102a43,#486581); color:white; margin-bottom:28px;">
    <div style="font-size:18px; margin-bottom:10px;">NYC AIR QUALITY</div>
    <div style="font-size:42px; font-weight:700; line-height:1.15;">NYC PM2.5 Predictive Dashboard</div>
    <div style="font-size:19px; margin-top:14px;">Traffic • Weather • Recent Air Quality</div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)


# =========================================================
# BACKGROUND PAGE
# =========================================================
if page == "Background":
    show_hero()

    st.header("🎯 Business Case")

    st.write(
        """
        PM2.5 air quality can change from day to day. This project tests whether
        recent air quality, traffic volume, and weather can be used to estimate
        daily PM2.5 Air Quality Index (AQI) in New York City.
        """
    )

    st.markdown(
        """
        **Problem:** Higher-AQI days can create greater health risk and require closer monitoring.  
        **Objective:** Build a simple linear regression model that estimates daily PM2.5 AQI.  
        **Use case:** Provide a quick screening estimate that can flag days that may deserve closer attention.
        """
    )

    st.header("📁 Dataset")

    st.write(
        """
        The app combines three 2024 data sources. Hourly weather observations are
        converted to daily averages, and all sources are joined by date.
        `Previous-Day AQI` is created from the air-quality data as an additional
        predictor for the next day's AQI.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Air-quality data", f"{len(aqi):,} days")
    c2.metric("Traffic data", f"{traffic['Date'].nunique():,} days")
    c3.metric("Weather data", f"{len(weather):,} hourly rows")

    st.write("A preview of the final modeling dataset is shown below:")

    preview = merged.head(6).copy()
    preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d")
    preview["Traffic Volume"] = preview["Traffic Volume"].map(
        lambda x: f"{x:,.0f}"
    )
    preview["Daily Avg Dew Point (°C)"] = (
        preview["Daily Avg Dew Point (°C)"].round(2)
    )

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"""
        The final modeling dataset contains **{len(merged)} matched daily observations**.
        """
    )

    st.markdown(
        """
        **Target:** `PM2.5 AQI Value`  
        **Predictors:** `Previous-Day AQI`, `Traffic Volume`, `Daily Avg Dew Point (°C)`
        """
    )

    st.subheader("Sources")

    st.markdown(
        """
        - [EPA Air Quality System (AQS)](https://www.epa.gov/aqs)
        - [NYC DOT Automated Traffic Volume Counts](https://data.cityofnewyork.us/d/7ym2-wayt)
        - [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
        """
    )

    st.caption(
        "This is a classroom model for the midterm project, not an official air-quality forecast."
    )


# =========================================================
# VISUALIZATIONS PAGE
# =========================================================
elif page == "Visualizations":
    st.title("📊 NYC PM2.5 Data Visualization")

    st.write(
        """
        These visuals show the main patterns in the data and identify which
        variables have the strongest linear relationship with daily PM2.5 AQI.
        """
    )

    avg_aqi = aqi[TARGET].mean()
    max_aqi = aqi[TARGET].max()
    max_date = aqi.loc[
        aqi[TARGET].idxmax(),
        "Date"
    ].strftime("%b %d, %Y")

    previous_corr = merged[TARGET].corr(
        merged["Previous-Day AQI"]
    )
    traffic_corr = merged[TARGET].corr(
        merged["Traffic Volume"]
    )
    dew_corr = merged[TARGET].corr(
        merged["Daily Avg Dew Point (°C)"]
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matched observations", f"{len(merged)} days")
    m2.metric("Average PM2.5 AQI", f"{avg_aqi:.1f}")
    m3.metric("Highest PM2.5 AQI", f"{max_aqi}")
    m4.metric("Peak date", max_date)

    st.divider()

    # AQI OVER TIME
    st.header("PM2.5 AQI Across 2024")

    aqi_plot = aqi.sort_values("Date").copy()
    aqi_plot["7-Day Average"] = (
        aqi_plot[TARGET]
        .rolling(7, min_periods=1)
        .mean()
    )

    fig1, ax1 = plt.subplots(figsize=(11, 4.5))

    ax1.plot(
        aqi_plot["Date"],
        aqi_plot[TARGET],
        alpha=0.45,
        label="Daily AQI"
    )
    ax1.plot(
        aqi_plot["Date"],
        aqi_plot["7-Day Average"],
        linewidth=2,
        label="7-day average"
    )

    ax1.set_title("Daily PM2.5 AQI in New York City")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PM2.5 AQI")
    ax1.legend()
    fig1.autofmt_xdate()
    fig1.tight_layout()

    st.pyplot(fig1)
    plt.close(fig1)

    st.write(
        f"""
        **Insight:** AQI changes substantially across the year. The highest daily
        value is **{max_aqi}** on **{max_date}**, compared with a full-year average
        of **{avg_aqi:.1f}**.
        """
    )

    st.divider()

    # PREDICTOR RELATIONSHIPS
    st.header("Predictor Relationships")

    tab1, tab2, tab3 = st.tabs(
        ["Previous-Day AQI", "Traffic Volume", "Dew Point"]
    )

    with tab1:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sns.regplot(
            data=merged,
            x="Previous-Day AQI",
            y=TARGET,
            scatter_kws={"alpha": 0.55},
            line_kws={"linewidth": 2},
            ax=ax2
        )
        ax2.set_title("Previous-Day AQI vs. Today's PM2.5 AQI")
        ax2.set_xlabel("Previous-Day AQI")
        ax2.set_ylabel("Today's PM2.5 AQI")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.write(
            f"""
            **Insight:** Previous-day AQI has the strongest relationship with
            today's AQI (**r = {previous_corr:.2f}**). This suggests air-quality
            conditions tend to persist from one day to the next.
            """
        )

    with tab2:
        traffic_limit = merged["Traffic Volume"].quantile(0.95)
        traffic_plot = merged[
            merged["Traffic Volume"] <= traffic_limit
        ].copy()

        traffic_corr_display = traffic_plot[TARGET].corr(
            traffic_plot["Traffic Volume"]
        )

        fig3, ax3 = plt.subplots(figsize=(9, 5))
        sns.regplot(
            data=traffic_plot,
            x="Traffic Volume",
            y=TARGET,
            scatter_kws={"alpha": 0.55},
            line_kws={"linewidth": 2},
            ax=ax3
        )
        ax3.set_title("Daily Traffic Volume vs. PM2.5 AQI")
        ax3.set_xlabel("Traffic Volume")
        ax3.set_ylabel("PM2.5 AQI")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.caption(
            "The highest 5% of traffic-volume observations are excluded from this chart only so the main pattern is easier to see."
        )

        st.write(
            f"""
            **Insight:** Traffic volume has almost no direct linear relationship
            with PM2.5 AQI in the typical range (**r = {traffic_corr_display:.2f}**).
            """
        )

    with tab3:
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        sns.regplot(
            data=merged,
            x="Daily Avg Dew Point (°C)",
            y=TARGET,
            scatter_kws={"alpha": 0.55},
            line_kws={"linewidth": 2},
            ax=ax4
        )
        ax4.set_title("Daily Average Dew Point vs. PM2.5 AQI")
        ax4.set_xlabel("Daily Average Dew Point (°C)")
        ax4.set_ylabel("PM2.5 AQI")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        st.write(
            f"""
            **Insight:** Dew point has a weak positive relationship with PM2.5 AQI
            (**r = {dew_corr:.2f}**).
            """
        )

    st.divider()

    st.header("Correlation Summary")

    correlation_data = merged[
        [
            TARGET,
            "Previous-Day AQI",
            "Traffic Volume",
            "Daily Avg Dew Point (°C)"
        ]
    ].corr()

    fig5, ax5 = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(
        correlation_data,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=ax5
    )
    ax5.set_title("Correlation Matrix")
    fig5.tight_layout()

    st.pyplot(fig5)
    plt.close(fig5)

    st.info(
        "Key finding: recent AQI is the strongest simple predictor. Traffic and dew point add context, but each has a much weaker direct relationship with daily AQI."
    )


# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "Prediction":
    st.title("📈 PM2.5 AQI Prediction")

    st.write(
        """
        The model uses linear regression to estimate today's PM2.5 AQI from
        yesterday's AQI, traffic volume, and daily average dew point.
        """
    )

    # INPUTS
    st.header("Make a Prediction")

    i1, i2, i3 = st.columns(3)

    with i1:
        previous_aqi_input = st.slider(
            "Previous-Day AQI",
            min_value=int(merged["Previous-Day AQI"].min()),
            max_value=int(merged["Previous-Day AQI"].max()),
            value=int(merged["Previous-Day AQI"].median()),
            step=1
        )

    with i2:
        traffic_input = st.slider(
            "Traffic Volume",
            min_value=int(merged["Traffic Volume"].min()),
            max_value=int(merged["Traffic Volume"].quantile(0.95)),
            value=int(merged["Traffic Volume"].median()),
            step=500
        )

    with i3:
        dew_input = st.slider(
            "Daily Avg Dew Point (°C)",
            min_value=float(
                round(merged["Daily Avg Dew Point (°C)"].min(), 1)
            ),
            max_value=float(
                round(merged["Daily Avg Dew Point (°C)"].max(), 1)
            ),
            value=float(
                round(merged["Daily Avg Dew Point (°C)"].median(), 1)
            ),
            step=0.5
        )

    user_input = pd.DataFrame(
        {
            "Previous-Day AQI": [previous_aqi_input],
            "Traffic Volume": [traffic_input],
            "Daily Avg Dew Point (°C)": [dew_input]
        }
    )

    predicted_aqi = float(model.predict(user_input)[0])
    predicted_aqi = max(predicted_aqi, 0)
    predicted_category = aqi_category(predicted_aqi)

    p1, p2 = st.columns(2)
    p1.metric("Predicted PM2.5 AQI", f"{predicted_aqi:.1f}")
    p2.metric("AQI Category", predicted_category)

    st.info(
        "Use case: this gives a quick daily estimate from information that is already known or can be observed. It can flag a day for closer air-quality monitoring."
    )

    st.divider()

    # MODEL PERFORMANCE
    st.header("Model Performance")

    e1, e2, e3 = st.columns(3)
    e1.metric("R²", f"{model_r2:.2f}")
    e2.metric("MAE", f"{model_mae:.1f} AQI")
    e3.metric("RMSE", f"{model_rmse:.1f} AQI")

    st.caption(
        "Evaluation uses an 80% training / 20% test split with random_state=42."
    )

    st.write(
        """
        **R²** shows how much variation in AQI the model explains.
        **MAE** is the average absolute prediction error.
        **RMSE** gives more weight to larger prediction errors.
        """
    )

    st.divider()

    # ACTUAL VS PREDICTED
    st.header("Actual vs. Predicted AQI")

    comparison = pd.DataFrame(
        {
            "Date": merged.loc[X_test.index, "Date"].values,
            "Actual AQI": y_test.values,
            "Predicted AQI": predictions
        }
    ).sort_values("Date")

    fig6, ax6 = plt.subplots(figsize=(10, 5))

    ax6.plot(
        comparison["Date"],
        comparison["Actual AQI"],
        marker="o",
        linewidth=2,
        label="Actual AQI"
    )
    ax6.plot(
        comparison["Date"],
        comparison["Predicted AQI"],
        marker="o",
        linewidth=2,
        label="Predicted AQI"
    )

    ax6.set_title("Test Set: Actual vs. Predicted PM2.5 AQI")
    ax6.set_xlabel("Date")
    ax6.set_ylabel("PM2.5 AQI")
    ax6.legend()
    fig6.autofmt_xdate()
    fig6.tight_layout()

    st.pyplot(fig6)
    plt.close(fig6)

    st.write(
        """
        **Insight:** The model follows part of the movement in AQI but still misses
        many sharp daily changes. It works as a baseline estimate, not a complete
        air-quality forecasting system.
        """
    )

    st.divider()

    # MODEL DRIVERS
    st.header("What Drives the Prediction?")

    driver_plot = standardized_coefs.copy()
    driver_plot["Absolute Influence"] = (
        driver_plot["Standardized Coefficient"].abs()
    )
    driver_plot = driver_plot.sort_values(
        "Absolute Influence",
        ascending=True
    )

    fig7, ax7 = plt.subplots(figsize=(8, 4.5))
    ax7.barh(
        driver_plot["Variable"],
        driver_plot["Absolute Influence"]
    )
    ax7.set_title("Relative Influence of Model Variables")
    ax7.set_xlabel("Absolute standardized coefficient")
    fig7.tight_layout()

    st.pyplot(fig7)
    plt.close(fig7)

    raw_coefs = dict(
        zip(
            FEATURES,
            model.coef_
        )
    )

    driver_table = pd.DataFrame(
        {
            "Variable": [
                "Previous-Day AQI",
                "Traffic Volume",
                "Daily Avg Dew Point (°C)"
            ],
            "Interpretation": [
                (
                    f"+10 AQI yesterday is associated with about "
                    f"{raw_coefs['Previous-Day AQI'] * 10:+.1f} AQI today, "
                    "holding the other variables constant."
                ),
                (
                    f"+10,000 traffic units is associated with about "
                    f"{raw_coefs['Traffic Volume'] * 10000:+.2f} AQI, "
                    "holding the other variables constant."
                ),
                (
                    f"+1°C dew point is associated with about "
                    f"{raw_coefs['Daily Avg Dew Point (°C)']:+.2f} AQI, "
                    "holding the other variables constant."
                )
            ]
        }
    )

    st.dataframe(
        driver_table,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        """
        **Main driver:** Previous-day AQI has the largest influence in the model.
        Traffic volume and dew point have much smaller effects after the variables
        are considered together.
        """
    )

    st.divider()

    st.header("What Does the Model Solve?")

    st.write(
        """
        The app turns recent AQI, traffic, and weather conditions into a single
        daily PM2.5 AQI estimate. This gives users a simple way to identify days
        that may need closer monitoring.

        The model is intentionally simple for this project. A real operational
        forecast would need additional variables such as wind speed, precipitation,
        temperature, humidity, smoke events, and more consistent traffic coverage.
        """
    )
