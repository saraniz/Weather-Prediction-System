import streamlit as st
import pandas as pd
import joblib
import os
import requests
from datetime import datetime, timedelta


st.set_page_config(
    page_title="Weather Forecast Dashboard",
    page_icon="🌦️",
    layout="wide",
)

API_KEY = "18d72659c9873b0a6465c49d14d91e90"

CITIES = {
    "Colombo": {"lat": 6.9271, "lon": 79.8612},
    "Gampaha": {"lat": 7.0873, "lon": 79.9992},
    "Kalutara": {"lat": 6.5854, "lon": 79.9607},
    "Kandy": {"lat": 7.2906, "lon": 80.6337},
    "Matale": {"lat": 7.4675, "lon": 80.6234},
    "Nuwara Eliya": {"lat": 6.9497, "lon": 80.7891},
    "Galle": {"lat": 6.0535, "lon": 80.2210},
    "Matara": {"lat": 5.9549, "lon": 80.5550},
    "Hambantota": {"lat": 6.1246, "lon": 81.1185},
    "Jaffna": {"lat": 9.6615, "lon": 80.0255},
    "Kilinochchi": {"lat": 9.3803, "lon": 80.3760},
    "Mannar": {"lat": 8.9810, "lon": 79.9042},
    "Vavuniya": {"lat": 8.7514, "lon": 80.4971},
    "Mullaitivu": {"lat": 9.2671, "lon": 80.8142},
    "Batticaloa": {"lat": 7.7170, "lon": 81.7000},
    "Ampara": {"lat": 7.2975, "lon": 81.6820},
    "Trincomalee": {"lat": 8.5874, "lon": 81.2152},
    "Kurunegala": {"lat": 7.4863, "lon": 80.3647},
    "Puttalam": {"lat": 8.0362, "lon": 79.8283},
    "Anuradhapura": {"lat": 8.3114, "lon": 80.4037},
    "Polonnaruwa": {"lat": 7.9403, "lon": 81.0188},
    "Badulla": {"lat": 6.9934, "lon": 81.0550},
    "Monaragala": {"lat": 6.8728, "lon": 81.3507},
    "Ratnapura": {"lat": 6.7056, "lon": 80.3847},
    "Kegalle": {"lat": 7.2513, "lon": 80.3464}
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "weather_clean.csv")

TARGETS = ["temperature", "humidity", "pressure", "rain_probability"]

UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "pressure": "hPa",
    "rain_probability": "%"
}


st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: #f8fafc;
}

/* HERO SECTION */
.hero {
    padding: 1.2rem;
    border-radius: 14px;
    background: linear-gradient(120deg, #e0f7fa, #ffffff);
    border: 1px solid #d6e6e3;
    margin-bottom: 1rem;
}

/* SIDEBAR FIX */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* SIDEBAR HEADINGS */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* METRICS FIX */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border-radius: 12px;
    padding: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* METRIC LABEL (TITLE TEXT) */
div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"] {
    color: #000000 !important;
    opacity: 1 !important;
}

/* METRIC VALUE */
div[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 700;
}

/* EXTRA SAFETY (Streamlit sometimes wraps text in spans) */
div[data-testid="stMetric"] * {
    color: #000000 !important;
}

/* TEXT COLOR */
.hero h1 {
    color: #0f172a;
}

.hero p {
    color: #334155;
}

</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="hero">
    <h1>Weather Forecast Dashboard</h1>
    <p>Machine Learning powered multi-day weather prediction system</p>
</div>
""", unsafe_allow_html=True)


def load_model(target):
    return joblib.load(os.path.join(MODEL_PATH, f"model_{target}.pkl"))


def get_last_known_data(city):
    df = pd.read_csv(DATA_PATH)
    df = df[df["city"] == city].sort_values("timestamp")

    if df.empty:
        raise ValueError("No data found for city")

    return df.iloc[-1]


def predict_future(model, last_row, steps=3):
    results = []
    current = last_row.copy()
    base_date = datetime.now()

    for i in range(steps):
        future_date = base_date + timedelta(days=i + 1)

        input_data = {
            "city": current["city"],
            "hour": future_date.hour,
            "day": future_date.day,
            "month": future_date.month,
            "temp_lag_1": current["temp_lag_1"],
            "temp_lag_2": current["temp_lag_2"],
            "temp_lag_3": current["temp_lag_3"],
            "humidity_lag_1": current["humidity_lag_1"],
            "pressure_lag_1": current["pressure_lag_1"],
        }

        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(df)[0]

        results.append((future_date.date(), pred))

        current["temp_lag_3"] = current["temp_lag_2"]
        current["temp_lag_2"] = current["temp_lag_1"]
        current["temp_lag_1"] = pred

    return results


with st.sidebar:
    st.title("Controls")

    city = st.selectbox("Select City", list(CITIES.keys()))
    parameter = st.selectbox("Select Parameter", TARGETS + ["ALL"])
    days = st.slider("Forecast Days", 1, 7, 3)

    run = st.button("Generate Forecast")


if run:
    # st.markdown("<h3 style='color: black;'>Forecast Overview</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<h4 style='color: black;'>{days}-Day Forecast Summary</h4>",
        unsafe_allow_html=True,
    )

    last_row = get_last_known_data(city)

    if parameter != "ALL":

        model = load_model(parameter)
        preds = predict_future(model, last_row, days)

        df = pd.DataFrame(preds, columns=["date", "value"])

        unit = UNITS[parameter]

        avg = df["value"].mean()
        mx = df["value"].max()
        mn = df["value"].min()

        trend = "Rising ↑" if df["value"].iloc[-1] > df["value"].iloc[0] else "Falling ↓"

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(f"Average {parameter}", f"{avg:.2f} {unit}")
        c2.metric(f"Maximum {parameter}", f"{mx:.2f} {unit}")
        c3.metric(f"Minimum {parameter}", f"{mn:.2f} {unit}")
        c4.metric("Trend", trend)

        st.dataframe(df, use_container_width=True)
        st.line_chart(df.set_index("date"))

    else:

        all_data = []

        for t in TARGETS:
            model = load_model(t)
            preds = predict_future(model, last_row, days)

            temp = pd.DataFrame(preds, columns=["date", "value"])
            temp["parameter"] = t
            all_data.append(temp)

        df_all = pd.concat(all_data)

        pivot = df_all.pivot(index="date", columns="parameter", values="value")

        st.dataframe(df_all, use_container_width=True)
        st.line_chart(pivot)

else:
    st.info("Select options and generate forecast")