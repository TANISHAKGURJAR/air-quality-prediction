import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Prediction System")
st.write("Machine Learning based Air Quality Index (AQI) Prediction")
st.markdown("---")

MODEL_FILE = "air_quality_prediction_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

if not os.path.exists(MODEL_FILE):
    st.error("Model file nahi mili. GitHub repository mein model file upload karo.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("Model load nahi hua. Scikit-learn version compatible nahi hai.")
    st.code(str(e))
    st.info("Model ko compatible scikit-learn version mein dobara train aur save karna pad sakta hai.")
    st.stop()

st.subheader("Pollution aur Weather Details")
st.caption("Apni readings enter karo, phir Predict AQI par click karo.")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=35.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1500.0, value=60.0)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=1000.0, value=20.0)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=1000.0, value=10.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=100.0, value=1.0)

with col2:
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0, max_value=1000.0, value=30.0)
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=28.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=2.0)

if st.button("🔍 Predict AQI", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3,
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind_Speed": wind_speed
    }])

    try:
        prediction = float(model.predict(input_data)[0])
        aqi = max(0, round(prediction))

        if aqi <= 50:
            category = "Good"
            message = "Air quality is satisfactory."
        elif aqi <= 100:
            category = "Moderate"
            message = "Acceptable air quality; sensitive people should take care."
        elif aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
            message = "Sensitive groups may experience health effects."
        elif aqi <= 200:
            category = "Unhealthy"
            message = "Some health effects may occur."
        elif aqi <= 300:
            category = "Very Unhealthy"
            message = "Health alert: reduce outdoor exposure."
        else:
            category = "Hazardous"
            message = "Serious health warning: avoid outdoor exposure."

        st.markdown("---")
        st.subheader("Prediction Result")
        result_col1, result_col2 = st.columns(2)
        result_col1.metric("Predicted AQI", aqi)
        result_col2.metric("Category", category)
        st.info(message)
        st.caption(
            "Note: This is an ML model estimate, not an official real-time AQI reading. "
            "Category thresholds may vary by AQI standard."
        )

        with st.expander("View input values"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("Prediction nahi ho payi. Model ke input columns check karo.")
        st.code(str(e))

st.markdown("---")
st.caption("Air Quality Prediction System | Machine Learning Project")
