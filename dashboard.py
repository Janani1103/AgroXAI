import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import os
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="AgroXAI Dashboard", page_icon="🌾", layout="wide")

# Custom CSS for YellowGreen theme
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        background-color: #9acd32;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #6b8e23;
        color: white;
    }
    .prediction-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: #9acd32;
    }
    </style>
    """, unsafe_allow_stdio=True)

@st.cache_resource
def load_model_artifacts():
    model_path = 'backend/models/xgboost_model.joblib'
    prep_path = 'backend/models/preprocessor.joblib'
    feat_path = 'backend/models/feature_names.joblib'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        feature_names = joblib.load(feat_path)
        explainer = shap.TreeExplainer(model)
        return model, preprocessor, feature_names, explainer
    return None, None, None, None

def main():
    st.title("🌾 AgroXAI: Rice Yield Prediction & XAI")
    st.markdown("Transparent agricultural decision support powered by XGBoost and SHAP.")

    model, preprocessor, feature_names, explainer = load_model_artifacts()

    if model is None:
        st.error("Model artifacts not found. Please run `python backend/train.py` first.")
        return

    # Sidebar Inputs
    st.sidebar.header("📍 Farm Parameters")
    
    with st.sidebar:
        st.subheader("🌦️ Climate")
        rainfall = st.slider("Rainfall (mm)", 800, 2000, 1200)
        temp = st.slider("Temperature (°C)", 24.0, 35.0, 28.0)
        humidity = st.slider("Humidity (%)", 60, 95, 78)
        sunshine = st.slider("Sunshine (hours)", 4.0, 9.0, 6.2)
        
        st.subheader("🌱 Soil")
        soil_type = st.selectbox("Soil Type", ["clay", "sandy", "loamy"], index=2)
        soil_ph = st.slider("Soil pH", 5.5, 7.5, 6.5)
        soil_moisture = st.slider("Soil Moisture (%)", 15, 40, 25)
        
        st.subheader("🚜 Farming")
        fertilizer = st.slider("Fertilizer (kg/acre)", 20, 100, 50)
        seed_type = st.selectbox("Seed Type", ["local", "hybrid"], index=1)
        irrigation = st.selectbox("Irrigation", ["low", "medium", "high"], index=2)

    # Prepare input for prediction
    input_data = pd.DataFrame({
        "Rainfall": [rainfall],
        "Temperature": [temp],
        "Humidity": [humidity],
        "Sunshine": [sunshine],
        "Soil_Type": [soil_type],
        "Soil_pH": [soil_ph],
        "Soil_Moisture": [soil_moisture],
        "Fertilizer": [fertilizer],
        "Seed_Type": [seed_type],
        "Irrigation": [irrigation]
    })

    # Preprocess
    X_processed = preprocessor.transform(input_data)
    
    # Prediction
    prediction = model.predict(X_processed)[0]

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Yield</h3>
                <div class="prediction-value">{prediction:.2f}</div>
                <p>tons/hectare</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Insight:** " + 
                ("High fertilizer and sunshine are boosting your yield!" if fertilizer > 60 else "Consider increasing fertilizer or irrigation for better results."))

    with col2:
        st.subheader("🔍 Explainable AI (SHAP Analysis)")
        
        # SHAP calculation
        shap_values = explainer(X_processed)
        
        # Display SHAP Waterfall Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Note: we need to use the processed feature names
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        plt.title("Feature Impact on Prediction")
        st.pyplot(plt)
        
        st.write("This chart shows how much each factor added to or subtracted from the base expected yield.")

    # Data Preview
    with st.expander("📊 View Input Data"):
        st.write(input_data)

if __name__ == "__main__":
    main()
