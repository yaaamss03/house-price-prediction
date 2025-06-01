#!/usr/bin/env python
# coding: utf-8

# # **Streamlit for House Price Prediction**

# In[6]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Define the custom transformer class used in your pipeline
from custom_transformers import CombinedAttributesAdder

# Load model and pipeline after defining the class
bundle = joblib.load("final_model.joblib")
model = bundle["model"]
pipeline = bundle["pipeline"]


# Page config and styling
st.set_page_config(page_title="Real e-State", layout="wide")

st.markdown("""
    <style>
        .stWarning, .stAlert, .stException {
            display: none !important;
        }
        #MainMenu, footer, header {
            visibility: hidden;
        }

        .top-bar {
            background-color: #d6e7ff;
            padding: 10px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .top-bar .left, .top-bar .right {
            font-size: 1em;
            color: #444;
            font-weight: 600;
        }

        .image-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            margin-top: 20px;
        }

        .background-image {
            width: 100%;
            height: auto;
            border-radius: 10px;
            opacity: 0.6;
        }

        .title-box {
            text-align: center;
            animation: fadeIn 2s ease-in-out;
            margin-top: -200px;
        }

        .title {
            font-size: 3em;
            font-weight: 700;
            color: #611f1f;
        }

        .tagline {
            font-size: 1.5em;
            font-style: italic;
            color: black;
        }

        .strikethrough {
            text-decoration: line-through;
            color: #555;
        }

        .info-section {
            margin-top: 60px;
            background-color: #eaf2ff;
            padding: 40px;
            border-radius: 15px;
        }

        .info-section h3 {
            text-align: center;
            font-size: 1.6em;
            margin-bottom: 30px;
            color: #333;
        }

        .info-grid {
            display: flex;
            justify-content: center;
            gap: 30px;
        }

        .info-tile {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: white;
            width: 140px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        }

        .info-tile:hover {
            background-color: #f0f8ff;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

    </style>
""", unsafe_allow_html=True)

# Top bar
st.markdown("""
    <div class="top-bar">
        <div class="left">☰ </div>
        <div class="right">🔍 | 🏠 </div>
    </div>
""", unsafe_allow_html=True)

# Hero image & title
col1, col2, col3 = st.columns([1, 2.5, 1])
with col2:
    st.markdown("""
        <div class="image-container">
            <img class="background-image" src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c" />
        </div>
        <div class="title-box">
            <div class="title">Welcome to Real e-State</div>
            <div class="tagline">
                <span class="strikethrough">Home is where the heart is</span><br>
                Heart is where the home is
            </div>
        </div>
    """, unsafe_allow_html=True)

# Info section
st.markdown("""
    <div class="info-section">
        <h3>Why Real e-State?</h3>
        <div class="info-grid">
            <div class="info-tile">🏡<br>California</div>
            <div class="info-tile">💰<br>Money</div>
            <div class="info-tile">🛡️<br>Safety</div>
            <div class="info-tile">🏅<br>Trust</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- House Price Prediction Form ---
st.title("🏘️ California House Price Predictor")
# Collect user input
longitude = st.slider("Longitude", -125.0, -114.0, step=0.1)
latitude = st.slider("Latitude", 32.0, 42.0, step=0.1)
housing_median_age = st.slider("Housing Age", 1, 52)
total_rooms = st.slider("Total Rooms", 1, 20000)
total_bedrooms = st.slider("Total Bedrooms", 1, 10000)
population = st.slider("Population", 1, 40000)
households = st.slider("Households", 1, 7000)
median_income = st.slider("Median Income (in $10,000s)", 0.5, 15.0, step=0.1)
ocean_proximity = st.selectbox("Ocean Proximity", ["INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND", "<1H OCEAN"])

# Assemble into DataFrame
input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

st.write("Your input:")
st.dataframe(input_data)

if st.button("Predict Price"):
    try:
        st.info("⏳ Running prediction...")
        st.write("Input data:", input_data)

        processed = pipeline.transform(input_data)
        st.write("Transformed input shape:", processed.shape)

        prediction = model.predict(processed)
        st.write("Raw prediction output:", prediction)

        st.success(f"💰 Predicted House Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


# In[ ]:




