import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Power Demand Prediction")

# Title
st.title("Power Demand Prediction")
st.write("Enter the power generation values to predict electricity demand")

# Loading model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Input section
st.header("Enter Input Values")

col1, col2 = st.columns(2)

with col1:
    gas = st.number_input("Gas (MW)", min_value=0.0, value=3000.0)
    liquid_fuel = st.number_input("Liquid Fuel (MW)", min_value=0.0, value=500.0)
    coal = st.number_input("Coal (MW)", min_value=0.0, value=1000.0)
    hydro = st.number_input("Hydro (MW)", min_value=0.0, value=200.0)
    solar = st.number_input("Solar (MW)", min_value=0.0, value=100.0)

with col2:
    india_bheramara = st.number_input("India Bheramara HVDC (MW)", min_value=0.0, value=500.0)
    india_tripura = st.number_input("India Tripura (MW)", min_value=0.0, value=100.0)
    generation_mw = st.number_input("Total Generation (MW)", min_value=0.0, value=5000.0)
    load_shedding = st.number_input("Load Shedding (MW)", min_value=0.0, value=0.0)
    hour = st.slider("Hour of Day", 0, 23, 12)

# Additional inputs
col3, col4 = st.columns(2)
with col3:
    day = st.slider("Day of Month", 1, 31, 15) # by default the day will be 15
    month = st.slider("Month", 1, 12, 6) # by default the month will be June
with col4:
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)

# Prediction button
if st.button("Predict Demand", type="primary"):
    # Calculating derived features
    total_india_import = india_bheramara + india_tripura
    # For supply_demand_gap, we estimate it as a small percentage of generation
    supply_demand_gap = generation_mw * 0.01  # Estimated 1% gap
    
    # Creating input dataframe
    input_data = pd.DataFrame({
        'generation_mw': [generation_mw],
        'load_shedding': [load_shedding],
        'gas': [gas],
        'liquid_fuel': [liquid_fuel],
        'coal': [coal],
        'hydro': [hydro],
        'solar': [solar],
        'india_bheramara_hvdc': [india_bheramara],
        'india_tripura': [india_tripura],
        'hour': [hour],
        'day': [day],
        'month': [month],
        'year': [year],
        'total_india_import': [total_india_import],
        'supply_demand_gap': [supply_demand_gap]
    })
    
    # Scaling the features
    features_to_scale = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 
                         'india_bheramara_hvdc', 'india_tripura', 'hour', 
                         'day', 'month', 'year', 'total_india_import',
                         'supply_demand_gap']
    
    input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])
    
    # Making prediction
    prediction = model.predict(input_data)[0]
    
    # Displaying result
    st.success(f"### Predicted Power Demand: {prediction:,.2f} MW")

# Footer
st.markdown("---")
st.caption("Model: Gradient Boosting Regressor | Student ID: 00015972")
