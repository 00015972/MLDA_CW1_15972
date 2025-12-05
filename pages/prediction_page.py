import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Prediction")

st.title("Power Demand Prediction")
st.write("Enter the power generation values to predict electricity demand")

# Train model from dataset (since model files are not in repo)
@st.cache_resource
def load_model():
    # Load dataset
    df = pd.read_csv('dataset/PGCB_date_power_demand.csv', sep=';')
    
    # Feature engineering (same as notebook)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    # Features and target
    feature_cols = ['generation_mw', 'load_shedding', 'gas', 'liquid_fuel', 'coal', 
                    'hydro', 'solar', 'india_bheramara_hvdc', 'india_tripura',
                    'hour', 'day', 'month', 'year']
    X = df[feature_cols]
    y = df['demand_mw']
    
    # Scale features
    features_to_scale = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 
                         'india_bheramara_hvdc', 'india_tripura', 'hour', 
                         'day', 'month', 'year']
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])
    
    # Train Random Forest (best model from evaluation)
    model = RandomForestRegressor(max_depth=20, min_samples_split=2, n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
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

col3, col4 = st.columns(2)
with col3:
    day = st.slider("Day of Month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
with col4:
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)

# Prediction button
if st.button("Predict Demand", type="primary"):
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
        'year': [year]
    })
    
    # Scale features
    features_to_scale = ['gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 
                         'india_bheramara_hvdc', 'india_tripura', 'hour', 
                         'day', 'month', 'year']
    
    input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])
    
    # Making prediction
    prediction = model.predict(input_data)[0]
    
    # Displaying result
    st.success(f"### Predicted Power Demand: {prediction:,.2f} MW")

st.caption("Model: Random Forest Regressor | Student ID: 00015972")
