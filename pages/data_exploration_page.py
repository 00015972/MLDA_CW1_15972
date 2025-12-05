import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Exploration")

st.title("Data Exploration")

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/PGCB_date_power_demand.csv', sep=';')
    return df

df = load_data()

# Dataset Shape
st.header("Dataset Shape")
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# Dataset Preview
st.subheader("Data Preview")
st.dataframe(df.head(10))

# Statistical Summary
st.header("Statistical Summary")
st.dataframe(df.describe().round(2))

# Correlation Matrix
st.header("Correlation Matrix")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
plt.title('Correlation Matrix')
plt.tight_layout()
st.pyplot(fig)

# Distribution plots
st.header("Feature Distributions")
selected_feature = st.selectbox("Select feature to visualize:", numerical_cols)

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.hist(df[selected_feature].dropna(), bins=30, edgecolor='black')
    ax1.set_title(f'Distribution of {selected_feature}')
    ax1.set_xlabel(selected_feature)
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.boxplot(df[selected_feature].dropna())
    ax2.set_title(f'Box Plot of {selected_feature}')
    ax2.set_ylabel(selected_feature)
    st.pyplot(fig2)

st.caption("Student ID: 00015972")
