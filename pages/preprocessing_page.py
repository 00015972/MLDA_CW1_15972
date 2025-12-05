import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Preprocessing")

st.title("Data Preprocessing")

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/PGCB_date_power_demand.csv', sep=';')
    return df

df = load_data()

# Missing Values Section
st.header("1. Missing Values Analysis")

missing_data = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})
st.dataframe(missing_data)

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(missing_data.index, missing_data['Missing %'])
plt.xticks(rotation=45, ha='right')
ax.set_xlabel('Columns')
ax.set_ylabel('Missing Values (%)')
ax.set_title('Missing Values Percentage by Column')
plt.tight_layout()
st.pyplot(fig)

# Handling Missing Values
st.header("2. Handling Missing Values")
st.markdown("""
**Actions taken:**
- **Dropped columns** with >70% missing values: `wind`, `india_adani`, `nepal`, `remarks`
- **Filled `solar`** with 0 (assuming no recorded solar means 0 MW generation)
""")

# Outlier Detection
st.header("3. Outlier Detection (IQR Method)")
st.markdown("""
**IQR Formula:**
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Lower bound = Q1 - 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR
""")

# Showing outliers for cleaned data
df_clean = df.drop(columns=['wind', 'india_adani', 'nepal', 'remarks'])
df_clean['solar'] = df_clean['solar'].fillna(0)
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

outliers_data = []
for col in numerical_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    outliers_data.append({
        'Column': col,
        'Outliers Count': len(outliers),
        'Lower Bound': round(lower, 2),
        'Upper Bound': round(upper, 2)
    })

st.dataframe(pd.DataFrame(outliers_data))


# Feature Engineering
st.header("4. Feature Engineering")
st.markdown("""
**New features created:**
| Feature | Description |
|---------|-------------|
| `hour` | Hour extracted from datetime |
| `day` | Day extracted from datetime |
| `month` | Month extracted from datetime |
| `year` | Year extracted from datetime |
""")

# Train-Test Split
st.header("5. Train-Test Split")
st.markdown("""
- **Training set:** 80% (74,120 samples)
- **Test set:** 20% (18,530 samples)
- **Random state:** 42 (for reproducibility)
""")

st.caption("Student ID: 00015972")
