import streamlit as st

st.set_page_config(page_title="Power Demand Prediction App")

st.title("Power Demand Prediction App")
st.write("### Student ID: 00015972")

st.markdown("---")

st.markdown("""


### Pages

| Page | Description |
|------|-------------|
| Data Exploration | View dataset statistics, correlations, and distributions |
|  Preprocessing | See data cleaning and feature engineering steps |
|  Prediction | Make predictions with custom input values |
|  Evaluation | Compare model performance metrics |

---
""")

st.info("Please navigate to a page from the sidebar")