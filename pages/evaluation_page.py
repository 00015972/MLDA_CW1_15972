import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Evaluation")

st.title("Model Evaluation")

# Model Comparison
st.header("Model Comparison")

results_data = {
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MSE': [128638.17, 85402.00, 87777.93],
    'MAE': [135.88, 56.16, 71.44],
    'R² Score': [0.9809, 0.9873, 0.9870]
}
results_df = pd.DataFrame(results_data)
st.dataframe(results_df)

# Explanation
st.header("Analysis")

st.subheader("Best Model Selection")
st.success("""
**Random Forest** is selected as the best model based on R² Score:
- **R² Score:** 0.9873 (explains 98.73% of variance)
- **MSE:** 85,402.00
- **MAE:** 56.16 MW average error
""")

# R² Score Comparison Chart
st.header("R² Score Comparison")
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
bars = ax.bar(results_df['Model'], results_df['R² Score'], color=colors)
ax.set_ylabel('R² Score')
ax.set_title('Model Performance Comparison')
ax.set_ylim(0.97, 1.0)
for bar, score in zip(bars, results_df['R² Score']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
            f'{score:.4f}', ha='center', va='bottom')
plt.tight_layout()
st.pyplot(fig)

st.caption("Student ID: 00015972")
