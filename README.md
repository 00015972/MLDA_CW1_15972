# Power Demand Prediction

Student ID: 00015972

## Project Structure

```
MLDA_CW1_15972/
├── app.py                  # Main Streamlit app (home page)
├── pages/                  # Streamlit pages
│   ├── data_exploration_page.py
│   ├── preprocessing_page.py
│   ├── prediction_page.py
│   └── evaluation_page.py
├── models/                 # Saved models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── dataset/
│   └── PGCB_date_power_demand.csv
├── MLDA_CW1_15972.ipynb    # Jupyter notebook with analysis
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository or download the project files

2. Create a virtual environment:
```
python -m venv .venv
.venv\Scripts\activate
```

3. Install required packages:
```
pip install -r requirements.txt
```

## Requirements

- Python 3.10 or higher
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- joblib
- streamlit

## How to Run

### Jupyter Notebook

Open and run `MLDA_CW1_15972.ipynb` to see the full analysis including:
- Data exploration
- Data preprocessing
- Model training
- Model evaluation

### Streamlit App

Run the following command in the terminal:
```
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

### Streamlit Pages

- Data Exploration: View dataset statistics and visualizations
- Preprocessing: See data cleaning and feature engineering steps
- Prediction: Enter values to predict power demand
- Evaluation: Compare model performance metrics