# Hybrid LSTM-SVM Weather Forecasting

## Overview
This project focuses on weather forecasting using a hybrid machine learning approach that combines Long Short-Term Memory (LSTM) networks and Support Vector Machines (SVM). By leveraging the temporal sequence learning capabilities of LSTM and the robust regression capabilities of SVM, the model aims to accurately predict weather conditions, specifically focusing on rainfall and other meteorological variables.

## Dataset
The dataset used in this project is historical daily weather data (`Data Capstone Fix BGt.xlsx` / `.csv`). It contains the following features:
- **Kelembapan Rata-Rata Udara Harian (%)** - Average Daily Air Humidity
- **Temperatur Rata-Rata Harian** - Average Daily Temperature
- **Temperatur Maksimum Harian** - Maximum Daily Temperature
- **Temperature Minimum Harian** - Minimum Daily Temperature
- **Angin Rata-Rata Harian (Knot)** - Average Daily Wind Speed
- **Curah Hujan** - Rainfall
- **Penyinaran Harian (%)** - Daily Sunshine Duration
- **Penguapan Harian (mm)** - Daily Evaporation
- **Tekanan Udara Harian (mb)** - Daily Air Pressure

## Methodology
The project explores and compares different predictive models:
1.  **Exploratory Data Analysis (EDA):** Visualizing historical trends, distributions (e.g., violin plots for rainfall), and performing autocorrelation (ACF/PACF) analysis to understand time-series dependencies.
2.  **Long Short-Term Memory (LSTM):** A deep learning model well-suited for capturing long-term dependencies in time-series weather data.
3.  **Support Vector Regression (SVR):** A machine learning algorithm used to capture non-linear relationships in the data.
4.  **Hybrid LSTM-SVM:** Combining the strengths of both models to improve forecasting accuracy.

## Project Structure
- `comparison-model.ipynb`: The main Jupyter Notebook containing the data loading, preprocessing, EDA, model training, and evaluation.
- `comparison-model no feature column.ipynb`: An alternative notebook for model comparison without specific feature columns.
- `Data Capstone Fix BGt.xlsx` / `Data Capstone Fix BGt.csv`: The datasets used for training and testing.
- `verse 1/`: Additional project files or iterations.

## Prerequisites
To run the notebooks in this project, you need the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `tensorflow`
- `keras-tuner`
- `optuna`
- `pywt` (PyWavelets)

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow keras-tuner optuna PyWavelets openpyxl
```

## Usage
1. Clone the repository or download the project files.
2. Ensure you have the required dependencies installed.
3. Open `comparison-model.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells sequentially to observe the data analysis, model training process, and evaluation results.
