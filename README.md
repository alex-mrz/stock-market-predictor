# Stock Market Predictor 📈

This project is a machine learning pipeline that attempts to forecast stock price trends using two models:

- A traditional supervised learning model with **scikit-learn**
- A gradient boosting model with **XGBoost**

The goal is to experiment with financial data, evaluate accuracy, and understand the strengths and limitations of common ML techniques in a market context.

---

## 🧠 What It Does

- Loads historical stock data
- Calculates technical indicators (e.g. EMA, RSI)
- Trains classification models to predict next-week trend (up/down)
- Compares model accuracy and basic feature impact
- Simulates trades with a fixed capital, leverage, risk-adjusted TP and SL

---

## 🛠️ Tech Used

- Python  
- Pandas, NumPy  
- scikit-learn  
- XGBoost  
- Matplotlib (optional for plots)

---

## 🚀 How to Run

1. Clone this repo  
2. Install requirements:  
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```
3. Run the notebook or script to train and test models

---

## 📚 What I Learned

- How to preprocess time series data  
- How to tune and compare different ML models  
- Limitations of ML in financial forecasting
