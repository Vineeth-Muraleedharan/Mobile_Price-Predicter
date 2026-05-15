# 📱 Mobile Price Predictor

A machine learning web app that predicts smartphone prices based on technical specifications using multiple regression models.

**🔗 Live App:** [Click to Open](https://mobileprice-predicter.streamlit.app/)

---

## 📌 Project Overview

This project is part of an end-to-end ML assignment covering:
- Exploratory Data Analysis (EDA)
- Feature Engineering & Transformation
- Outlier Detection & Treatment
- Regression Modelling with Regularization
- Model Comparison & Evaluation

---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| Linear Regression (OLS) | Baseline model — no regularization |
| Ridge Regression (L2) | Shrinks coefficients, handles multicollinearity |
| Lasso Regression (L1) | Automatic feature selection via zeroing |
| ElasticNet | Combines Ridge + Lasso penalties |

---

## 📊 App Features

- 🎛️ Adjust phone specs via interactive sliders
- 💰 Live price prediction
- 📈 Actual vs Predicted chart
- 🔍 Feature importance (coefficients)
- 📊 Model comparison — R², RMSE, MAE
- ✅ Overfitting check

---

## 📁 Project Structure

```
mobile-price-predictor/
│
├── app.py               ← Streamlit app
├── requirements.txt     ← Dependencies
├── Cellphone.csv        ← Dataset
└── README.md            ← You are here
```

---

## 📦 Dataset

- **Source:** [Kaggle — Mobile Price Prediction]([https://www.kaggle.com/datasets/mohannapd/mobile-price-prediction/data])
- **Features:** RAM, Battery, Camera, CPU, Internal Memory, etc.
- **Target:** Price (USD)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.2-green)

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/mobile-price-predictor.git
cd mobile-price-predictor
pip install -r requirements.txt
streamlit run app.py
```
