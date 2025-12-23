# 🚗 Car Price Prediction – Machine Learning Project

A **web-based machine learning project** that predicts the resale value of cars based on various features. This project demonstrates an end-to-end ML workflow including **data preprocessing, feature engineering, model training, evaluation, and deployment** with a **user-friendly web interface**.

---

## 📌 Project Overview

Car price prediction is a **regression problem** commonly used in automotive analytics, resale valuation, and dealership pricing. The model estimates car prices based on factors such as:

- Year of manufacture  
- Brand  
- Present price  
- Kilometers driven  
- Owner count  
- Fuel type (Petrol/Diesel)  
- Seller type (Dealer/Individual)  
- Transmission type (Manual/Automatic)

The goal is to help users, dealerships, and platforms **estimate car resale value accurately**.

---

## 🧠 ML Pipeline Architecture

### 1. Data Collection
- Collected car-related datasets containing features and selling prices.

### 2. Data Cleaning
- Handling missing values  
- Removing duplicate entries  
- Correcting data inconsistencies

### 3. Feature Engineering
- Encode categorical variables: Brand, Fuel Type, Seller Type, Transmission  
- One-hot encoding for brands and other categorical features  
- Feature scaling (optional, depending on model)

### 4. Exploratory Data Analysis (EDA)
- Analyzing distributions of features  
- Visualizing correlations  
- Identifying trends and outliers

### 5. Train-Test Split
- Split dataset into **training** and **testing** sets  
- Ensured proper representation of all categories

### 6. Model Training
- Trained multiple regression models:  
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  

### 7. Model Evaluation
- Evaluated using metrics:  
  - R² Score  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  

### 8. Hyperparameter Tuning
- Applied to improve model performance and reduce overfitting

### 9. Best Model Selection
- Selected the model with highest accuracy and lowest error

### 10. Model Saving
- Saved the final trained model for deployment using **Pickle**

### 11. Prediction & Deployment
- Integrated model with a **Flask-based web app**  
- User-friendly UI to input car details and get predicted price

---

## ⚙️ Technologies Used

**Programming Language:** Python  

**Libraries:**  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Flask (for deployment)  

**Model Type:** Regression  

**Model Storage:** Pickle (.pkl)  

---

## 🧪 Dataset Description

**Features:**
- Year  
- Brand  
- Present Price  
- Kilometers Driven  
- Owner  
- Fuel Type  
- Seller Type  
- Transmission  

**Target Variable:**
- Selling Price / Resale Price

---

## 🔧 Feature Engineering

- Handle missing values  
- Encode categorical features (Brand, Fuel Type, Seller Type, Transmission)  
- One-hot encoding for categorical variables  
- Feature scaling using **StandardScaler**  

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
