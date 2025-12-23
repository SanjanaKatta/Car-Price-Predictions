import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('models')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle


class CAR_PRICE_MODEL:
    def train_linear_regression(X_train, y_train, X_test, y_test):
        try:
            logger.info("Starting Linear Regression Training")
            logger.info(f'{X_train.isnull().sum()}')

            reg_lr = LinearRegression()
            reg_lr.fit(X_train, y_train)
            y_train_pred = reg_lr.predict(X_train)
            y_test_pred = reg_lr.predict(X_test)

            train_data_r2_score = r2_score(y_train, y_train_pred)
            test_data_r2_score = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            logger.info(f"Train R2 Score : {train_data_r2_score}")
            logger.info(f"Test R2 Score  : {test_data_r2_score}")
            logger.info(f"MAE            : {mae}")
            logger.info(f"RMSE           : {rmse}")

            with open('linear_regression_model.pkl', 'wb') as f:
                pickle.dump(reg_lr, f)

            logger.info("Linear Regression model saved successfully")

            return reg_lr

            '''
            # Example car input (MUST MATCH feature order)
            # Example:
            # Year, Present_Price, Kms_Driven, Owner,
            # Fuel_Type_Diesel, Seller_Type_Individual, Transmission_Manual
            sample_input = np.array([[2017, 8.5, 45000, 0, 1, 0, 1]])

            sample_scaled = scaler.transform(sample_input)
            predicted_price = model.predict(sample_scaled)

            logger.info(f"Sample Predicted Car Price : ₹{predicted_price[0]:.2f} Lakhs")
            '''
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
