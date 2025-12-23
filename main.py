import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import category_encoders as ce

import warnings
warnings.filterwarnings('ignore')

import pickle

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from random_sample import RSITechnique
from var_out import VT_OUT
#from feature_selection import FEATURE_SELECTION
from categorical_to_numerical import CategoricalToNumerical
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imbalanced_data import SCALE_DATA
from models import CAR_PRICE_MODEL

class CAR_PRICE_PREDICTOR:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(path)

            logger.info('Data loaded')
            logger.info(f'{self.df.shape}')
            logger.info(f'{self.df.head()}')
            #Dependent column
            self.y = self.df['Selling_Price']
            # Independent columns (features)
            self.X = self.df.drop(['Selling_Price'], axis=1)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)
            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')
            logger.info(f'{self.X_train['Car_Name'].unique()}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Training Data : {self.X_train.head(10)}')
            logger.info(f'Testing data size : {self.X_test.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            logger.info(f'Missing Values')
            logger.info(f"X_train columns: {self.X_train.columns}")
            logger.info(f"X_test columns: {self.X_test.columns}")

            if self.X_train.isnull().sum().any() > 0 or self.X_test.isnull().sum().any() > 0:
                self.X_train, self.X_test = RSITechnique.random_sample_imputation_technique(self.X_train, self.X_test)
            else:
                logger.info(f'No Missing Values')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def VarTrasform_Outliers(self):
        try:
            logger.info(f'Variable Transform Outliers Columns')
            logger.info(f"X_train columns: {self.X_train.columns}")
            logger.info(f"X_test columns: {self.X_test.columns}")

            self.X_train_num = self.X_train.select_dtypes(exclude = 'object')
            self.X_train_cat = self.X_train.select_dtypes(include = 'object')
            self.X_test_num = self.X_test.select_dtypes(exclude = 'object')
            self.X_test_cat = self.X_test.select_dtypes(include = 'object')

            logger.info(f'{self.X_train_num.columns}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_test_cat.shape}')

            self.X_train_num, self.X_test_num = VT_OUT.variable_transformation_outliers(self.X_train_num, self.X_test_num)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    '''
    def fs(self):
        try:
            logger.info(f'Feature Selection')
            logger.info(f" Before : {self.training_data.columns} -> {self.training_data.shape}")
            logger.info(f"Before : {self.testing_data.columns} -> {self.testing_data.shape}")

            self.X_train, self.X_test = FEATURE_SELECTION.complete_feature_selection(self.training_data,self.testing_data,self.y_train)

            logger.info(f" After : {self.training_data.columns} -> {self.training_data.shape}")
            logger.info(f"After : {self.testing_data.columns} -> {self.testing_data.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')    
    '''
    def feature_engineering(self):
        try:
            logger.info('Starting feature engineering (Brand extraction)')
            def extract_brand(car_name):
                if pd.isna(car_name):
                    return 'Unknown'
                name = car_name.strip().lower()
                multi_word_brands = [
                    'royal enfield',
                    'hero honda'
                ]
                for brand in multi_word_brands:
                    if name.startswith(brand):
                        return brand.title()
                return name.split()[0].title()

            # Create Brand column
            self.X_train_cat['Brand'] = self.X_train_cat['Car_Name'].apply(extract_brand)
            self.X_test_cat['Brand'] = self.X_test_cat['Car_Name'].apply(extract_brand)
            # Drop high-cardinality column
            #self.X_train_cat.drop(['Car_Name'], axis=1, inplace=True)
            #self.X_test_cat.drop(['Car_Name'], axis=1, inplace=True)
            logger.info('Brand extraction completed')
            logger.info(f'X_train columns after feature engineering: {self.X_train_cat.columns}')
            logger.info(f'X_test columns after feature engineering: {self.X_test_cat.columns}')
            logger.info(f'\n{self.X_train_cat.head(10)}')
            logger.info(f'\n{self.X_test_cat}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def cat_to_num(self):
        try:
            logger.info('Categorical to Numerical Conversion Started')
            logger.info(f'{self.X_train_cat.columns}')

            # Encode TRAIN
            self.X_train, self.brand_map, self.global_mean = CategoricalToNumerical.cat_to_num_train(self.X_train, self.y_train)

            self.X_test = CategoricalToNumerical.cat_to_num_test(self.X_test, self.brand_map, self.global_mean)

            logger.info("Brand map and global mean saved")

            logger.info(f'After encoding, X_train shape: {self.X_train.shape}')
            logger.info(f'After encoding, X_test shape: {self.X_test.shape}')

            # Check types to confirm everything is numeric
            logger.info(f'X_train dtypes:{self.X_train.dtypes}')
            logger.info(f'{self.X_train.isnull().sum()}')
            logger.info(f'{self.X_test.isnull().sum()}')

            logger.info("Categorical to Numerical Conversion Completed Successfully")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def data_scaling(self):
        try:
            logger.info('Scaling Data Before Regression')

            self.X_train, self.X_test = SCALE_DATA.scale(self.X_train, self.X_test, save_path="scaler.pkl")
            logger.info('Scaling Completed')
            logger.info(f'X_train columns : {self.X_train.columns}')
            logger.info(f'X_test columns: {self.X_test.columns}')
            logger.info(f'X_train : {self.X_train.shape}')
            logger.info(f'X_test : {self.X_test.shape}')
            logger.info(self.X_train.head(4))
            logger.info(self.X_test.head(4))
            logger.info(f'{self.X_train.isnull().sum()}')
            logger.info(f'{self.X_test.isnull().sum()}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def model_training(self):
        try:
            logger.info('Training Model Started')
            CAR_PRICE_MODEL.train_linear_regression(self.X_train, self.y_train, self.X_test, self.y_test)
            logger.info('Training Model Completed')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


if __name__ == "__main__":
    try:
        obj = CAR_PRICE_PREDICTOR('C:\\Users\\Rajesh\\Downloads\\Mini Projects\\Predict Car Values\\car price.csv')
        obj.missing_values()
        obj.VarTrasform_Outliers()
        obj.feature_engineering()
        obj.cat_to_num()
        #obj.fs()
        obj.data_scaling()
        obj.model_training()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

