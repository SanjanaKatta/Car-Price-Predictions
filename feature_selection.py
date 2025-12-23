import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

import logging
import os

from log_code import setup_logging
logger = setup_logging('feature_selection')

from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold

# Constant & Quasi-constant
reg_con = VarianceThreshold(threshold=0.0)
reg_quasi = VarianceThreshold(threshold=0.01)

class FEATURE_SELECTION:
    @staticmethod
    def complete_feature_selection(X_train, X_test, y_train):
        try:
            logger.info(f"Initial Train Shape : {X_train.shape}")
            logger.info(f"Initial Test Shape  : {X_test.shape}")

            # ================= CONSTANT FEATURES =================
            reg_con.fit(X_train)

            keep_cols = X_train.columns[reg_con.get_support()]
            logger.info(f"Constant Removed : {X_train.columns[~reg_con.get_support()]}")

            X_train = X_train[keep_cols]
            X_test = X_test[keep_cols]

            # ================= QUASI-CONSTANT FEATURES =================
            reg_quasi.fit(X_train)

            keep_cols = X_train.columns[reg_quasi.get_support()]
            logger.info(f"Quasi-Constant Removed : {X_train.columns[~reg_quasi.get_support()]}")

            X_train = X_train[keep_cols]
            X_test = X_test[keep_cols]

            # ================= CORRELATION (Hypothesis Testing) =================
            drop_cols = []

            for col in X_train.columns:
                corr, p_value = pearsonr(X_train[col], y_train)
                if p_value > 0.05:
                    drop_cols.append(col)

            logger.info(f"Correlation Dropped Columns : {drop_cols}")

            X_train = X_train.drop(columns=drop_cols)
            X_test = X_test.drop(columns=drop_cols)

            logger.info(f"Final Train Shape : {X_train.shape}")
            logger.info(f'{X_train.head}')
            logger.info(f"Final Test Shape  : {X_test.shape}")
            logger.info(f'{X_test.head}')

            return X_train, X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
