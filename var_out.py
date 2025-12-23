import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('var_out')

from scipy import stats

class VT_OUT:
    def variable_transformation_outliers(X_train_num,X_test_num):
        try:
            logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
            logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")

            logger.info('Before Variable Transformation : ')
            logger.info(f'{X_train_num.head}')
            logger.info(f'{X_test_num.head}')

            PLOT_PATH = "plots_path"
            for i in X_train_num.columns:
                plt.figure()
                X_train_num[i].plot(kind='kde', color='r')
                plt.title(f'KDE-{i}')
                plt.savefig(f'{PLOT_PATH}/kde_{i}.png')
                plt.close()
            for i in X_train_num.columns:
                plt.figure()
                sns.boxplot(x=X_train_num[i])
                plt.title(f'Boxplot-{i}')
                plt.savefig(f'{PLOT_PATH}/boxplot_{i}.png')
                plt.close()

            # ================= TRANSFORMATION + CAPPING =================
            for col in X_train_num.columns:
                # 1️⃣ Log transform (BEST for calories-like features)
                X_train_num[col] = np.log1p(X_train_num[col])
                X_test_num[col] = np.log1p(X_test_num[col])

                # 2️⃣ Quantile capping (from train only)
                lower = X_train_num[col].quantile(0.01)
                upper = X_train_num[col].quantile(0.99)

                X_train_num[col] = X_train_num[col].clip(lower, upper)
                X_test_num[col] = X_test_num[col].clip(lower, upper)

            logger.info(f'After transform {X_train_num.shape}')
            logger.info(f'After transform {X_test_num.shape}')

            logger.info('After Variable Transformation : ')

            for i in X_train_num.columns:
                plt.figure()
                X_train_num[i].plot(kind='kde', color='r')
                plt.title(f'KDE-{i}')
                plt.savefig(f'{PLOT_PATH}/kde_{i}.png')
                plt.close()
            for i in X_train_num.columns:
                plt.figure()
                sns.boxplot(x=X_train_num[i])
                plt.title(f'Boxplot-{i}')
                plt.savefig(f'{PLOT_PATH}/boxplot_{i}.png')
                plt.close()

            logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
            logger.info(f'{X_train_num.head}')
            logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")
            logger.info(f'{X_test_num.head}')

            return X_train_num, X_test_num

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

