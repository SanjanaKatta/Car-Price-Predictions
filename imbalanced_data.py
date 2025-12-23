import pandas as pd
import pickle
import sys
from sklearn.preprocessing import StandardScaler
from log_code import setup_logging

logger = setup_logging('scaling')

class SCALE_DATA:
    def scale(X_train, X_test, save_path="scaler.pkl"):
        try:
            logger.info("Selective Feature Scaling Started")
            scaler = StandardScaler()

            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)

            X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index )

            with open(save_path, "wb") as f:
                pickle.dump(scaler, f)

            logger.info("Feature scaling completed and scaler saved")
            logger.info(f"Train Shape: {X_train_scaled.shape}")
            logger.info(f'{X_train_scaled.head(10)}')
            logger.info(f"Test Shape : {X_test_scaled.shape}")
            logger.info(f'{X_test_scaled.head(10)}')

            return X_train_scaled, X_test_scaled

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in Line no {error_line.tb_lineno}: {error_msg}"
            )
