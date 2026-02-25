import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)

class CreditPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering to the data."""
        logger.info("Applying feature engineering...")
        processed_df = df.copy()
        
        # Debt-to-income ratio
        processed_df['debt_to_income_ratio'] = processed_df['total_debt_outstanding'] / processed_df['income']
        
        # Drop columns not used for training if present
        if 'customer_id' in processed_df.columns:
            processed_df = processed_df.drop(columns=['customer_id'])
            
        return processed_df

    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        """Fits the scaler and applies SMOTE for resampling."""
        logger.info("Fitting preprocessor and resampling data...")
        self.feature_columns = X.columns
        
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        self.scaler.fit(X_res)
        X_scaled = self.scaler.transform(X_res)
        self.is_fitted = True
        
        return X_scaled, y_res

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms new features for inference."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transformation.")
        
        # Ensure correct column order
        X = X[self.feature_columns]
        return self.scaler.transform(X)
