import lightgbm as lgb
import logging
import numpy as np
from typing import Tuple
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

class CreditRiskModel:
    def __init__(self, random_state: int = 42):
        self.model = lgb.LGBMClassifier(random_state=random_state)
        self.is_trained = False

    def train(self, X_train, y_train):
        """Trains the LightGBM classifier."""
        logger.info("Training LightGBM model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def evaluate(self, X_test, y_test):
        """Evaluates model performance."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Model Accuracy: {acc:.4f}")
        return acc, report

    def predict_proba(self, X) -> np.ndarray:
        """Predicts probability of default."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        
        return self.model.predict_proba(X)

    def get_calibration_info(self, X_test, y_test, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates probability calibration curve."""
        y_prob = self.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=bins)
        return prob_true, prob_pred
