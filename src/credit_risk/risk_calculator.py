import logging
import pandas as pd
from .config import settings

logger = logging.getLogger(__name__)

class RiskCalculator:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def calculate_risk_profile(self, customer_profile: dict):
        """Calculates all risk metrics for a given customer profile."""
        # Convert to DataFrame
        customer_df = pd.DataFrame([customer_profile])
        
        # Apply transformation
        customer_processed = self.preprocessor.transform_features(customer_df)
        customer_scaled = self.preprocessor.transform(customer_processed)
        
        # Probability of default
        pd_value = self.model.predict_proba(customer_scaled)[0][1]
        
        # Risk level
        risk_level = self._get_risk_level(pd_value)
        
        # Expected Loss
        el = self._calculate_expected_loss(
            customer_profile.get('loan_amt_outstanding', 0),
            pd_value
        )
        
        return {
            "probability_of_default": pd_value,
            "risk_level": risk_level,
            "expected_loss": el
        }

    def _get_risk_level(self, pd: float) -> str:
        if pd < settings.LOW_RISK_THRESHOLD:
            return "Low Risk"
        elif pd < settings.MEDIUM_RISK_THRESHOLD:
            return "Medium Risk"
        return "High Risk"

    def _calculate_expected_loss(self, loan_amt: float, pd: float) -> float:
        # LGD = (1 - recovery_rate)
        lgd = 1 - settings.DEFAULT_RECOVERY_RATE
        return pd * loan_amt * lgd
