import pytest
import pandas as pd
import numpy as np
from credit_risk.preprocessor import CreditPreprocessor
from credit_risk.risk_calculator import RiskCalculator
from unittest.mock import MagicMock

def test_preprocessor_transformation():
    preprocessor = CreditPreprocessor()
    df = pd.DataFrame({
        'total_debt_outstanding': [1000, 2000],
        'income': [50000, 60000],
        'customer_id': [1, 2]
    })
    
    transformed = preprocessor.transform_features(df)
    
    assert 'debt_to_income_ratio' in transformed.columns
    assert 'customer_id' not in transformed.columns
    assert transformed.loc[0, 'debt_to_income_ratio'] == 1000 / 50000

def test_risk_calculator():
    # Mock model and preprocessor
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]]) # 10% PD
    
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform_features.return_value = pd.DataFrame([{'val': 1}])
    mock_preprocessor.transform.return_value = np.array([[1]])
    
    calculator = RiskCalculator(mock_model, mock_preprocessor)
    
    profile = {
        "income": 50000,
        "total_debt_outstanding": 5000,
        "loan_amt_outstanding": 10000
    }
    
    results = calculator.calculate_risk_profile(profile)
    
    assert results['probability_of_default'] == 0.1
    assert results['risk_level'] == "Low Risk"
    # EL = 0.1 * 10000 * 0.9 = 900
    assert results['expected_loss'] == 900.0
